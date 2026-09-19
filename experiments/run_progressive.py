"""Train a 256-bit quantizer and validate an optional compact routing exit.

Reuses pinned public embeddings. No base-model fine-tuning or test-set tuning.
"""

import argparse
import hashlib
import json
import platform
import statistics
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from context_stamps.learning import fit_family
from context_stamps.routing import ProgressiveRouter, fit_routing_policy
from experiments.run_spherical_public import agreement, ndcg, qrels, read, save, sha
from stamps import Family, _planes, stamp_vector

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'evidence/progressive-v1'


def binary(matrix, family):
    values = matrix.astype(np.float64)
    if family.mean:
        values -= np.array(family.mean)
    codes = np.packbits(values @ np.array(_planes(family)).T >= 0, axis=1, bitorder='little')
    for i in range(min(2, len(matrix))):
        assert int.from_bytes(codes[i].tobytes(), 'little') == stamp_vector(matrix[i], family).value
    return codes


def main(args):
    protocol = read(OUT / 'protocol.json')
    prior = read(ROOT / 'evidence/spherical-public-v1/manifest.json')
    split = read(ROOT / 'evidence/scifact-v1/manifest.json')
    metadata = prior['datasets'][0]
    key = hashlib.sha256(json.dumps([metadata['data_sha256'], prior['protocol']['encoder'],
                                     prior['protocol']['revision'], metadata['query_ids']], sort_keys=True).encode()).hexdigest()
    queries = np.load(args.scifact_cache / (key + '-Q.npy'), allow_pickle=False)
    positions = {qid: i for i, qid in enumerate(metadata['query_ids'])}
    train = queries[[positions[qid] for qid in split['training_queries']]]
    started = time.perf_counter()
    learned = fit_family(train, encoder='minilm-pinned', bits=256, method='itq', seed=17, iterations=50)
    training_seconds = time.perf_counter() - started
    (OUT / 'itq256.json').write_text(learned.to_json() + '\n', encoding='utf-8', newline='\n')
    families = {f'gaussian_{bits}': Family('minilm-pinned', 384, bits, 17) for bits in (128, 256, 512)}
    families['trained_itq_256'] = learned
    records, summaries, calibration = [], [], {}
    policy = None
    router = ProgressiveRouter()
    for metadata in prior['datasets']:
        name = metadata['dataset']
        source = args.data / name
        assert all(sha(source / p) == digest for p, digest in metadata['data_sha256'].items())
        ids = [json.loads(line)['_id'] for line in (source / 'corpus.jsonl').read_text(encoding='utf-8').splitlines()]
        positions = {qid: i for i, qid in enumerate(metadata['query_ids'])}
        key = hashlib.sha256(json.dumps([metadata['data_sha256'], prior['protocol']['encoder'],
                                         prior['protocol']['revision'], metadata['query_ids']], sort_keys=True).encode()).hexdigest()
        cache = args.scifact_cache if name == 'scifact' else args.replication_cache
        dpath, qpath = cache / (key + '-D.npy'), cache / (key + '-Q.npy')
        assert sha(dpath) == metadata['embedding_sha256']['documents']
        assert sha(qpath) == metadata['embedding_sha256']['queries']
        documents, queries = np.load(dpath, allow_pickle=False), np.load(qpath, allow_pickle=False)
        codes = {method: (binary(documents, family), binary(queries, family)) for method, family in families.items()}
        doc_positions = {key: i for i, key in enumerate(ids)}

        def ranks(qid, method):
            index = positions[qid]
            if method == 'dense':
                scores = np.einsum('ij,j->i', documents, queries[index].astype(np.float64), dtype=np.float64)
            else:
                dc, qc = codes[method]
                scores = agreement(qc[index], dc, families[method].bits)
            if qid in doc_positions:
                scores[doc_positions[qid]] = -np.inf
            order = np.argsort(-scores, kind='stable')[:11].tolist()
            return order, scores

        if name == 'scifact':
            for partition, query_ids in (('train', split['training_queries']), ('validation', split['validation_queries'])):
                rows = []
                for qid in query_ids:
                    order, scores = ranks(qid, 'trained_itq_256')
                    dense, _ = ranks(qid, 'dense')
                    rows.append({'query_id': qid, 'score': float(scores[order[9]]),
                                 'margin': float(scores[order[9]] - scores[order[10]]),
                                 'correct': order[:10] == dense[:10]})
                calibration[partition] = rows
            policy, report = fit_routing_policy(calibration['train'], calibration['validation'],
                                                scope='scifact:minilm-pinned:itq256:v1')
            save(OUT / 'policy.json', {'policy': asdict(policy), 'report': report})
            print('Calibration', report, flush=True)
        labels = qrels(source / 'qrels/test.tsv')
        for qid in sorted(labels):
            dense_order, _ = ranks(qid, 'dense')
            for method in ('dense', *families):
                order, _ = ranks(qid, method)
                records.append({'dataset': name, 'query_id': qid, 'method': method,
                                'ranked_ids': [ids[i] for i in order[:10]],
                                'ndcg10': ndcg(order[:10], labels[qid], ids),
                                'same_order_as_dense': order[:10] == dense_order[:10]})
            eligible = tuple(key for key in ids if key != qid)
            def callback(method):
                def run(allowed, count):
                    order, scores = ranks(qid, method)
                    return [(ids[i], float(scores[i])) for i in order[:count]]
                return run
            started = time.perf_counter_ns()
            result = router.search(eligible=eligible, precise=callback('dense'), compact=callback('trained_itq_256'),
                                   policy=policy, scope=f'{name}:minilm-pinned:itq256:v1')
            milliseconds = (time.perf_counter_ns() - started) / 1e6
            order = [doc_positions[key] for key in result['ids']]
            records.append({'dataset': name, 'query_id': qid, 'method': 'progressive',
                            'ranked_ids': result['ids'], 'ndcg10': ndcg(order, labels[qid], ids),
                            'same_order_as_dense': order == dense_order[:10], 'route': result['route'],
                            'compact_attempted': result['compact_attempted'], 'milliseconds': milliseconds})
        for method in ('dense', *families, 'progressive'):
            rows = [r for r in records if r['dataset'] == name and r['method'] == method]
            summaries.append({'dataset': name, 'method': method, 'queries': len(rows),
                              'ndcg10': statistics.mean(r['ndcg10'] for r in rows),
                              'same_order_fraction': statistics.mean(r['same_order_as_dense'] for r in rows)})
        print(name, summaries[-6:], flush=True)
    for filename, value in (('results', records), ('summary', summaries), ('calibration', calibration)):
        save(OUT / (filename + '.json'), value)
    save(OUT / 'manifest.json', {'protocol': protocol, 'numpy': np.__version__, 'platform': platform.platform(),
                                'training_seconds': training_seconds, 'training_queries': len(train),
                                'model_bytes': (OUT / 'itq256.json').stat().st_size,
                                'source_sha256': {p: sha(ROOT / p) for p in (
                                    'experiments/run_progressive.py', 'context_stamps/routing.py',
                                    'context_stamps/learning.py', 'stamps.py')},
                                'input_sha256': {p: sha(ROOT / p) for p in (
                                    'evidence/spherical-public-v1/manifest.json', 'evidence/scifact-v1/manifest.json')}})
    save(OUT / 'checksums.json', {p.name: sha(p) for p in OUT.glob('*.json') if p.name != 'checksums.json'})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--scifact-cache', type=Path, required=True)
    parser.add_argument('--replication-cache', type=Path, required=True)
    with threadpool_limits(limits=1):
        main(parser.parse_args())

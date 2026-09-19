"""Replication across three fixed ITQ seeds; no seed selected using test scores."""

import argparse
import hashlib
import json
import statistics
import sys
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from context_stamps.learning import fit_family
from experiments.run_progressive import binary
from experiments.run_spherical_public import agreement, ndcg, qrels, read, save, sha

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'evidence/quantizer-seeds-v1'


def main(args):
    OUT.mkdir(exist_ok=True)
    protocol = {'seeds': [17, 41, 83], 'bits': 256, 'iterations': 50,
                'training': '391 original SciFact train query embeddings only; frozen MiniLM',
                'selection': 'No model/seed selection. Report all seeds and aggregate.',
                'status': 'Exploratory replication after seed-17 test results were inspected. No hyperparameter change.',
                'limits': 'Existing ITQ baseline, single-view; no base-model fine-tuning or independent new dataset.'}
    save(OUT / 'protocol.json', protocol)
    prior = read(ROOT / 'evidence/spherical-public-v1/manifest.json')
    split = read(ROOT / 'evidence/scifact-v1/manifest.json')
    records, models = [], {}
    for metadata in prior['datasets']:
        name = metadata['dataset']
        source = args.data / name
        assert all(sha(source / p) == digest for p, digest in metadata['data_sha256'].items())
        ids = [json.loads(line)['_id'] for line in (source / 'corpus.jsonl').read_text(encoding='utf-8').splitlines()]
        positions = {qid: i for i, qid in enumerate(metadata['query_ids'])}
        doc_positions = {key: i for i, key in enumerate(ids)}
        key = hashlib.sha256(json.dumps([metadata['data_sha256'], prior['protocol']['encoder'],
                                         prior['protocol']['revision'], metadata['query_ids']], sort_keys=True).encode()).hexdigest()
        cache = args.scifact_cache if name == 'scifact' else args.replication_cache
        dpath, qpath = cache / (key + '-D.npy'), cache / (key + '-Q.npy')
        assert sha(dpath) == metadata['embedding_sha256']['documents']
        assert sha(qpath) == metadata['embedding_sha256']['queries']
        documents, queries = np.load(dpath, allow_pickle=False), np.load(qpath, allow_pickle=False)
        if name == 'scifact':
            train = queries[[positions[qid] for qid in split['training_queries']]]
            for seed in protocol['seeds']:
                model = fit_family(train, encoder='minilm-pinned', bits=256, method='itq', seed=seed, iterations=50)
                models[seed] = model
                (OUT / f'itq256-{seed}.json').write_text(model.to_json() + '\n', encoding='utf-8', newline='\n')
        labels = qrels(source / 'qrels/test.tsv')
        for seed, model in models.items():
            dc, qc = binary(documents, model), binary(queries, model)
            for qid in sorted(labels):
                scores = agreement(qc[positions[qid]], dc, 256)
                if qid in doc_positions:
                    scores[doc_positions[qid]] = -np.inf
                order = np.argsort(-scores, kind='stable')[:10].tolist()
                records.append({'dataset': name, 'seed': seed, 'query_id': qid,
                                'ranked_ids': [ids[i] for i in order], 'ndcg10': ndcg(order, labels[qid], ids)})
        print(name, 'complete', flush=True)
    summary = []
    for metadata in prior['datasets']:
        name = metadata['dataset']
        scores = {str(seed): statistics.mean(r['ndcg10'] for r in records if r['dataset'] == name and r['seed'] == seed)
                  for seed in protocol['seeds']}
        summary.append({'dataset': name, 'per_seed': scores, 'mean': statistics.mean(scores.values()),
                        'minimum': min(scores.values()), 'maximum': max(scores.values())})
    save(OUT / 'results.json', records)
    save(OUT / 'summary.json', summary)
    save(OUT / 'manifest.json', {'protocol': protocol, 'numpy': np.__version__,
                                'source_sha256': {p: sha(ROOT / p) for p in (
                                    'experiments/run_quantizer_seeds.py', 'experiments/run_progressive.py',
                                    'context_stamps/learning.py', 'stamps.py')},
                                'input_sha256': {p: sha(ROOT / p) for p in (
                                    'evidence/spherical-public-v1/manifest.json', 'evidence/scifact-v1/manifest.json')}})
    save(OUT / 'checksums.json', {p.name: sha(p) for p in OUT.glob('*.json') if p.name != 'checksums.json'})
    print(summary, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--scifact-cache', type=Path, required=True)
    parser.add_argument('--replication-cache', type=Path, required=True)
    with threadpool_limits(limits=1):
        main(parser.parse_args())

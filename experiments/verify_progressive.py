"""Offline checks of new routing, quantization and context reuse records."""

import hashlib
import json
import statistics
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from context_stamps.routing import fit_routing_policy
from stamps import Family

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def historical_source(filename, digest):
    current = ROOT / filename
    if current.is_file() and hashlib.sha256(current.read_bytes()).hexdigest() == digest:
        return current
    snapshot = ROOT / 'evidence/source-snapshots' / f'{Path(filename).stem}-{digest}{Path(filename).suffix}'
    return snapshot


def main():
    for name in ('progressive-v1', 'quantizer-seeds-v1', 'context-reuse-v1', 'context-reuse-v2'):
        folder = ROOT / 'evidence' / name
        for filename, digest in read(folder / 'checksums.json').items():
            assert hashlib.sha256((folder / filename).read_bytes()).hexdigest() == digest
        manifest = read(folder / 'manifest.json')
        for field in ('source_sha256', 'input_sha256'):
            for filename, digest in manifest.get(field, {}).items():
                assert hashlib.sha256(historical_source(filename, digest).read_bytes()).hexdigest() == digest, filename
    folder = ROOT / 'evidence/progressive-v1'
    rows = read(folder / 'results.json')
    assert len(rows) == 2029 * 6
    gold = {(r['dataset'], r['query_id']): r for r in rows if r['method'] == 'dense'}
    assert len(gold) == 2029
    for row in rows:
        reference = gold[(row['dataset'], row['query_id'])]
        assert row['query_id'] not in row['ranked_ids']
        assert row['same_order_as_dense'] == (row['ranked_ids'] == reference['ranked_ids'])
        if row['method'] == 'progressive':
            assert row['same_order_as_dense'] and row['ndcg10'] == reference['ndcg10']
            assert row['route'] == 'precise' and not row['compact_attempted']
    for summary in read(folder / 'summary.json'):
        group = [r for r in rows if (r['method'], r['dataset']) == (summary['method'], summary['dataset'])]
        assert len(group) == summary['queries']
        for metric in ('ndcg10', 'same_order_fraction'):
            key = 'same_order_as_dense' if metric == 'same_order_fraction' else metric
            assert abs(statistics.mean(r[key] for r in group) - summary[metric]) < 1e-12
    calibration = read(folder / 'calibration.json')
    train_ids = {r['query_id'] for r in calibration['train']}
    val_ids = {r['query_id'] for r in calibration['validation']}
    assert not train_ids & val_ids
    assert not (train_ids | val_ids) & {qid for dataset, qid in gold if dataset == 'scifact'}
    policy, report = fit_routing_policy(calibration['train'], calibration['validation'],
                                        scope='scifact:minilm-pinned:itq256:v1')
    recorded = read(folder / 'policy.json')
    for key, value in recorded['policy'].items():
        assert asdict(policy)[key] == value
    assert report == recorded['report']
    assert policy.certified_error_upper_bound == report['error_upper_bound']
    assert policy.calibration_count == report['validation_accepted']
    model = Family.load(folder / 'itq256.json')
    assert model.bits == 256 and model.method == 'itq-v1' and model.seed == 17
    replication = ROOT / 'evidence/quantizer-seeds-v1'
    repeated = read(replication / 'results.json')
    assert len(repeated) == 2029 * 3
    first_run = {(r['dataset'], r['query_id']): r for r in rows if r['method'] == 'trained_itq_256'}
    for row in repeated:
        assert row['query_id'] not in row['ranked_ids']
        if row['seed'] == 17:
            original = first_run[(row['dataset'], row['query_id'])]
            assert row['ranked_ids'] == original['ranked_ids'] and row['ndcg10'] == original['ndcg10']
    assert Family.load(replication / 'itq256-17.json') == model
    for summary in read(replication / 'summary.json'):
        for seed in (17, 41, 83):
            group = [r for r in repeated if r['dataset'] == summary['dataset'] and r['seed'] == seed]
            assert len(group) == sum(dataset == summary['dataset'] for dataset, _ in gold)
            assert abs(statistics.mean(r['ndcg10'] for r in group) - summary['per_seed'][str(seed)]) < 1e-12
        values = summary['per_seed'].values()
        assert summary['mean'] == statistics.mean(values)
        assert summary['minimum'] == min(values) and summary['maximum'] == max(values)
    # Full quantizer retraining needs pinned external embeddings; not an offline CI claim.
    for name in ('context-reuse-v1', 'context-reuse-v2'):
        folder = ROOT / 'evidence' / name
        rows = read(folder / 'results.json')
        assert len(rows) == 400
        paired = {(r['root'], r['iteration']): r for r in rows if r['method'] == 'exact_graph'}
        for row in rows:
            assert row['evidence_bytes'] == paired[(row['root'], row['iteration'])]['evidence_bytes']
            if row['method'] == 'session_cache':
                assert row['cache_hit'] == (row['iteration'] > 0)
                assert row['transfer_payload_bytes'] == (32 if row['cache_hit'] else row['evidence_bytes'] + 32)
        for summary in read(folder / 'summary.json'):
            group = [r for r in rows if r['method'] == summary['method']]
            assert len(group) == summary['handoffs']
            assert sum(r['transfer_payload_bytes'] for r in group) == summary['transfer_payload_bytes']
            assert sum(r['evidence_bytes'] for r in group) == summary['resolved_evidence_bytes']
            assert abs(sum(r['milliseconds'] for r in group) - summary['total_ms']) < 1e-9
    print('Verified 18,261 public ranking records, three training seeds, disjoint calibration, 800 handoffs and hashes.')


if __name__ == '__main__':
    main()

"""Data-only replay of candidate code training; never loads executable weights."""

import gzip
import json
import math
from pathlib import Path

from local_adapter_train import sha
from mbpp_adapter import CONFIG
from source_evidence import verify_sources

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / 'evidence/mbpp-code-adapter-v1'


def read(name):
    return json.loads((EVIDENCE / name).read_text(encoding='utf-8'))


def main():
    manifest, plan, trained = read('manifest.json'), read('plan.json'), read('trained.json')
    for name, digest in manifest['files'].items():
        assert sha(EVIDENCE / name) == digest
    verify_sources(plan['source_hashes'])
    verify_sources(manifest['sources'])
    assert plan['config'] == CONFIG and not plan['promotion_allowed']
    assert plan['data_sha256'] == sha(EVIDENCE / 'training-data.json')
    assert plan['base_manifest_sha256'] == sha(EVIDENCE / 'base.json')
    assert plan['overlap_audit_sha256'] == sha(EVIDENCE / 'overlap-audit.json')
    source = ROOT / 'evidence/mbpp-training-v1'
    assert plan['qualification_plan_sha256'] == sha(source / 'plan.json')
    assert plan['qualification_summary_sha256'] == sha(source / 'summary.json')
    rows, overlap = read('training-data.json'), read('overlap-audit.json')
    qualified = json.loads((source / 'summary.json').read_text())['eligible']
    assert [r['task_id'] for r in rows] == [k for k in qualified if k not in overlap['excluded_task_ids']]
    originals = {r['task_id']: r for r in json.loads(gzip.decompress((source / 'source-training-rows.json.gz').read_bytes()))}
    assert all(r == originals[r['task_id']] for r in rows)
    training = [json.loads(s) for s in (EVIDENCE / 'training.jsonl').read_text().splitlines()]
    tokens = read('tokenization.json')
    for epoch in range(CONFIG['epochs']):
        epoch_rows = [r for r in training if r['epoch'] == epoch]
        assert sorted(k for r in epoch_rows for k in r['task_ids']) == sorted(r['task_id'] for r in rows)
        for r in epoch_rows:
            assert r['target_tokens'] == sum(tokens[str(k)]['target'] for k in r['task_ids'])
    assert all(math.isfinite(r['loss']) and math.isfinite(r['gradient_norm']) and r['seconds'] > 0 for r in training)
    assert trained['rows'] == len(rows) and trained['epochs'] == CONFIG['epochs']
    assert trained['frozen_base_unchanged'] and trained['seconds'] > 0
    assert trained['adapter_sha256'] == sha(EVIDENCE / 'adapter/adapter_model.safetensors')
    assert trained['adapter_bytes'] == (EVIDENCE / 'adapter/adapter_model.safetensors').stat().st_size
    assert not trained['activation'] and trained['benchmark_results'] is None and trained['energy_joules'] is None
    print(f"Local code SFT replay: {len(rows)} verified-source examples, {len(training)} steps; frozen base unchanged.")
    print('Checkpoint retained and inactive. Training completion does not establish coding or Context Stamps benefit.')


if __name__ == '__main__':
    main()

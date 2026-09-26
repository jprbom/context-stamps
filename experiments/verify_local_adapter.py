"""Data-only audit of the local adapter engineering run; never loads weights."""

import json
import math
from pathlib import Path

from local_adapter_data import ACTIONS, decision, fixtures, validate_rows
from local_adapter_train import CONFIG, score_summary, sha
from source_evidence import verify_sources

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / 'evidence/local-adapter-v1'


def read(name):
    return json.loads((EVIDENCE / name).read_text(encoding='utf-8'))


def main():
    manifest, plan = read('manifest.json'), read('plan.json')
    for name, digest in manifest['files'].items():
        assert sha(EVIDENCE / name) == digest, name
    verify_sources(plan['sources'])
    verify_sources(manifest['source_hashes'])
    assert plan['config'] == CONFIG and not plan['promotion_allowed']
    assert plan['data_sha256'] == sha(EVIDENCE / 'data.json')
    assert plan['base_manifest_sha256'] == sha(EVIDENCE / 'base.json')
    data = read('data.json')
    validate_rows(data)
    assert data == fixtures()
    by_id = {r['id']: r for r in data}
    training = [json.loads(s) for s in (EVIDENCE / 'training.jsonl').read_text().splitlines()]
    for epoch in range(CONFIG['epochs']):
        seen = [k for r in training if r['epoch'] == epoch for k in r['rows']]
        expected = [r['id'] for r in data if r['split'] == 'train']
        assert sorted(seen) == sorted(expected)
    assert len(training) == 72
    assert all(math.isfinite(r['loss']) and math.isfinite(r['gradient_norm']) and r['seconds'] > 0 for r in training)
    rows = [json.loads(s) for s in (EVIDENCE / 'evaluation.jsonl').read_text().splitlines()]
    assert sorted(r['id'] for r in rows) == sorted(r['id'] for r in data if r['split'] != 'train')
    for r in rows:
        original = by_id[r['id']]
        for name in ('target', 'cluster', 'split', 'kind'):
            assert r[name] == original[name]
        if r['kind'] == 'workflow':
            assert decision(original['state']) == r['target']
        for kind in ('base', 'adapter'):
            record = r[kind]
            assert len(record['logits']) == len(ACTIONS) and all(map(math.isfinite, record['logits']))
            assert record['prediction'] == 'ABCDEF'[max(range(6), key=record['logits'].__getitem__)]
            assert record['batch_seconds'] > 0 and record['batch_size'] == 4
        assert {r['base']['order'], r['adapter']['order']} == {0, 1}
    summary, trained = read('summary.json'), read('trained.json')
    for split, expected in score_summary(rows).items():
        assert summary[split] == expected
    assert not summary['promotion'] and summary['provider_charges_usd'] == 0
    assert summary['energy_joules'] is None and trained['frozen_base_unchanged']
    assert trained['adapter_sha256'] == sha(EVIDENCE / 'adapter/adapter_model.safetensors')
    assert read('reload.json')['passed']
    assert read('shape-probe.json')['tokenizer_parity']
    for record, source in (('reload.json', 'local_adapter_reload.py'),
                           ('shape-probe.json', 'local_adapter_probe.py'),
                           ('precision-probe.json', 'local_adapter_precision_probe.py')):
        assert read(record)['source_sha256'] == manifest['source_hashes']['experiments/' + source]
    print('One local LoRA fit: 144 training fixtures, 72 held-out workflow rows in 12 fixture clusters.')
    print('Recorded base 12/72, adapter 52/72, exact rule 72/72; one new failure. No activation.')
    print('Authored fixture study, weak retention controls; no coding, edge or general self-improvement qualification.')


if __name__ == '__main__':
    main()

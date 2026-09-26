"""Replay published training-data and native-grader evidence without execution."""

import gzip
import hashlib
import json
from pathlib import Path

from humaneval_local import parse_result as parse_humaneval
from mbpp_training_data import canonical
from qualify_mbpp_training import parse_result as parse_mbpp
from source_evidence import verify_sources

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    raw = path.read_bytes()
    if path.suffix == '.gz':
        raw = gzip.decompress(raw)
    return json.loads(raw)


def verify_files(directory):
    manifest = read(directory / 'manifest.json')
    for name, expected in manifest['files'].items():
        assert hashlib.sha256((directory / name).read_bytes()).hexdigest() == expected, name
    verify_sources(manifest['sources'])


def main():
    directory = ROOT / 'evidence/mbpp-training-v1'
    verify_files(directory)
    plan, summary = read(directory / 'plan.json'), read(directory / 'summary.json')
    verify_sources(plan['source_hashes'])
    rows, records = read(directory / 'source-training-rows.json.gz'), read(directory / 'records.json.gz')
    assert [r['task_id'] for r in rows] == list(range(601, 975))
    assert [r['task_id'] for r in records] == plan['task_ids'] == list(range(601, 975))
    by_id = {r['task_id']: r for r in rows}
    eligible = []
    for r in records:
        assert not r['containers_remaining']
        assert hashlib.sha256(canonical(by_id[r['task_id']])).hexdigest() == r['source_sha256']
        for key, value in plan['source_audit'][str(r['task_id'])].items():
            assert r[key] == value
        for kind in ('empty', 'reference'):
            if kind in r:
                assert parse_mbpp(r[kind]['execution']) == r[kind]['native']
        admitted = (not r['exact_reserved_overlap'] and 'error' not in r
                    and not r.get('empty', {}).get('native', {}).get('passed', True)
                    and r.get('reference', {}).get('native', {}).get('passed', False))
        assert admitted == r['eligible']
        if admitted:
            eligible.append(r['task_id'])
    assert summary['eligible'] == eligible and summary['total'] == 374
    assert summary['model_calls'] == summary['training_runs'] == 0 and not summary['containers_remaining']
    assert summary['exact_reserved_overlap'] == [r['task_id'] for r in records if r['exact_reserved_overlap']]
    assert summary['reference_failures'] == [r['task_id'] for r in records
                                            if not r.get('reference', {}).get('native', {}).get('passed', False)]
    print(f'MBPP source qualification replay: {len(eligible)}/374 eligible; no model training in this data audit.')

    directory = ROOT / 'evidence/humaneval-grader-v1'
    verify_files(directory)
    source = read(directory / 'source.json')
    full = read(directory / 'reference-audit.json.gz')
    for name in ('initial-smoke.json.gz', 'revised-smoke.json.gz', 'reference-audit.json.gz'):
        audit = read(directory / name)
        verify_sources(audit['plan']['sources'])
        assert len(audit['records']) == audit['summary']['total']
        for r in audit['records']:
            assert not r['containers_remaining']
            for mode in ('empty', 'reference'):
                if mode in r:
                    assert parse_humaneval(r[mode]['execution'], {'task_id': r['task_id'], 'mode': mode}) == r[mode]['native']
            passed = (not r.get('empty', {}).get('native', {}).get('passed', True)
                      and r.get('reference', {}).get('native', {}).get('passed', False))
            assert passed == r['passed']
        assert audit['summary']['qualified'] == sum(r['passed'] for r in audit['records'])
    assert len(full['records']) == source['tasks'] == 164
    assert sum(r['reference']['native']['checks']['base']['inputs'] for r in full['records']) == source['base_inputs']
    assert sum(r['reference']['native']['checks']['plus']['inputs'] for r in full['records']) == source['plus_inputs']
    for name in ('root-probe.json', 'numerical-probe.json'):
        probe = read(directory / name)
        verify_sources(probe['sources'])
        assert probe['model_calls'] == 0 and not probe['containers_remaining']
    numerical = read(directory / 'numerical-probe.json')['result']
    for r in numerical['records']:
        assert len(r['residuals']) == 65
        assert r['minimum_absolute_residual'] == min(abs(v['residual']) for v in r['residuals'])
        assert r['minimum_absolute_residual'] > numerical['atol']
    print(f"HumanEval+ reference audit replay: {full['summary']['qualified']}/164 reference/empty pairs meet expectations.")
    print('Includes retained transfer failure and numerical diagnostics; no model benchmark or automatic activation claim.')


if __name__ == '__main__':
    main()

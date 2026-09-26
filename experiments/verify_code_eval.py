"""Replay the retained four-task pipeline smoke, not a full benchmark claim."""

import gzip
import hashlib
import json
from pathlib import Path

from code_adapter_generate import extract_program
from humaneval_local import parse_result
from source_evidence import verify_sources

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / 'evidence/mbpp-code-eval-v1'


def main():
    manifest = json.loads((EVIDENCE / 'manifest.json').read_text())
    for name, digest in manifest['files'].items():
        assert hashlib.sha256((EVIDENCE / name).read_bytes()).hexdigest() == digest
    verify_sources(manifest['sources'])
    data = json.loads(gzip.decompress((EVIDENCE / 'smoke.json.gz').read_bytes()))
    verify_sources(data['generation_plan']['source_hashes'])
    verify_sources(data['grading_plan']['sources'])
    assert data['generation_plan']['smoke'] and data['completed']['generations'] == 8
    raw = (EVIDENCE / 'smoke-generations.jsonl').read_bytes()
    assert hashlib.sha256(raw).hexdigest() == data['grading_plan']['generation_sha256']
    rows = [json.loads(s) for s in raw.splitlines()]
    assert rows == data['generations']
    by_key = {(r['task_id'], r['arm']): r for r in rows}
    ids = ['HumanEval/' + str(i) for i in range(4)]
    assert set(by_key) == {(k, a) for k in ids for a in ('base', 'adapter')}
    for task_id in ids:
        base, adapter = by_key[(task_id, 'base')], by_key[(task_id, 'adapter')]
        assert base['prompt_sha256'] == adapter['prompt_sha256']
        assert {base['order'], adapter['order']} == {0, 1}
    results = {}
    for r in data['grading_records']:
        assert not r['containers_remaining']
        original = by_key[(r['task_id'], r['arm'])]
        assert all(original[k] == v for k, v in extract_program(original['text']).items())
        if not original['valid_format'] or original['reached_token_limit']:
            assert not r['passed'] and r['reason'] == 'invalid_format_or_generation_truncated'
            assert 'native' not in r
        else:
            request = {'task_id': r['task_id'], 'mode': 'candidate', 'code': original['code']}
            assert r['native'] == parse_result(r['execution'], request)
            assert r['passed'] == r['native']['passed']
        results[(r['task_id'], r['arm'])] = r['passed']
    summary = data['grading_summary']
    assert summary['tasks'] == 4 and summary['evaluations'] == 8
    for arm in ('base', 'adapter'):
        assert summary[arm + '_passed'] == sum(results[(k, arm)] for k in ids)
    assert not summary['errors'] and not summary['activation'] and not summary['containers_remaining']
    print('Paired four-task smoke: base 0/4, adapter 2/4; seven native grades and one format failure. No full-benchmark claim.')


if __name__ == '__main__':
    main()

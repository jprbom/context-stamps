"""Pinned MBPP training-data preparation, with no host code execution.

Copyright (c) 2026 Prashant Jagtap. MIT License.
Upstream MBPP records retain CC BY 4.0 attribution; see the evidence card.
"""

import ast
import hashlib
import json
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REVISION = 'f46ca8374b4cddef97ca4208ad986049d74d296a'
HF_REVISION = '4bb6404fdc6cacfda99d4ac4205087b89d32030c'
EVIDENCE = ROOT / 'evidence/mbpp-training-v1'


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(',', ':'), allow_nan=False).encode()


def write_new(path, value):
    with path.open('x', encoding='utf-8', newline='\n') as stream:
        stream.write(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')


def fresh(path):
    path = path.resolve()
    if path.is_relative_to(ROOT):
        raise ValueError('Keep raw data and execution artifacts outside the repository')
    path.mkdir(parents=True, exist_ok=False)
    return path


def verify_source(directory):
    expected = json.loads((EVIDENCE / 'source.json').read_text(encoding='utf-8'))
    if expected['repository'] != 'google-research/google-research' or expected['revision'] != REVISION:
        raise ValueError('Unexpected MBPP revision')
    for name, value in expected['files'].items():
        if (directory / name).stat().st_size != value['bytes'] or sha((directory / name).read_bytes()) != value['sha256']:
            raise ValueError('Changed source file: ' + name)
    if sha((directory / 'HF_DATASET_CARD.md').read_bytes()) != expected['license_evidence']['sha256']:
        raise ValueError('Changed license evidence')
    if expected['license_evidence']['license'] != ['cc-by-4.0']:
        raise ValueError('Unreviewed data license')
    if expected['license_evidence']['revision'] != HF_REVISION:
        raise ValueError('Unreviewed dataset card revision')
    return expected


def fetch(path):
    out = fresh(path)
    expected = json.loads((EVIDENCE / 'source.json').read_text(encoding='utf-8'))
    class NoRedirect(urllib.request.HTTPRedirectHandler):
        def redirect_request(self, *args, **kwargs):
            raise ValueError('Unexpected source redirect')
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect())
    downloads = [(name, 'https://raw.githubusercontent.com/google-research/google-research/' + REVISION + '/' + name,
                  record['bytes'], record['sha256']) for name, record in expected['files'].items()]
    card = expected['license_evidence']
    card_url = 'https://huggingface.co/datasets/google-research-datasets/mbpp/raw/' + HF_REVISION + '/README.md'
    if card['url'] != card_url:
        raise ValueError('Unreviewed dataset card URL')
    downloads.append(('HF_DATASET_CARD.md', card_url, 200000, card['sha256']))
    for name, url, bound, digest in downloads:
        if '\\' in name or '..' in Path(name).parts or Path(name).is_absolute():
            raise ValueError('Invalid data path')
        with opener.open(url, timeout=45) as response:
            raw = response.read(bound + 1)
        if len(raw) > bound or sha(raw) != digest:
            raise ValueError('Source checksum mismatch')
        target = out / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(raw)
    verify_source(out)


def solution_key(code):
    try:
        return sha(ast.dump(ast.parse(code), include_attributes=False).encode())
    except SyntaxError:
        return sha(code.encode())


def training_rows(directory):
    verify_source(directory)
    rows = [json.loads(line) for line in (directory / 'mbpp/mbpp.jsonl').read_text(encoding='utf-8').splitlines()]
    if len(rows) != 974 or {r['task_id'] for r in rows} != set(range(1, 975)):
        raise ValueError('Unexpected source task inventory')
    keys = {solution_key(r['code']) for r in rows if r['task_id'] <= 600}
    texts = {' '.join(r['text'].lower().split()) for r in rows if r['task_id'] <= 600}
    train = sorted((r for r in rows if 601 <= r['task_id'] <= 974), key=lambda r: r['task_id'])
    for row in train:
        if (len(row['test_list']) != 3 or len(row['code'].encode()) > 4096
                or len(canonical(row)) > 8192 or row.get('challenge_test_list')):
            raise ValueError('Unexpected training record contract')
    audit = {str(r['task_id']): {'source_sha256': sha(canonical(r)), 'solution_ast_sha256': solution_key(r['code']),
             'exact_reserved_overlap': solution_key(r['code']) in keys or ' '.join(r['text'].lower().split()) in texts}
             for r in train}
    return train, audit

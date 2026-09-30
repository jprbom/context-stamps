"""Replay the conservative adapter's 328 paired development outcomes."""

import argparse
import gzip
import hashlib
import json
from pathlib import Path

from mbpp_training_data import ROOT
from source_evidence import verify_sources
from summarize_code_comparison import summarize_records


def verify(out):
    manifest = json.loads((out / 'manifest.json').read_text(encoding='utf-8'))
    verify_sources(manifest['sources'])
    if set(manifest['files']) != {'records.json.gz', 'generations.jsonl.gz', 'summary.json'}:
        raise ValueError('Unexpected evidence inventory')
    for name, digest in manifest['files'].items():
        if hashlib.sha256((out / name).read_bytes()).hexdigest() != digest:
            raise ValueError('Evidence checksum mismatch: ' + name)
    data = json.loads(gzip.decompress((out / 'records.json.gz').read_bytes()))
    raw = gzip.decompress((out / 'generations.jsonl.gz').read_bytes())
    rows = [json.loads(line) for line in raw.splitlines()]
    plan, grading = data['generation_plan'], data['grading_plan']
    verify_sources(plan['source_hashes'])
    verify_sources(grading['sources'])
    if grading['generation_sha256'] != hashlib.sha256(raw).hexdigest() or grading['generation_plan'] != plan:
        raise ValueError('Generation binding changed')
    completed = data['generation_completed']
    if completed['generations'] != 328 or completed['tasks'] != 164 or completed['activation']:
        raise ValueError('Incomplete generation or unqualified activation')
    trained = json.loads((ROOT / 'evidence/mbpp-conservative-adapter-v2/trained.json').read_text())
    if plan['adapter_sha256'] != trained['adapter_sha256'] or plan['activation_allowed']:
        raise ValueError('Changed or activated candidate')
    actual = summarize_records(rows, plan, data['grading_records'], data['grading_summary'])
    expected = json.loads((out / 'summary.json').read_text(encoding='utf-8'))
    if actual != expected:
        raise ValueError('Derived paired statistics or resources changed')
    for arm in ('base', 'adapter'):
        if sum(actual['models'][arm]['outcome_stages'].values()) != 164:
            raise ValueError('Dropped failure stage')
    print('Full paired code comparison: 164 tasks, 328 retained outputs, native grades and descriptive statistics replayed.')
    print('Candidate remains inactive; no runtime-treatment, edge-device or retention qualification.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--evidence', type=Path, default=ROOT / 'evidence/mbpp-conservative-comparison-v2')
    verify(parser.parse_args().evidence)

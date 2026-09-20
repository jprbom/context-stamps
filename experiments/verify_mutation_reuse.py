"""Check mutation-reuse records and aggregate calculations offline."""

import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'evidence/mutation-reuse-v1'


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def historical_source(filename, digest):
    current = ROOT / filename.replace('\\', '/')
    if current.is_file() and hashlib.sha256(current.read_bytes()).hexdigest() == digest:
        return current
    return ROOT / 'evidence/source-snapshots' / f'{Path(filename).stem}-{digest}{Path(filename).suffix}'


def main():
    for filename, digest in read(OUT / 'checksums.json').items():
        assert hashlib.sha256((OUT / filename).read_bytes()).hexdigest() == digest
    manifest = read(OUT / 'manifest.json')
    for field in ('source_sha256', 'input_sha256'):
        for filename, digest in manifest[field].items():
            path = historical_source(filename, digest)
            assert hashlib.sha256(path.read_bytes()).hexdigest() == digest, filename
    rows = read(OUT / 'results.json')
    assert len(rows) == 6600
    exact = {(r['pairs'], r['round'], r['pair']): r for r in rows if r['method'] == 'exact_graph'}
    assert len(exact) == 2200
    for row in rows:
        control = exact[(row['pairs'], row['round'], row['pair'])]
        assert row['packet_sha256'] == control['packet_sha256']
        assert row['evidence_bytes'] == control['evidence_bytes']
        hit = (row['method'] == 'selective_cache' and row['round'] > 0
               and row['pair'] != (row['round'] - 1) % row['pairs'])
        assert row['cache_hit'] == hit
        transfer = row['evidence_bytes'] if row['method'] == 'exact_graph' else (32 if hit else row['evidence_bytes'] + 32)
        assert row['transfer_payload_bytes'] == transfer
    for summary in read(OUT / 'summary.json'):
        group = [r for r in rows if (r['pairs'], r['method']) == (summary['pairs'], summary['method'])]
        assert len(group) == summary['handoffs']
        assert sum(r['cache_hit'] for r in group) == summary['cache_hits']
        assert sum(r['transfer_payload_bytes'] for r in group) == summary['transfer_payload_bytes']
        assert abs(sum(r['milliseconds'] for r in group) - summary['handoff_ms']) < 1e-9
    receipts = read(OUT / 'receipt-checks.json')
    assert len(receipts) == 152
    for row in receipts:
        expected = 'complete' if row['method'] == 'selective_cache' and not row['affected'] else 'insufficient'
        assert row['status'] == expected
    print('Verified 6,600 handoffs, identical packets, 152 receipt checks and source hashes.')


if __name__ == '__main__':
    main()

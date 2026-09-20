"""Compare exact, global-cache and selective-cache handoffs under updates."""

import hashlib
import importlib.util
import random
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from context_stamps import ContextGraph, ContextNode, ContextSession
from experiments.run_spherical_public import read, save, sha

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'evidence/mutation-reuse-v1'
REFERENCE = 'evidence/source-snapshots/session-3973e527a95587dd4cd4294cd1dbf75a093add14c0aa15a6a98092eaba3875cf.py'


def main():
    OUT.mkdir(exist_ok=True)
    assert sha(ROOT / REFERENCE) == '3973e527a95587dd4cd4294cd1dbf75a093add14c0aa15a6a98092eaba3875cf'
    spec = importlib.util.spec_from_file_location('context_stamps._global_cache_reference', ROOT / REFERENCE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    protocol = {'pairs': [10, 100], 'rounds': 20, 'seed': 5819,
                'workload': '20 actual source files reused across explicitly declared independent pairs; 100-pair case duplicates text under synthetic IDs, not 100 independent tasks',
                'mutation': 'Before each round after the first, change one dependency at the same revision and revalidate its edge',
                'comparison': ['uncached exact graph', 'v0.3.2 global invalidation', 'v0.3.3 selective invalidation'],
                'measurement': 'Randomized method order per round, full packet equality and live receipt checks. Handoff and mutation times separate. No model/network calls; transfer bytes are modeled payload accounting.'}
    save(OUT / 'protocol.json', protocol)
    paths = [ROOT / p for p in sorted(read(ROOT / 'evidence/context-reuse-v2/manifest.json')['input_sha256'])]
    texts = [p.read_text(encoding='utf-8') for p in paths]
    records, mutations, receipt_checks, setups = [], [], [], []
    rng = random.Random(protocol['seed'])
    for pairs in protocol['pairs']:
        stores = {'exact_graph': ContextGraph(), 'global_cache': module.ContextSession(), 'selective_cache': ContextSession()}
        revisions = {f'n{i}': '1' for i in range(pairs * 2)}
        last_receipts = {'global_cache': {}, 'selective_cache': {}}
        for method, store in stores.items():
            start = time.perf_counter_ns()
            for i, key in enumerate(revisions):
                store.put(ContextNode(key, texts[i % len(texts)], '1', frozenset({'developer'})))
            for pair in range(pairs):
                store.link(f'n{2*pair}', f'n{2*pair+1}', 'depends_on', provenance='declared-pair')
            setups.append({'pairs': pairs, 'method': method, 'milliseconds': (time.perf_counter_ns() - start) / 1e6})
        for iteration in range(protocol['rounds']):
            changed = (iteration - 1) % pairs
            if iteration:
                for method, store in stores.items():
                    start = time.perf_counter_ns()
                    target = 2 * changed + 1
                    store.put(ContextNode(f'n{target}', texts[target % len(texts)] + f'\n# revision event {iteration}\n',
                                          '1', frozenset({'developer'})))
                    store.link(f'n{2*changed}', f'n{target}', 'depends_on', provenance='revalidated')
                    mutations.append({'pairs': pairs, 'iteration': iteration, 'method': method,
                                      'milliseconds': (time.perf_counter_ns() - start) / 1e6})
                    if method != 'exact_graph':
                        for pair in (changed, (changed + 1) % pairs):
                            result = store.resolve(last_receipts[method][pair], role='developer', revisions=revisions,
                                                   budget_bytes=1048576)
                            expected = 'complete' if method == 'selective_cache' and pair != changed else 'insufficient'
                            assert result.status == expected
                            receipt_checks.append({'pairs': pairs, 'iteration': iteration, 'method': method,
                                                   'affected': pair == changed, 'status': result.status})
            methods = list(stores)
            rng.shuffle(methods)
            packet_hashes = {}
            for method in methods:
                for pair in range(pairs):
                    start = time.perf_counter_ns()
                    if method == 'exact_graph':
                        packet = stores[method].handoff([f'n{2*pair}'], role='developer', revisions=revisions,
                                                        budget_bytes=1048576)
                        transfer, reused = packet.units, False
                    else:
                        packet, token, reused = stores[method].issue([f'n{2*pair}'], role='developer',
                                                                   revisions=revisions, budget_bytes=1048576)
                        assert len(token) == 32
                        last_receipts[method][pair] = token
                        transfer = 32 if reused else packet.units + 32
                    milliseconds = (time.perf_counter_ns() - start) / 1e6
                    assert packet.status == 'complete'
                    digest = hashlib.sha256(packet.text.encode()).hexdigest()
                    assert packet_hashes.setdefault(pair, digest) == digest
                    records.append({'pairs': pairs, 'round': iteration, 'pair': pair, 'method': method,
                                    'milliseconds': milliseconds, 'cache_hit': reused, 'evidence_bytes': packet.units,
                                    'transfer_payload_bytes': transfer, 'packet_sha256': digest})
    summary = []
    for pairs in protocol['pairs']:
        for method in ('exact_graph', 'global_cache', 'selective_cache'):
            rows = [r for r in records if (r['pairs'], r['method']) == (pairs, method)]
            changes = [r for r in mutations if (r['pairs'], r['method']) == (pairs, method)]
            summary.append({'pairs': pairs, 'method': method, 'handoffs': len(rows),
                            'cache_hits': sum(r['cache_hit'] for r in rows),
                            'handoff_ms': sum(r['milliseconds'] for r in rows),
                            'mutation_ms': sum(r['milliseconds'] for r in changes),
                            'median_handoff_ms': statistics.median(r['milliseconds'] for r in rows),
                            'transfer_payload_bytes': sum(r['transfer_payload_bytes'] for r in rows),
                            'resolved_evidence_bytes': sum(r['evidence_bytes'] for r in rows)})
    for name, value in (('results', records), ('mutations', mutations), ('receipt-checks', receipt_checks),
                        ('setups', setups), ('summary', summary)):
        save(OUT / (name + '.json'), value)
    save(OUT / 'manifest.json', {'protocol': protocol,
                                'source_sha256': {p: sha(ROOT / p) for p in (
                                    'experiments/run_mutation_reuse.py', 'context_stamps/session.py',
                                    'context_stamps/workflow.py', REFERENCE)},
                                'input_sha256': {p.relative_to(ROOT).as_posix(): sha(p) for p in paths}})
    save(OUT / 'checksums.json', {p.name: sha(p) for p in OUT.glob('*.json') if p.name != 'checksums.json'})
    print(summary, flush=True)


if __name__ == '__main__':
    main()

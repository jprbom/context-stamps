"""Replay repeated evidence handoffs from actual repository source files.

This is a context-transport microbenchmark, not a coding-agent success study.
"""

import platform
import random
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from context_stamps.session import ContextSession
from context_stamps.workflow import ContextGraph, ContextNode
from experiments.run_spherical_public import save, sha

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'evidence/context-reuse-v1'


def main():
    OUT.mkdir(exist_ok=True)
    protocol = {'workload': '20 sorted existing context_stamps Python files excluding new routing/session modules',
                'iterations': 20, 'methods': ['exact_graph', 'session_cache'], 'seed': 9381,
                'measurement': 'Two source files per packet; declared paired-file dependency. Time includes authorization/version checks and packet construction/cache lookup, excludes ingestion and network.',
                'transfer': 'First delivery sends full packet plus a 32-byte receipt; subsequent deliveries send receipt only to an already populated shared resolver. Receipt metadata, protocol headers and authorization traffic are external.',
                'limits': 'A repeated-context replay using real code text, not real user tasks or agent completions. No model token, network latency or production scale claim.'}
    save(OUT / 'protocol.json', protocol)
    paths = [p for p in sorted((ROOT / 'context_stamps').glob('*.py'))
             if p.name not in ('routing.py', 'session.py', '__init__.py')][:20]
    assert len(paths) == 20
    session, graph = ContextSession(), ContextGraph()
    revisions = {f'file{i}': 'v1' for i in range(len(paths))}
    setup = time.perf_counter_ns()
    for i, path in enumerate(paths):
        node = ContextNode(f'file{i}', path.read_text(encoding='utf-8'), 'v1', frozenset({'developer'}))
        session.put(node)
        graph.put(node)
    for i in range(0, len(paths), 2):
        for store in (session, graph):
            store.link(f'file{i}', f'file{i+1}', 'depends_on', provenance='declared-replay-pair')
    setup_ms = (time.perf_counter_ns() - setup) / 1e6
    records, rng = [], random.Random(protocol['seed'])
    for iteration in range(protocol['iterations']):
        for i in range(0, len(paths), 2):
            methods = list(protocol['methods'])
            rng.shuffle(methods)
            packets = []
            for method in methods:
                start = time.perf_counter_ns()
                if method == 'session_cache':
                    packet, receipt, reused = session.issue([f'file{i}'], role='developer', revisions=revisions,
                                                           budget_bytes=1048576)
                    transfer = 32 if reused else packet.units + 32
                    assert len(receipt) == 32
                else:
                    packet = graph.handoff([f'file{i}'], role='developer', revisions=revisions, budget_bytes=1048576)
                    transfer, reused = packet.units, False
                milliseconds = (time.perf_counter_ns() - start) / 1e6
                assert packet.status == 'complete'
                packets.append(packet)
                records.append({'root': f'file{i}', 'iteration': iteration, 'method': method,
                                'milliseconds': milliseconds, 'evidence_bytes': packet.units,
                                'transfer_payload_bytes': transfer, 'cache_hit': reused})
            assert packets[0] == packets[1]
    summary = []
    for method in protocol['methods']:
        rows = [r for r in records if r['method'] == method]
        summary.append({'method': method, 'handoffs': len(rows), 'cache_hits': sum(r['cache_hit'] for r in rows),
                        'total_ms': sum(r['milliseconds'] for r in rows),
                        'median_ms': statistics.median(r['milliseconds'] for r in rows),
                        'transfer_payload_bytes': sum(r['transfer_payload_bytes'] for r in rows),
                        'resolved_evidence_bytes': sum(r['evidence_bytes'] for r in rows)})
    for name, value in (('results', records), ('summary', summary)):
        save(OUT / (name + '.json'), value)
    save(OUT / 'manifest.json', {'protocol': protocol, 'platform': platform.platform(), 'combined_setup_ms': setup_ms,
                                'source_sha256': {p: sha(ROOT / p) for p in (
                                    'experiments/run_context_reuse.py', 'context_stamps/session.py', 'context_stamps/workflow.py')},
                                'input_sha256': {p.relative_to(ROOT).as_posix(): sha(p) for p in paths}})
    save(OUT / 'checksums.json', {p.name: sha(p) for p in OUT.glob('*.json') if p.name != 'checksums.json'})
    print(summary)


if __name__ == '__main__':
    main()

"""Persistent context tests and bounded-residency replay; no model-quality claim."""

import argparse
import hashlib
import io
import json
import os
import platform
import random
import secrets
import sqlite3
import statistics
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from enterprise_state_validation import RecordedResult  # noqa: E402

from context_stamps.context_state import AccessScope, CanonicalNode, ContextClaim, TemporalScope  # noqa: E402
from context_stamps.state_store import ContextStore  # noqa: E402
from context_stamps.working_set import ContextWorkingSet  # noqa: E402


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def stream(repeats, seed):
    positions = set(random.Random(seed).sample(range(1, 20), repeats))
    result = [0]
    for position in range(1, 20):
        result.append(result[-1] if position in positions else result[-1] + 1)
    return result


def benchmark():
    scope = AccessScope("lab", "researcher", "p1", ("reader",))
    key, results = secrets.token_bytes(32), []
    authorize = lambda s, op, subject: s == scope  # noqa: E731
    with tempfile.TemporaryDirectory(prefix="context-residency-measurement-") as directory:
        path = Path(directory) / "state.sqlite"
        policy = CanonicalNode("policy", "v1", "lab", "Fictional policy. " + "p" * 24000, "POLICY",
            TemporalScope(0, 10), ("reader",), "fixture", claims=(ContextClaim("limit", "128"),), validation_revision="v1")
        nodes = [CanonicalNode(f"record-{i}", "v1", "lab", f"Fictional record {i}. " + "x" * 24000, "OBSERVATION",
            TemporalScope(0, 10), ("reader",), "fixture", claims=(ContextClaim("value", str(i)),),
            dependencies=(policy.ref,), validation_revision="v1") for i in range(21)]
        with ContextStore(path, tenant="lab", signing_key=key, authorize=authorize, create=True,
                          policy_revision="p1", clock=lambda: 100) as store:
            for node in (policy, *nodes):
                store.put(node, scope=scope, mutation_id=node.key)
            anchor = store.checkpoint(scope=scope, verify=True)
        order = [(repeats, seed, mode) for repeats in (0, 2, 10) for seed in (7, 29, 61)
                 for mode in ("direct", "working_set", "prefetch")]
        random.Random(1701).shuffle(order)
        for repeats, seed, mode in order:
            started = time.perf_counter()
            with ContextStore(path, tenant="lab", signing_key=key, authorize=authorize, checkpoint=anchor) as store:
                open_ms = (time.perf_counter() - started) * 1000
                reads, rows, tickets, replies = [], [], [], []
                original = store._row
                def measured_row(sequence):
                    row = original(sequence)
                    reads.append(len(row[3].encode()))
                    return row
                store._row = measured_row
                memory = None if mode == "direct" else ContextWorkingSet(store, scope=scope, max_nodes=4, max_bytes=120000)
                values = stream(repeats, seed)
                start = time.perf_counter()
                failure = None
                try:
                    for position, value in enumerate(values):
                        if memory is not None:
                            for ticket in list(tickets):
                                reply = memory.prefetch_result(ticket)
                                if reply.status != "pending":
                                    replies.append(dict(status=reply.status, cold_reads=reply.cold_reads, reused=reply.reused))
                                    tickets.remove(ticket)
                        t0 = time.perf_counter()
                        if mode == "direct":
                            snapshot = store.snapshot(scope, at=100, known_at=100, roots=(nodes[value].ref,))
                        else:
                            snapshot = memory.activate((nodes[value].ref,), at=100, known_at=100)
                        answers = [int(c.value) for r in snapshot.records if r.node.key == nodes[value].key
                                   for c in r.node.claims if c.name == "value"]
                        correct = answers == [value] and len(snapshot.records) == 2
                        rows.append(dict(position=position, value=value, correct=correct,
                                         milliseconds=(time.perf_counter() - t0) * 1000))
                        if not correct:
                            raise RuntimeError("incorrect replay result")
                        if mode == "prefetch" and len(tickets) < memory.max_prefetch:
                            # Fixed successor rule; never consults the next request.
                            # Repeated requests make this prediction unnecessary.
                            tickets.append(memory.prefetch((nodes[value + 1].ref,), at=100, known_at=100))
                    if memory is not None:
                        # Include all wasted/in-flight work in whole-stream cost.
                        deadline = time.perf_counter() + 10
                        while tickets:
                            for ticket in list(tickets):
                                reply = memory.prefetch_result(ticket)
                                if reply.status != "pending":
                                    replies.append(dict(status=reply.status, cold_reads=reply.cold_reads, reused=reply.reused))
                                    tickets.remove(ticket)
                            if time.perf_counter() > deadline:
                                raise TimeoutError("prefetch drain exceeded fixture deadline")
                            if tickets:
                                time.sleep(.001)
                    stats = memory.stats() if memory is not None else None
                except Exception as exc:
                    failure = type(exc).__name__
                    stats = None
                finally:
                    if memory is not None:
                        memory.close()
                elapsed_ms = (time.perf_counter() - start) * 1000
                durations = sorted(r["milliseconds"] for r in rows)
                results.append(dict(repeats=repeats, requests=20, seed=seed, mode=mode, stream=values,
                    passed=failure is None and len(rows) == 20, failure_type=failure, rows=rows,
                    open_with_replay_ms=open_ms, total_stream_ms=elapsed_ms,
                    median_demand_ms=statistics.median(durations) if durations else None,
                    p95_demand_ms=durations[(len(durations) * 95 + 99) // 100 - 1] if durations else None,
                    payload_read_count=len(reads), payload_read_bytes=sum(reads),
                    working_stats=stats, prefetch_replies=replies))
    return dict(cases=results, source_nodes=22, requests_per_case=20,
        source_text_filler_bytes=24000, max_resident_nodes=4, max_resident_bytes=120000,
        gpu_used=False, predictor="fixed numerical successor; no next-request access",
        scope="fictional exact-reference replay; metadata remains in RAM, SQLite/OS caches uncontrolled; no model or external I/O delay")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise ValueError("fresh evidence directory required")
    args.out.mkdir(parents=True)
    log = io.StringIO()
    result = unittest.TextTestRunner(stream=log, verbosity=2, resultclass=RecordedResult).run(
        unittest.defaultTestLoader.discover(str(ROOT / "tests")))
    with (args.out / "tests.log").open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(log.getvalue())
    sources = [*ROOT.glob("context_stamps/**/*.py"), *ROOT.glob("tests/*.py"), Path(__file__),
               ROOT / "examples/persistent_context.py", ROOT / "experiments/enterprise_state_validation.py",
               ROOT / "experiments/source_evidence.py"]
    report = dict(schema=1, python=sys.version, platform=platform.platform(), sqlite=sqlite3.sqlite_version,
        source_base=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        source_hashes={p.relative_to(ROOT).as_posix(): sha(p) for p in sorted(sources)},
        discovered=result.testsRun, tests=result.rows, passed=result.wasSuccessful(), skipped=len(result.skipped),
        cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"), test_log_sha256=sha(args.out / "tests.log"),
        duration_clock=vars(time.get_clock_info("perf_counter")),
        limits=["Fictional exact-reference tasks; no learned acquisition, trained prediction or model quality measurement.",
                "Warm cache compares against the same durable store with direct body materialization; the OS page cache is not flushed.",
                "Serialized resident bytes exclude Python allocator metadata, immutable snapshots held by callers and in-flight materialization.",
                "Stream time includes ticket handling and prefetch drain; initial verified store replay is reported separately.",
                "One bounded local workload; no network storage, production concurrency, deletion policy or independent replication qualification."])
    if report["passed"] and not report["skipped"]:
        report["residency"] = benchmark()
        report["passed"] = all(case["passed"] for case in report["residency"]["cases"])
        example = subprocess.run([sys.executable, str(ROOT / "examples/persistent_context.py")],
                                 cwd=ROOT, capture_output=True, text=True, timeout=30)
        report["example"] = dict(returncode=example.returncode, stdout=example.stdout, stderr=example.stderr)
        report["passed"] = report["passed"] and example.returncode == 0
    with (args.out / "manifest.json").open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")
    summaries = []
    for repeats in (0, 2, 10):
        for mode in ("direct", "working_set", "prefetch"):
            cases = [c for c in report.get("residency", {}).get("cases", []) if c["repeats"] == repeats and c["mode"] == mode]
            if cases:
                summaries.append(dict(repeat_fraction=repeats / 20, mode=mode,
                    correct=sum(r["correct"] for c in cases for r in c["rows"]),
                    median_stream_ms=statistics.median(c["total_stream_ms"] for c in cases),
                    median_demand_ms=statistics.median(c["median_demand_ms"] for c in cases),
                    payload_reads=sum(c["payload_read_count"] for c in cases)))
    print(json.dumps(dict(tests=report["discovered"], passed=report["passed"], skipped=report["skipped"], summaries=summaries), indent=2))
    if not report["passed"] or report["skipped"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

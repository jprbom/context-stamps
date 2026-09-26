"""Record full tests and same-workflow controls for exact incremental computation."""

import argparse
import hashlib
import io
import json
import os
import platform
import random
import statistics
import subprocess
import sys
import time
import unittest
from dataclasses import replace
from fractions import Fraction
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from enterprise_state_validation import RecordedResult  # noqa: E402

from context_stamps.incremental import IncrementalExecutor  # noqa: E402
from examples.incremental_metrics import ADAPTERS, TARGETS, count_node, fixture  # noqa: E402


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def benchmark():
    jobs = [(r, seed, mode, scope) for r, scope in ((0, "local"), (2, "local"), (10, "local"), (0, "policy"))
            for seed in (7, 29, 61) for mode in ("no_reuse", "full_invalidation", "incremental")]
    random.Random(1701).shuffle(jobs)
    cases = []
    for repeats, seed, mode, mutation in jobs:
        state, scope, sources, graph = fixture()
        repeat_positions = set(random.Random(seed).sample(range(1, 20), repeats))
        values, current = [], 10
        for i in range(20):
            if i > 0 and i not in repeat_positions:
                current += 1
            values.append(current)
        executor = IncrementalExecutor(state, ADAPTERS, authorize=lambda s, spec: s == scope)
        if mode == "no_reuse":
            graph = tuple(replace(s, reusable=False) for s in graph)
        rows, failure = [], None
        start = time.perf_counter()
        try:
            for position, value in enumerate(values):
                began = time.perf_counter()
                if position > 0 and value != values[position - 1]:
                    previous = sources["fp"]
                    sources["fp"] = count_node("fp", value, f"v{position + 1}", (previous.ref,))
                    state.put(sources["fp"])
                    graph = tuple(replace(s, sources=(sources["fp"].ref,)) if s.key == "fp" else s for s in graph)
                    if mutation == "policy":
                        scope = replace(scope, policy_revision=f"p{position}")
                        state.set_policy(scope.policy_revision)
                    if mode == "full_invalidation":
                        executor.clear()
                result = executor.run(graph, TARGETS, scope=scope, at=200, known_at=200)
                correct = (result.status == "complete" and {r.key: r.text for r in result.results}
                           == {"f1": str(Fraction(80, 85 + value)), "other": "99"})
                receipt_checks = all(executor.verify_receipt(r.receipt, r.text) for r in result.results)
                rows.append(dict(position=position, false_positives=value, status=result.status,
                    computed=[s.key for s in result.steps if s.status == "computed"],
                    reused=[s.key for s in result.steps if s.status == "reused"],
                    correct=correct, receipt_checks=receipt_checks,
                    milliseconds=(time.perf_counter() - began) * 1000,
                    model_calls=sum(s.usage.model_calls for s in result.steps),
                    input_tokens=sum(s.usage.input_tokens for s in result.steps),
                    cache=executor.cache_info()))
                if not correct or not receipt_checks:
                    raise RuntimeError("incremental metric mismatch")
        except Exception as error:
            failure = type(error).__name__
        finally:
            executor.close()
        duration = (time.perf_counter() - start) * 1000
        cases.append(dict(repeats=repeats, seed=seed, mode=mode, mutation=mutation,
            values=values, rows=rows, failure_type=failure, passed=failure is None and len(rows) == 20,
            total_stream_ms=duration,
            computed=sum(len(r["computed"]) for r in rows), reused=sum(len(r["reused"]) for r in rows)))
    return dict(cases=cases, source="fictional confusion counts", nodes=7, requests_per_case=20,
        order_seed=1701, gpu_used=False,
        scope="pure rational arithmetic; same state/graph/adapters across controls; stream time includes mutation, verification and cleanup")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise ValueError("fresh evidence directory required")
    args.out.mkdir(parents=True)
    log = io.StringIO()
    tests = unittest.TextTestRunner(stream=log, verbosity=2, resultclass=RecordedResult).run(
        unittest.defaultTestLoader.discover(str(ROOT / "tests")))
    (args.out / "tests.log").write_text(log.getvalue(), encoding="utf-8", newline="\n")
    sources = [*ROOT.glob("context_stamps/**/*.py"), *ROOT.glob("tests/*.py"), Path(__file__),
               ROOT / "examples/incremental_metrics.py", ROOT / "experiments/enterprise_state_validation.py",
               ROOT / "experiments/source_evidence.py"]
    report = dict(schema=1, python=sys.version, platform=platform.platform(),
        source_base=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        source_hashes={p.relative_to(ROOT).as_posix(): sha(p) for p in sorted(sources)},
        discovered=tests.testsRun, tests=tests.rows, passed=tests.wasSuccessful(), skipped=len(tests.skipped),
        cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"), test_log_sha256=sha(args.out / "tests.log"),
        duration_clock=vars(time.get_clock_info("perf_counter")),
        limits=["Fictional deterministic metrics, not model training or public benchmark accuracy.",
                "All model calls and tokens are actually zero. Fewer callback invocations do not establish token savings.",
                "Tiny in-memory functions; overhead may dominate. No hard callback interruption or distributed atomic publication.",
                "Single-flight local executor; no production concurrency or durable result-cache qualification."])
    if report["passed"] and not report["skipped"]:
        report["benchmark"] = benchmark()
        report["passed"] = all(c["passed"] for c in report["benchmark"]["cases"])
        example = subprocess.run([sys.executable, str(ROOT / "examples/incremental_metrics.py")],
                                 cwd=ROOT, capture_output=True, text=True, timeout=30)
        report["example"] = dict(returncode=example.returncode, stdout=example.stdout, stderr=example.stderr)
        report["passed"] = report["passed"] and example.returncode == 0
    (args.out / "manifest.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8", newline="\n")
    summaries = []
    for mutation, repeats in (("local", 0), ("local", 2), ("local", 10), ("policy", 0)):
        for mode in ("no_reuse", "full_invalidation", "incremental"):
            cases = [c for c in report.get("benchmark", {}).get("cases", [])
                     if (c["mutation"], c["repeats"], c["mode"]) == (mutation, repeats, mode)]
            if cases:
                summaries.append(dict(mutation=mutation, repeat_fraction=repeats / 20, mode=mode,
                    computations=sum(c["computed"] for c in cases),
                    median_stream_ms=statistics.median(c["total_stream_ms"] for c in cases)))
    print(json.dumps(dict(tests=tests.testsRun, passed=report["passed"], skipped=report["skipped"], summaries=summaries), indent=2))
    if not report["passed"] or report["skipped"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

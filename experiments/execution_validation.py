"""Record full tests and owned-worker overhead on fictional offline operations."""

import argparse
import hashlib
import io
import json
import os
import platform
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

from context_stamps.audit import AuditStore  # noqa: E402
from context_stamps.execution import ExecutionAdapter, ManagedExecutor, WorkerResult  # noqa: E402
from context_stamps.experience import (  # noqa: E402
    Action,
    Authority,
    Episode,
    EpisodeEnd,
    EvidenceRef,
    ResourceUse,
    TransitionPlan,
)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def now():
    return time.time_ns() // 1000000


def provider(payload, key, deadline):
    return WorkerResult(payload.upper(), ResourceUse(0, 0, 0, 1, 0.))


def verifier(payload, output):
    return output == payload.upper()


def benchmark():
    actor = Authority("lab", "researcher", "p1")
    payload = b"fictional local operation"
    ref = EvidenceRef("lab", "input", "v1", hashlib.sha256(payload).hexdigest(), "observation")
    output_ref = EvidenceRef("lab", "output", "v1", hashlib.sha256(payload.upper()).hexdigest(), "observation")
    adapter = ExecutionAdapter("uppercase", "v1", "pure", "exact-v1", provider, verifier)
    rows = []
    with tempfile.TemporaryDirectory(prefix="execution-measurement-") as directory:
        with AuditStore(Path(directory) / "audit.sqlite", tenant="lab", signing_key=secrets.token_bytes(32),
                        authorize=lambda a, op, r: a == actor, create=True) as store:
            def publish(raw):
                if raw != payload.upper():
                    raise ValueError("wrong output")
                (Path(directory) / "output.bin").write_bytes(raw)
                return output_ref
            executor = ManagedExecutor(store, adapters=(adapter,), resolve=lambda r: payload,
                                       publish=publish, is_current=lambda p: True)
            for repeat in range(20):
                row = {"repeat": repeat, "order": ["direct", "managed"] if repeat % 2 == 0 else ["managed", "direct"]}
                for method in row["order"]:
                    start = time.perf_counter()
                    if method == "direct":
                        value = provider(payload, ref.digest, now() + 10000)
                        accepted = verifier(payload, value.output)
                    else:
                        episode_id = f"case-{repeat:02d}"
                        episode = Episode(episode_id, "uppercase", "fictional-lab", "exact-v1", "development",
                            "offline-v1", 7, actor, "python-v1", "exact-v1", now())
                        action = Action(adapter.tool, adapter.revision, ref, "pure", ref.digest)
                        plan = TransitionPlan(episode_id, 0, (ref,), "state-v1", action, None, None, now())
                        store.append(episode, authority=actor)
                        store.append(plan, authority=actor)
                        result = executor.run(plan, authority=actor, deadline_ms=now() + 10000)
                        accepted = result.outcome.record.status == "succeeded"
                        if not accepted:
                            raise RuntimeError("managed benchmark fixture failed")
                        store.append(EpisodeEnd(episode_id, "succeeded", now()), authority=actor)
                        row["managed_worker_wall_ms"] = result.outcome.record.cost.wall_ms
                    row[method + "_ms"] = (time.perf_counter() - start) * 1000
                    row[method + "_correct"] = accepted
                rows.append(row)
            checkpoint = store.verify(authority=actor)
    summary = {}
    for method in ("direct", "managed"):
        values = sorted(r[method + "_ms"] for r in rows)
        summary[method] = dict(median_ms=statistics.median(values), p95_ms=values[18], correct=sum(r[method + "_correct"] for r in rows))
    return dict(rows=rows, summary=summary, events=checkpoint.sequence, gpu_used=False,
                scope="20 alternating pairs; direct function and verifier versus full journal episode plus spawned worker; no model")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise ValueError("new evidence directory required")
    args.out.mkdir(parents=True)
    stream = io.StringIO()
    result = unittest.TextTestRunner(stream=stream, verbosity=2, resultclass=RecordedResult).run(
        unittest.defaultTestLoader.discover(str(ROOT / "tests")))
    with (args.out / "tests.log").open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(stream.getvalue())
    sources = [*ROOT.glob("context_stamps/**/*.py"), *ROOT.glob("tests/*.py"), Path(__file__),
               ROOT / "examples/managed_decision.py", ROOT / "experiments/enterprise_state_validation.py",
               ROOT / "experiments/source_evidence.py"]
    report = dict(schema=1, python=sys.version, sqlite=sqlite3.sqlite_version, platform=platform.platform(),
        duration_clock=vars(time.get_clock_info("perf_counter")),
        source_base=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        source_hashes={p.relative_to(ROOT).as_posix(): sha(p) for p in sorted(sources)},
        tests=result.rows, discovered=result.testsRun, passed=result.wasSuccessful(), skipped=len(result.skipped),
        cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"), test_log_sha256=sha(args.out / "tests.log"),
        limits=["Fictional offline inputs and exact Python functions; no training or benchmark-quality gain.",
                "Managed timing includes Python process startup and five separately committed audit records.",
                "Single sequential local writer for timing; separate tests cover process contention/crashes.",
                "Killing an owned worker does not cancel remote effects or unmanaged descendants.",
                "Host authorization, current-state, artifact and reconciliation callbacks remain trusted and must return promptly."])
    if result.wasSuccessful() and not result.skipped:
        report["execution_overhead"] = benchmark()
        example = subprocess.run([sys.executable, str(ROOT / "examples/managed_decision.py")],
                                 capture_output=True, text=True, timeout=30, cwd=ROOT)
        report["example"] = dict(returncode=example.returncode, stdout=example.stdout, stderr=example.stderr)
        report["passed"] = example.returncode == 0
    with (args.out / "manifest.json").open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")
    print(json.dumps(dict(tests=report["discovered"], passed=report["passed"], skipped=report["skipped"],
                         overhead=report.get("execution_overhead", {}).get("summary")), indent=2))
    if not report["passed"] or report["skipped"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

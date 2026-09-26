"""Record complete tests, durable append overhead and an integrated offline task."""

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
from context_stamps.experience import (  # noqa: E402
    Action,
    Authority,
    Episode,
    EpisodeEnd,
    EvidenceRef,
    ResourceUse,
    TransitionOutcome,
    TransitionPlan,
    encode_record,
)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def percentile95(values):
    return sorted(values)[int((len(values) * 95 + 99) // 100) - 1]


def benchmark():
    results = []
    actor = Authority("lab", "researcher", "p1")
    ref = EvidenceRef("lab", "fixture", "v1", hashlib.sha256(b"fictional audit cost input").hexdigest(), "observation")
    action = Action("exact-check", "v1", ref, "pure", ref.digest)
    for episodes in (32, 256):
        with tempfile.TemporaryDirectory(prefix="context-audit-benchmark-") as directory:
            path = Path(directory) / "audit.sqlite"
            key = secrets.token_bytes(32)
            arguments = dict(tenant="lab", signing_key=key, authorize=lambda a, op, record: a == actor,
                             clock=lambda: 10000)
            rows = []
            start = time.perf_counter()
            with AuditStore(path, create=True, **arguments) as store:
                open_ms = (time.perf_counter() - start) * 1000
                payload_bytes = 0
                for i in range(episodes):
                    episode_id = f"episode-{i:04d}"
                    records = (
                        Episode(episode_id, "exact-check", "fictional-project", "template-v1", "development",
                                "offline-v1", 7, actor, "exact-provider-v1", "exact-verifier-v1", 100),
                        TransitionPlan(episode_id, 0, (ref,), "belief-v1", action, None, None, 101),
                        TransitionOutcome(episode_id, 0, action, (ref,), "succeeded", True, "exact-verifier-v1",
                                          None, ResourceUse(0, 0, 0, 1, 0.), (), 102),
                        EpisodeEnd(episode_id, "succeeded", 103),
                    )
                    for record in records:
                        payload_bytes += len(encode_record(record).encode())
                        start = time.perf_counter()
                        receipt = store.append(record, authority=actor)
                        rows.append(dict(episode=i, kind=type(record).__name__, sequence=receipt.checkpoint.sequence,
                                         milliseconds=(time.perf_counter() - start) * 1000))
                anchor = store.checkpoint(authority=actor)
                if anchor.sequence != episodes * 4:
                    raise RuntimeError("unexpected audit event count")
                start = time.perf_counter()
                store.verify(authority=actor, checkpoint=anchor)
                verify_ms = (time.perf_counter() - start) * 1000
            size = path.stat().st_size
            start = time.perf_counter()
            with AuditStore(path, checkpoint=anchor, **arguments) as reopened:
                reopen_ms = (time.perf_counter() - start) * 1000
                if len(reopened.history(f"episode-{episodes-1:04d}", authority=actor)) != 4:
                    raise RuntimeError("last episode failed restart replay")
            durations = [r["milliseconds"] for r in rows]
            first = [r["milliseconds"] for r in rows if r["episode"] < 8]
            last = [r["milliseconds"] for r in rows if r["episode"] >= episodes - 8]
            results.append(dict(episodes=episodes, events=len(rows), rows=rows, open_ms=open_ms,
                full_verify_ms=verify_ms, reopen_with_full_replay_ms=reopen_ms, file_bytes=size,
                record_body_bytes=payload_bytes, median_append_ms=statistics.median(durations),
                p95_append_ms=percentile95(durations), first_8_episodes_median_ms=statistics.median(first),
                last_8_episodes_median_ms=statistics.median(last)))
    return results


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
               ROOT / "examples/audited_decision.py", ROOT / "experiments/enterprise_state_validation.py"]
    report = dict(schema=1, python=sys.version, sqlite=sqlite3.sqlite_version, platform=platform.platform(),
        source_base=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        source_hashes={p.relative_to(ROOT).as_posix(): sha(p) for p in sorted(sources)},
        tests=result.rows, discovered=result.testsRun, passed=result.wasSuccessful(), skipped=len(result.skipped),
        cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"), test_log_sha256=sha(args.out / "tests.log"),
        limits=["Reference-only fictional records; timing append operations does not execute the represented tools.",
                "Local SQLite WAL with synchronous=FULL, one sequential writer in timing fixtures; no network filesystem or power-loss test.",
                "Full test suite separately tests two-process contention and process crashes, not enterprise-scale concurrent load.",
                "Signing detects changed records; protecting the key/storage and retaining an external checkpoint are host responsibilities."])
    if result.wasSuccessful() and not result.skipped:
        report["durable_append"] = benchmark()
        example = subprocess.run([sys.executable, str(ROOT / "examples/audited_decision.py")],
                                 capture_output=True, text=True, timeout=30, cwd=ROOT)
        report["example"] = dict(returncode=example.returncode, stdout=example.stdout, stderr=example.stderr)
        report["passed"] = example.returncode == 0
    with (args.out / "manifest.json").open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")
    print(json.dumps({"tests": report["discovered"], "passed": report["passed"], "skipped": report["skipped"],
        "durable_append": [{k: v for k, v in row.items() if k != "rows"} for row in report.get("durable_append", [])]}, indent=2))
    if not report["passed"] or report["skipped"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

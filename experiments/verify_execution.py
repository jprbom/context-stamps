"""Replay recorded execution engineering evidence, including the earlier run."""

import hashlib
import json
import math
import statistics
from pathlib import Path

from source_evidence import verify_sources

ROOT = Path(__file__).resolve().parents[1]


def main():
    for directory, expected_tests, execution_tests in (("local-cpu", 231, 28), ("local-cpu-v2", 233, 30),
                                                      ("local-cpu-v3", 233, 30)):
        evidence = ROOT / "evidence/enterprise-execution-v1" / directory
        report = json.loads((evidence / "manifest.json").read_text(encoding="utf-8"))
        assert report["passed"] and report["discovered"] == expected_tests and report["skipped"] == 0
        assert report["cuda_visible_devices"] == ""
        assert len(report["tests"]) == len({r["test"] for r in report["tests"]}) == expected_tests
        assert all(r["outcome"] == "passed" and math.isfinite(r["seconds"]) and r["seconds"] >= 0 for r in report["tests"])
        assert sum(r["test"].startswith("test_execution.") for r in report["tests"]) == execution_tests
        assert sum(r["test"].startswith("test_source_evidence.") for r in report["tests"]) == 2
        assert hashlib.sha256((evidence / "tests.log").read_bytes()).hexdigest() == report["test_log_sha256"]
        verify_sources(report["source_hashes"])
        overhead = report["execution_overhead"]
        assert overhead["gpu_used"] is False and overhead["events"] == 100
        assert len(overhead["rows"]) == 20 and [r["repeat"] for r in overhead["rows"]] == list(range(20))
        for row in overhead["rows"]:
            assert row["order"] == (["direct", "managed"] if row["repeat"] % 2 == 0 else ["managed", "direct"])
            assert row["direct_correct"] and row["managed_correct"]
            assert math.isfinite(row["managed_worker_wall_ms"]) and row["managed_worker_wall_ms"] > 0
            if directory == "local-cpu-v3":
                assert row["managed_worker_wall_ms"] <= row["managed_ms"]
                assert report["duration_clock"]["monotonic"] and report["duration_clock"]["resolution"] <= 1e-6
        for method in ("direct", "managed"):
            values = sorted(r[method + "_ms"] for r in overhead["rows"])
            summary = overhead["summary"][method]
            assert all(math.isfinite(v) and v > 0 for v in values)
            assert math.isclose(summary["median_ms"], statistics.median(values), rel_tol=1e-12)
            assert math.isclose(summary["p95_ms"], values[18], rel_tol=1e-12)
            assert summary["correct"] == 20
        assert report["example"]["returncode"] == 0
        example = json.loads(report["example"]["stdout"])
        assert example["verified_decision"] and example["committed_events"] == 5
        assert example["duplicate_blocked"] and example["reopened_and_verified"]
        assert example["external_references_verified"] == 2 and example["model_calls"] == 0
        assert math.isfinite(example["managed_wall_ms"]) and example["managed_wall_ms"] > 0
    failure = json.loads((ROOT / "evidence/enterprise-execution-v1/timing-failure.json").read_text())
    for directory, expected in failure["rows_with_inner_duration_exceeding_outer"].items():
        report = json.loads((ROOT / "evidence/enterprise-execution-v1" / directory / "manifest.json").read_text())
        actual = [r["repeat"] for r in report["execution_overhead"]["rows"] if r["managed_worker_wall_ms"] > r["managed_ms"]]
        assert actual == expected
    print("233 final test records and two earlier runs verified; 60 offline paired timing cases replayed.")
    print("Recorded engineering evidence only; no model training, latency improvement or distributed exactly-once claim.")


if __name__ == "__main__":
    main()

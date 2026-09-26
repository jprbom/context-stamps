"""Replay stored audit engineering measurements; no model or production claim."""

import hashlib
import json
import math
import statistics
from pathlib import Path

from source_evidence import verify_sources

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "evidence/enterprise-audit-v1/local-cpu"


def main():
    report = json.loads((EVIDENCE / "manifest.json").read_text(encoding="utf-8"))
    assert report["passed"] and report["discovered"] == 201 and report["skipped"] == 0
    assert report["cuda_visible_devices"] == ""
    assert len(report["tests"]) == len({r["test"] for r in report["tests"]}) == 201
    assert all(r["outcome"] == "passed" and math.isfinite(r["seconds"]) and r["seconds"] >= 0 for r in report["tests"])
    assert sum(r["test"].startswith("test_audit.") for r in report["tests"]) == 19
    assert hashlib.sha256((EVIDENCE / "tests.log").read_bytes()).hexdigest() == report["test_log_sha256"]
    verify_sources(report["source_hashes"])
    assert [case["episodes"] for case in report["durable_append"]] == [32, 256]
    for case in report["durable_append"]:
        rows = case["rows"]
        assert len(rows) == case["events"] == case["episodes"] * 4
        assert [r["sequence"] for r in rows] == list(range(1, len(rows) + 1))
        assert [r["episode"] for r in rows] == [i // 4 for i in range(len(rows))]
        assert [r["kind"] for r in rows] == ["Episode", "TransitionPlan", "TransitionOutcome", "EpisodeEnd"] * case["episodes"]
        durations = sorted(r["milliseconds"] for r in rows)
        assert all(math.isfinite(v) and v > 0 for v in durations)
        assert math.isclose(case["median_append_ms"], statistics.median(durations), rel_tol=1e-12)
        assert math.isclose(case["p95_append_ms"], durations[(len(rows) * 95 + 99) // 100 - 1], rel_tol=1e-12)
        for name, selected in (("first_8", rows[:32]), ("last_8", rows[-32:])):
            assert math.isclose(case[f"{name}_episodes_median_ms"], statistics.median(r["milliseconds"] for r in selected), rel_tol=1e-12)
        assert case["file_bytes"] >= case["record_body_bytes"] > 0
        assert all(math.isfinite(case[key]) and case[key] > 0 for key in ("open_ms", "full_verify_ms", "reopen_with_full_replay_ms"))
    assert report["example"]["returncode"] == 0
    example = json.loads(report["example"]["stdout"])
    assert example["verified_decision"] and example["committed_events"] == 4
    assert example["external_references_verified"] == 3 and example["reopened_and_verified"]
    assert example["revocation_blocks_history"] and example["model_calls"] == 0
    print("201 passing test records, 1,152 durable appends and the integrated offline example replayed.")
    print("Engineering evidence only; no exactly-once external execution, learned model or production-scale claim.")


if __name__ == "__main__":
    main()

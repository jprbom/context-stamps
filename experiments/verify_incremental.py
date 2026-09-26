"""Replay recorded DAG evidence; no model inference or new timing measurement."""

import hashlib
import json
import math
import statistics
from pathlib import Path

from source_evidence import verify_sources

ROOT = Path(__file__).resolve().parents[1]


def main():
    directory = ROOT / "evidence/enterprise-incremental-v1/local-cpu"
    report = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    assert report["passed"] and report["discovered"] == 290 and report["skipped"] == 0
    assert len(report["tests"]) == len({t["test"] for t in report["tests"]}) == 290
    assert all(t["outcome"] == "passed" and math.isfinite(t["seconds"]) and t["seconds"] >= 0 for t in report["tests"])
    assert sum(t["test"].startswith("test_incremental.") for t in report["tests"]) == 23
    assert report["cuda_visible_devices"] == ""
    assert hashlib.sha256((directory / "tests.log").read_bytes()).hexdigest() == report["test_log_sha256"]
    verify_sources(report["source_hashes"])
    benchmark = report["benchmark"]
    assert benchmark["nodes"] == 7 and benchmark["gpu_used"] is False
    cases = benchmark["cases"]
    assert len(cases) == 36
    assert {(c["mutation"], c["repeats"], c["seed"], c["mode"]) for c in cases} == {
        (scope, r, seed, mode) for scope, r in (("local", 0), ("local", 2), ("local", 10), ("policy", 0))
        for seed in (7, 29, 61) for mode in ("no_reuse", "full_invalidation", "incremental")}
    nodes = {"tp", "fp", "fn", "precision", "recall", "f1", "other"}
    for case in cases:
        assert case["passed"] and case["failure_type"] is None
        rows, values = case["rows"], case["values"]
        assert len(rows) == len(values) == 20
        assert sum(a == b for a, b in zip(values, values[1:])) == case["repeats"]
        assert math.isfinite(case["total_stream_ms"]) and case["total_stream_ms"] > 0
        assert sum(r["milliseconds"] for r in rows) <= case["total_stream_ms"]
        assert case["computed"] == sum(len(r["computed"]) for r in rows)
        assert case["reused"] == sum(len(r["reused"]) for r in rows)
        assert case["computed"] + case["reused"] == 140
        for i, row in enumerate(rows):
            assert row["position"] == i and row["false_positives"] == values[i]
            assert row["correct"] and row["receipt_checks"] and row["status"] == "complete"
            assert row["model_calls"] == row["input_tokens"] == 0
            assert math.isfinite(row["milliseconds"]) and row["milliseconds"] > 0
            computed, reused = set(row["computed"]), set(row["reused"])
            assert computed | reused == nodes and not computed.intersection(reused)
            if i == 0 or case["mode"] == "no_reuse":
                expected = nodes
            elif values[i] == values[i - 1]:
                expected = set()
            elif case["mutation"] == "policy" or case["mode"] == "full_invalidation":
                expected = nodes
            else:
                expected = {"fp", "precision", "f1"}
            assert computed == expected
            assert 0 <= row["cache"]["entries"] <= 256
            assert 0 <= row["cache"]["serialized_bytes"] <= 4 * 1024 * 1024
    example = json.loads(report["example"]["stdout"])
    assert report["example"]["returncode"] == 0
    assert (example["initial_f1"], example["updated_f1"]) == ("16/19", "16/21")
    assert example["first_computations"] == 7 and example["repeat_computations"] == 0
    assert set(example["changed_computations"]) == {"fp", "precision", "f1"}
    assert example["model_calls"] == 0 and example["revoked_run"] == "unavailable_context"
    summaries = []
    for mutation, repeats in (("local", 0), ("local", 2), ("local", 10), ("policy", 0)):
        selected = [c for c in cases if c["mutation"] == mutation and c["repeats"] == repeats]
        summaries.append(dict(mutation=mutation, repeat_fraction=repeats / 20,
            milliseconds={m: statistics.median(c["total_stream_ms"] for c in selected if c["mode"] == m)
                          for m in ("no_reuse", "full_invalidation", "incremental")}))
    print(json.dumps(summaries, indent=2))
    print("290 test records and 720 exact workflow requests verified; no training or model benchmark claim.")


if __name__ == "__main__":
    main()

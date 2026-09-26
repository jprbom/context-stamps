"""Replay recorded engineering evidence; not independent model evaluation."""

import hashlib
import json
import math
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "evidence/enterprise-state-v1/local-cpu"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    report = json.loads((EVIDENCE / "manifest.json").read_text(encoding="utf-8"))
    assert report["passed"] and report["discovered"] == 182
    assert report["cuda_visible_devices"] == ""
    assert len(report["tests"]) == len({r["test"] for r in report["tests"]}) == 182
    assert all(r["outcome"] == "passed" and math.isfinite(r["seconds"]) and r["seconds"] >= 0 for r in report["tests"])
    assert sum(r["test"].startswith("test_enterprise_context.") for r in report["tests"]) == 27
    assert sha(EVIDENCE / "tests.log") == report["test_log_sha256"]
    for name, expected in report["source_hashes"].items():
        assert sha(ROOT / name) == expected, name
    fixture = report["compiler_fixture"]
    assert len(fixture["rows"]) == 80 and len(fixture["summary"]) == 4
    assert [s["records"] for s in fixture["summary"]] == [4, 16, 64, 256]
    for summary in fixture["summary"]:
        rows = [r for r in fixture["rows"] if r["records"] == summary["records"]]
        assert len(rows) == summary["trials"] == 20
        assert {r["repeat"] for r in rows} == set(range(1, 21))
        assert all(r["compiled_selected"] == 1 and r["full_selected"] == summary["records"] for r in rows)
        for method in ("full", "compiled"):
            durations = sorted(r[f"{method}_ms"] for r in rows)
            assert all(math.isfinite(d) and d > 0 for d in durations)
            assert math.isclose(summary[f"{method}_median_ms"], statistics.median(durations), rel_tol=1e-12)
            assert math.isclose(summary[f"{method}_p95_ms"], durations[18], rel_tol=1e-12)
            assert all(type(r[f"{method}_bytes"]) is int and r[f"{method}_bytes"] > 0 for r in rows)
        reduction = 1 - sum(r["compiled_bytes"] for r in rows) / sum(r["full_bytes"] for r in rows)
        assert math.isclose(summary["byte_reduction_fraction"], reduction, rel_tol=1e-12)
    print("182 test outcomes and 80 synthetic compiler measurements replayed; no model-quality claim.")


if __name__ == "__main__":
    main()

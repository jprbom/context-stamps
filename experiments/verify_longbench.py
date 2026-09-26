"""Replay published pilot accounting; no model, dataset download or network."""

import json
import math
from pathlib import Path

from longbench_eval import DATA_SHA, METHODS, extract_answer, file_hash, summarize, templates
from report_longbench import derive
from source_evidence import verify_sources

ROOT = Path(__file__).resolve().parents[1]


def main():
    directory = ROOT / "evidence/longbench-v2-pilot/run-v2"
    for suffix, count in (("", 15), ("-v2", 17)):
        boundary = json.loads((directory.parent / f"boundary{suffix}-manifest.json").read_text())
        verify_sources(boundary["source_hashes"])
        assert boundary["exit_code"] == 0 and boundary["tests"] == count
        assert file_hash(directory.parent / f"boundary{suffix}-tests.txt") == boundary["log_sha256"]
    original = json.loads((directory.parent / "run-v1/plan.json").read_text())
    verify_sources(original["source_hashes"])
    superseded = json.loads((directory.parent / "run-v1/superseded.json").read_text())
    assert superseded["generation_calls"] == 0
    assert superseded["plan_sha256"] == file_hash(directory.parent / "run-v1/plan.json")
    plan = json.loads((directory / "plan.json").read_text())
    summary = json.loads((directory / "summary.json").read_text())
    started = json.loads((directory / "started.json").read_text())
    expected = file_hash(directory / "plan.json")
    assert expected == summary["plan_sha256"] == started["plan_sha256"]
    assert plan["input_hashes"]["data.json"] == DATA_SHA
    verify_sources(plan["source_hashes"])
    templates()
    assert len(plan["eligibility"]) == 503
    assert len({r["id"] for r in plan["eligibility"]}) == 503
    for row in plan["eligibility"]:
        assert row["eligible"] == (row["full_input_tokens"] <= plan["max_input"])
    samples = {s["id"]: s for s in plan["samples"]}
    assert len(samples) == len(plan["samples"])
    rows, seen = [], set()
    for name, expected in summary["attempt_hashes"].items():
        assert name.startswith("attempt-") and Path(name).name == name and name.endswith(".json")
        assert file_hash(directory / name) == expected
        row = json.loads((directory / name).read_text())
        key = row["id"], row["method"]
        assert key not in seen and row["method"] in METHODS
        seen.add(key)
        assert row["whole_request_seconds"] > 0 and math.isfinite(row["whole_request_seconds"])
        if row["status"] == "ok":
            meta = samples[row["id"]]["prompts"][row["method"]]
            assert all(row[k] == value for k, value in meta.items())
            assert row["usage"]["prompt_eval_count"] == row["input_tokens"]
            assert row["prediction"] == extract_answer(row["response"])
            assert row["correct"] == (row["prediction"] == row["target"])
        else:
            assert row.get("error")
        rows.append(row)
    assert file_hash(directory / "telemetry.json") == summary["telemetry_sha256"]
    replayed = summarize(plan, rows)
    assert len(replayed) == len(summary["methods"])
    for actual, recorded in zip(replayed, summary["methods"]):
        assert actual.keys() == recorded.keys()
        for key, value in actual.items():
            if isinstance(value, float):
                assert math.isfinite(value) and math.isclose(value, recorded[key], rel_tol=1e-12, abs_tol=1e-12)
            else:
                assert value == recorded[key]
    if summary["status"] == "completed":
        assert len(rows) == len(samples)*len(METHODS) and all(r["status"] == "ok" for r in rows)
    else:
        assert summary["fatal"]
    analysis = json.loads((directory / "analysis.json").read_text())
    verify_sources(analysis["source_hashes"])
    def equivalent(a, b):
        if isinstance(a, dict):
            return a.keys() == b.keys() and all(equivalent(a[k], b[k]) for k in a)
        if isinstance(a, list):
            return len(a) == len(b) and all(equivalent(x, y) for x, y in zip(a, b))
        if isinstance(a, float):
            return math.isfinite(a) and math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-12)
        return a == b
    assert equivalent(derive(directory), analysis)
    print(json.dumps(dict(status=summary["status"], tasks=len(samples), attempts=len(rows),
                          correct={r["method"]: r["correct"] for r in summary["methods"]})))


if __name__ == "__main__":
    main()

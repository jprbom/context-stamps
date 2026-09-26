"""Replay frozen acquisition trajectories and training/artifact bindings on CPU."""

import gzip
import hashlib
import json
import math
import statistics
import sys
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from source_evidence import verify_sources  # noqa: E402

from context_stamps.acquisition import AcquisitionModel, StopPolicy, Trajectory, assess_policy  # noqa: E402


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    directory = ROOT / "evidence/acquisition-v1"
    read = lambda name: json.loads((directory / name).read_text(encoding="utf-8"))  # noqa: E731
    protocol, report = read("protocol.json"), read("manifest.json")
    calibration, selection = read("calibration.json"), read("selection.json")
    verify_sources(protocol["sources"])
    assert protocol["data_manifest_sha256"] == sha(ROOT / "evidence/controller-v2/data-manifest.json")
    for filename, field in (("protocol.json", "protocol_sha256"), ("selection.json", "selection_sha256"),
                            ("calibration.json", "calibration_sha256"), ("trajectories.json.gz", "records_sha256")):
        assert sha(directory / filename) == report[field]
    assert report["train_queries"] == 3134 and report["tune_queries"] == 290
    assert selection["frozen_before_calibration"] is True
    trials = selection["trials"]
    assert len(trials) == 6 and {(t["width"], t["seed"]) for t in trials} == {
        (w, s) for w in protocol["architectures"] for s in protocol["seeds"]}
    for trial in trials:
        path = directory / (trial["name"] + ".json")
        assert sha(path) == trial["file_sha256"]
        model = AcquisitionModel.load(path)
        assert model.revision == trial["model_revision"]
        assert len(trial["history"]) == protocol["epochs"] + 1
        assert trial["tune_loss"] == min(trial["history"]) == trial["history"][trial["selected_epoch"]]
        assert trial["parameters"] == (1187 if trial["width"] else 102)
        assert 0 <= trial["portable_cuda_max_error"] <= 1e-5 and trial["seconds"] > 0
    selected = min(trials, key=lambda t: (t["tune_loss"], t["name"]))["name"]
    assert report["selected"] == selection["selected"] == selected
    model = AcquisitionModel.load(directory / (selected + ".json"))
    with gzip.open(directory / "trajectories.json.gz", "rb") as stream:
        raw = stream.read(32 * 1024 * 1024 + 1)
    assert len(raw) <= 32 * 1024 * 1024
    data = json.loads(raw)
    assert len(data["calibration"]) == 788 and len(data["test"]) == 3677
    assignments = json.loads((ROOT / "evidence/controller-v2/data-manifest.json").read_text())["assignments"]
    for split, rows in data.items():
        assert len({(r["dataset"], r["query_id"]) for r in rows}) == len(rows)
        for name in {r["dataset"] for r in rows}:
            assert [r["query_id"] for r in rows if r["dataset"] == name] == assignments[name][split]
        for row in rows:
            n = row["count"]
            assert n == len(row["lengths"]) == len(row["predictions"]) == len(row["complete"]) <= 9
            assert row["lengths"] == sorted(set(row["lengths"])) and row["lengths"][-1] <= 256
            assert row["positive_counts"] == sorted(row["positive_counts"])
            assert row["gains"] == sorted(row["gains"])
            assert row["pool_gain"] == row["gains"][-1]
            assert row["full_pool_positive_count"] == row["positive_counts"][-1] <= row["total_relevant"]
            assert row["complete"] == [v > 0 and v == row["full_pool_positive_count"] for v in row["positive_counts"]]
            assert all(len(p) == 3 and all(type(v) is float and math.isfinite(v) and 0 <= v <= 1 for v in p)
                       for p in row["predictions"])
    family = tuple(tuple(v) for v in calibration["family"])
    assert len(family) == len(calibration["reports"]) == 12
    for row in calibration["reports"]:
        policy = StopPolicy(model.revision, row["threshold"])
        trajectories = tuple(Trajectory(r["query_id"], tuple(p[0] for p in r["predictions"]), tuple(r["complete"]))
                             for r in data["calibration"] if r["dataset"] == row["scope"])
        risk = assess_policy(policy, trajectories, scope=row["scope"], family=family)
        assert asdict(risk) == row["risk"]
        assert not risk.permits(policy, scope=row["scope"], maximum_error=protocol["maximum_error"])
    assert calibration["choices"] == {"scifact": None, "nfcorpus": None, "fiqa": None}
    assert calibration["independent_final_qualification"] is False
    assert len(report["summary"]) == 30
    for summary in report["summary"]:
        rows = [r for r in data["test"] if r["dataset"] == summary["dataset"]]
        method = summary["method"]
        if method in ("full-pool", "risk-gated"):
            indexes = [r["count"] - 1 for r in rows]
        elif method == "raw-selected-.90":
            policy = StopPolicy(model.revision, .9)
            indexes = [policy.choose(tuple(p[0] for p in r["predictions"])) for r in rows]
        else:
            limit = int(method.split("-")[1])
            indexes = [min(range(r["count"]), key=lambda i: abs(r["lengths"][i] - min(limit, r["lengths"][-1]))) for r in rows]
        early = sum(i < r["count"] - 1 for r, i in zip(rows, indexes))
        errors = sum(i < r["count"] - 1 and not r["complete"][i] for r, i in zip(rows, indexes))
        retained = sum(r["lengths"][i] for r, i in zip(rows, indexes))
        assert (summary["queries"], summary["early_stops"], summary["false_early_stops"]) == (len(rows), early, errors)
        assert summary["conditional_error"] == (errors / early if early else None)
        assert math.isclose(summary["mean_retained"], retained / len(rows), abs_tol=1e-12)
        assert math.isclose(summary["retained_fraction"], retained / sum(r["lengths"][-1] for r in rows))
        assert math.isclose(summary["mean_global_judged_recall"], statistics.mean(
            r["positive_counts"][i] / max(1, r["total_relevant"]) for r, i in zip(rows, indexes)), abs_tol=1e-12)
        assert math.isclose(summary["mean_pool_gain_recall"], statistics.mean(
            r["gains"][i] / r["pool_gain"] if r["pool_gain"] else 0 for r, i in zip(rows, indexes)), abs_tol=1e-12)
        assert summary["complete_pool_queries"] == sum(r["complete"][i] for r, i in zip(rows, indexes))
        assert summary["no_positive_pool_queries"] == sum(r["pool_gain"] == 0 for r in rows)
    engineering = read("engineering/manifest.json")
    assert engineering["passed"] and engineering["discovered"] == 303 and engineering["skipped"] == 0
    assert engineering["cuda_visible_devices"] == ""
    assert len(engineering["tests"]) == len({r["test"] for r in engineering["tests"]}) == 303
    assert all(r["outcome"] == "passed" and math.isfinite(r["seconds"]) and r["seconds"] >= 0
               for r in engineering["tests"])
    assert sum(r["test"].startswith("test_acquisition.") for r in engineering["tests"]) == 13
    assert sha(directory / "engineering/tests.log") == engineering["test_log_sha256"]
    verify_sources(engineering["source_hashes"])
    assert engineering["example"]["returncode"] == 0
    example = json.loads(engineering["example"]["stdout"])
    assert example["plan"]["status"] == "unqualified_full_pool" and set(example["plan"]["indices"]) == {0, 1, 2}
    assert example["limited"]["status"] == "budget_exhausted" and example["limited"]["upper_error"] is None
    current = read("engineering-v2/manifest.json")
    assert current["passed"] and current["discovered"] == 304 and current["skipped"] == 0
    assert current["cuda_visible_devices"] == ""
    assert len(current["tests"]) == len({r["test"] for r in current["tests"]}) == 304
    assert all(r["outcome"] == "passed" for r in current["tests"])
    assert sum(r["test"].startswith("test_acquisition.") for r in current["tests"]) == 14
    assert sha(directory / "engineering-v2/tests.log") == current["test_log_sha256"]
    verify_sources(current["source_hashes"])
    assert current["example"]["returncode"] == 0
    for profile_name in ("latency", "latency-v2"):
        profile = read(profile_name + "/manifest.json")
        verify_sources(profile["source_hashes"])
        assert profile["model_sha256"] == sha(directory / "mlp32-seed29.json")
        assert profile["prediction_parity_max_error"] <= 1e-12 and profile["warmup_calls"] == 300
        assert len(profile["cases"]) == 100
        for name in ("scifact", "nfcorpus", "arguana", "scidocs", "fiqa"):
            assert [c["query_id"] for c in profile["cases"] if c["dataset"] == name] == assignments[name]["test"][:20]
        path = directory / profile_name / "timings.json.gz"
        assert sha(path) == profile["timing_sha256"]
        with gzip.open(path, "rb") as stream:
            raw = stream.read(4 * 1024 * 1024 + 1)
        assert len(raw) <= 4 * 1024 * 1024
        timings = json.loads(raw)
        methods = {"full-pool-order", "guarded-fallback", "eager-feature-control"}
        assert len(timings) == 9000
        assert {(r["case"], r["method"], r["seed"], r["repeat"]) for r in timings} == {
            (i, m, s, j) for i in range(100) for m in methods for s in (7, 29, 61) for j in range(10)}
        assert all(r["correct"] is True and type(r["nanoseconds"]) is int and r["nanoseconds"] > 0 for r in timings)
        for summary in profile["summary"]:
            values = sorted(r["nanoseconds"] / 1000000 for r in timings if r["method"] == summary["method"])
            assert summary["samples"] == len(values) == 3000
            assert summary["p50_ms"] == statistics.median(values)
            assert summary["p95_ms"] == values[math.ceil(.95 * len(values)) - 1]
    print("Six RTX fits, 788 calibration and 3677 regression trajectories verified. No stop policy qualified; full-pool fallback retained.")
    print("Initial 303 and optimized 304 engineering tests verified separately from model quality.")
    print("Two planner profiles replayed; published serial profile preserves all 9000 output orders.")


if __name__ == "__main__":
    main()

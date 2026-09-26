"""Check recorded CUDA capacity arithmetic and training-only provenance."""

import hashlib
import json
import math
import statistics
from pathlib import Path

from source_evidence import verify_sources

ROOT = Path(__file__).resolve().parents[1]


def main():
    report = json.loads((ROOT / "evidence/enterprise-gpu-v1/capacity.json").read_text(encoding="utf-8"))
    assert report["status"] == "measured" and report["failure_type"] is None
    assert report["window_seconds"] == 20 and len(report["rows"]) == len(report["case_order"]) == 8
    expected_cases = {(r, p, s) for s in (7, 29) for r in (False, True) for p in ("fp32", "bf16")}
    assert {(r["recurrent"], r["precision"], r["seed"]) for r in report["rows"]} == expected_cases
    verify_sources(report["source_hashes"])
    manifest_path = ROOT / "evidence/controller-v2/data-manifest.json"
    assert hashlib.sha256(manifest_path.read_bytes()).hexdigest() == report["data_manifest_sha256"]
    manifest = json.loads(manifest_path.read_text())
    assert report["training_partitions"] == [p for p in manifest["partitions"] if p["split"] == "train"]
    assert {p["dataset"] for p in report["training_partitions"]} == {"scifact", "nfcorpus"}
    for row, case in zip(report["rows"], report["case_order"]):
        assert all(row[key] == value for key, value in case.items())
        assert row["status"] == "measured" and row["batch"] == 128
        durations = row["step_seconds"]
        assert len(durations) == row["measured_steps"] > 0
        assert all(math.isfinite(v) and v > 0 for v in durations)
        assert sum(durations) <= row["window_seconds"] and row["window_seconds"] >= report["window_seconds"]
        assert math.isclose(row["queries_per_second"], 128 * len(durations) / row["window_seconds"], rel_tol=1e-12)
        assert math.isclose(row["median_step_ms"], statistics.median(durations) * 1000, rel_tol=1e-12)
        assert 0 < row["peak_allocated_bytes"] <= row["peak_reserved_bytes"] <= report["vram_bytes"]
        assert all(math.isfinite(row[key]) for key in ("minimum_loss", "maximum_loss"))
        assert 0 <= row["minimum_loss"] <= row["maximum_loss"]
        assert all(row["minimum_loss"] <= sample["loss"] <= row["maximum_loss"] for sample in row["sampled_losses"])
        numerical = row["precision_comparison"]
        assert all(math.isfinite(numerical[key]) for key in ("rmse", "max_abs_score_error"))
        assert 0 <= numerical["rmse"] <= numerical["max_abs_score_error"] + 1e-12
        assert 0 <= numerical["unchanged_top10_order"] <= numerical["unchanged_top1"] <= numerical["queries"] == 128
    summary = []
    for recurrent in (False, True):
        values = {}
        for precision in ("fp32", "bf16"):
            rows = [r for r in report["rows"] if r["recurrent"] == recurrent and r["precision"] == precision]
            values[precision] = statistics.median(r["queries_per_second"] for r in rows)
        summary.append(dict(recurrent=recurrent, **values, bf16_speed_ratio=values["bf16"] / values["fp32"]))
    telemetry = [list(map(float, r["sample"].split(","))) for r in report["gpu_telemetry"]]
    assert telemetry and all(len(r) == 4 and all(math.isfinite(v) for v in r) for r in telemetry)
    print(json.dumps(dict(training_capacity=summary, telemetry_samples=len(telemetry),
        median_gpu_utilization=statistics.median(r[0] for r in telemetry),
        peak_gpu_utilization=max(r[0] for r in telemetry), peak_temperature_c=max(r[2] for r in telemetry),
        unchanged_top10_order=sum(r["precision_comparison"]["unchanged_top10_order"] for r in report["rows"]),
        compared_training_lists=sum(r["precision_comparison"]["queries"] for r in report["rows"])), indent=2))
    print("Recorded training-only capacity and arithmetic checks; no held-out quality or deployment precision qualification.")


if __name__ == "__main__":
    main()

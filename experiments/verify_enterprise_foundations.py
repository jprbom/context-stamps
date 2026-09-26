"""Replay foundation-record arithmetic; not independent training replication."""

import hashlib
import json
import math
import statistics
from pathlib import Path

from source_evidence import verify_sources

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "evidence/enterprise-context-v1"


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    requirements = read(OUT / "requirements.json")
    assert [r["id"] for r in requirements["requirements"]] == [f"EC{i:02d}" for i in range(1, 21)]
    assert all((ROOT / p).is_file() for r in requirements["requirements"] for p in r["current_sources"])
    report = read(OUT / "gpu-capacity.json")
    assert report["data_manifest_sha256"] == sha(ROOT / "evidence/controller-v2/data-manifest.json")
    assert report["ranker_source_sha256"] == sha(ROOT / "context_stamps/contractive_controller.py")
    assert report["profile_source_sha256"] == sha(ROOT / "experiments/profile_context_training.py")
    assert {p["dataset"] for p in report["training_partitions"]} == {"scifact", "nfcorpus"}
    assert all(p["split"] == "train" for p in report["training_partitions"])
    assert len(report["rows"]) == 10
    for row in report["rows"]:
        assert row["status"] == "measured"
        durations = row["step_seconds"]
        assert len(durations) == len(row["training_losses"]) == report["measured_steps"]
        assert all(math.isfinite(v) and v > 0 for v in durations)
        assert all(math.isfinite(v) for v in row["training_losses"])
        assert math.isclose(row["median_step_ms"], statistics.median(durations) * 1000, rel_tol=1e-12)
        assert math.isclose(row["queries_per_second"], row["batch_size"] * len(durations) / sum(durations), rel_tol=1e-12)
        assert row["headroom_qualified"] == (row["peak_reserved_bytes"] <= .85 * report["vram_bytes"])
    for recurrent in (False, True):
        rows = [r for r in report["rows"] if r["recurrent"] == recurrent and r["headroom_qualified"]]
        assert max(rows, key=lambda r: r["queries_per_second"])["batch_size"] == report["selected_batch_by_recurrence"][str(recurrent)]
    for name in ("foundation-checks", "foundation-cuda"):
        directory = OUT / name
        manifest = read(directory / "manifest.json")
        tests = manifest["tests"]
        assert manifest["test_log_sha256"] == sha(directory / "tests.log")
        assert manifest["passed"] and manifest["isolated"]
        assert tests["passed"] == tests["discovered"] == 155
        assert not tests["skipped"] and not tests["failures"] and not tests["errors"]
        assert all(c["returncode"] == 0 for c in manifest["checks"])
        if name == "foundation-cuda":
            assert manifest["cuda"]["available"] and manifest["cuda"]["matmul_parity"]
            verify_sources(manifest["source_hashes"])
    print("20 requirements, 10 GPU capacity cases and two 155-test foundation records replayed.")
    print("Recorded evidence integrity only; no independent hardware reproduction or accuracy improvement claim.")


if __name__ == "__main__":
    main()

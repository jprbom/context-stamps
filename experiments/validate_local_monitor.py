"""Retain a reproducible simulated local adaptation/rollback cycle."""

import argparse
import gzip
import hashlib
import json
import platform
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from examples.local_adaptation_cycle import fixture_report  # noqa: E402

SOURCES = (
    "context_stamps/local_monitor.py", "context_stamps/local_learning.py", "context_stamps/local_policy.py",
    "context_stamps/decisions/calibration.py", "context_stamps/experience.py", "context_stamps/security.py",
    "tests/test_local_monitor.py", "examples/local_adaptation_cycle.py",
    "experiments/validate_local_monitor.py", "experiments/verify_local_monitor.py",
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    with tempfile.TemporaryDirectory() as directory:
        report = fixture_report(directory)
    payload = json.dumps(report, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    result = gzip.compress(payload, mtime=0)
    (args.output/"fixture.json.gz").write_bytes(result)
    manifest = dict(evidence_class=report["evidence_class"], model_calls=0, gpu_training_runs=0,
                    source_hashes={p: hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in SOURCES},
                    result_sha256=hashlib.sha256(result).hexdigest(), python=platform.python_version(),
                    simulation_only=True, device_or_model_benefit_claimed=False)
    (args.output/"manifest.json").write_text(json.dumps(manifest, indent=2)+"\n", encoding="utf-8", newline="\n")
    print(json.dumps(dict(training_tasks=len(report["training"])//2, evaluation_tasks=len(report["evaluation"]),
                         monitored_tasks=len(report["events"]), rollback=report["restored_baseline"], model_calls=0)))


if __name__ == "__main__":
    main()

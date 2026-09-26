"""Retain local-learning simulation and statistical boundary-test evidence."""

import argparse
import hashlib
import json
import platform
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from examples.local_learning import fixture_report  # noqa: E402


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    sources = ["context_stamps/local_learning.py", "context_stamps/local_policy.py",
               "context_stamps/decisions/calibration.py", "context_stamps/experience.py", "context_stamps/security.py",
               "examples/local_learning.py", "tests/test_local_learning.py", "experiments/validate_local_learning.py"]
    with tempfile.TemporaryDirectory() as temporary:
        report = fixture_report(temporary)
    result_path = args.out / "results.json"
    result_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8", newline="\n")
    run = subprocess.run([sys.executable, "-m", "unittest", "discover", "-s", "tests", "-p", "test_local_learning.py", "-v"],
                         cwd=ROOT, capture_output=True, text=True, check=False)
    log = args.out / "tests.txt"
    log.write_text(run.stdout + run.stderr, encoding="utf-8", newline="\n")
    manifest = {"schema": 1, "evidence_class": report["evidence_class"], "python": platform.python_version(),
                "platform": platform.platform(), "model_calls": 0, "gpu_training_runs": 0,
                "training_task_clusters": 60, "training_paired_action_observations": 120,
                "evaluation_fixture_clusters": 1200, "tests_exit_code": run.returncode,
                "tests_sha256": sha(log), "results_sha256": sha(result_path),
                "source_hashes": {name: sha(ROOT / name) for name in sources}}
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps(manifest, indent=2))
    if run.returncode:
        raise SystemExit(run.returncode)


if __name__ == "__main__":
    main()

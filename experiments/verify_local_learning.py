"""Replay simulated adaptation and its failures; not model benchmark evidence."""

import hashlib
import json
import math
import sys
import tempfile
from pathlib import Path

from source_evidence import verify_sources

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from examples.local_learning import fixture_report  # noqa: E402


def equal(a, b):
    if type(a) is float and type(b) is float:
        assert math.isclose(a, b, abs_tol=1e-12, rel_tol=1e-12)
    elif type(a) is dict and type(b) is dict:
        assert a.keys() == b.keys()
        for key in a:
            equal(a[key], b[key])
    elif type(a) is list and type(b) is list:
        assert len(a) == len(b)
        for x, y in zip(a, b):
            equal(x, y)
    else:
        assert a == b


def main():
    directory = ROOT / "evidence/local-learning-v1"
    manifest = json.loads((directory / "manifest.json").read_text())
    engineering = json.loads((directory / "engineering.json").read_text())
    full_log = directory / "full-suite.txt"
    assert hashlib.sha256(full_log.read_bytes()).hexdigest() == engineering["log_sha256"]
    assert engineering["tests"] == 328
    assert engineering["failures"] == engineering["errors"] == engineering["skipped"] == 0
    assert "Ran 328 tests" in full_log.read_text() and full_log.read_text().rstrip().endswith("OK")
    report = json.loads((directory / "results.json").read_text())
    for name in ("tests", "results"):
        path = directory / (name + (".txt" if name == "tests" else ".json"))
        assert hashlib.sha256(path.read_bytes()).hexdigest() == manifest[name + "_sha256"]
    verify_sources(manifest["source_hashes"])
    assert manifest["tests_exit_code"] == manifest["model_calls"] == manifest["gpu_training_runs"] == 0
    assert "Ran 24 tests" in (directory / "tests.txt").read_text()
    with tempfile.TemporaryDirectory() as temporary:
        repeated = json.loads(json.dumps(fixture_report(temporary)))
    equal(report, repeated)
    assert report["rounds"][0]["result"]["eligible"]
    assert not report["rounds"][1]["result"]["eligible"]
    assert report["active_before_rollback"] != report["active_after_rollback"]
    print("Local learning: 24 boundary tests, 60 training fixtures, 1200 evaluation fixtures, rejection and rollback replayed.")
    print("Simulated resources and task outcomes; no SLM or hardware qualification.")


if __name__ == "__main__":
    main()

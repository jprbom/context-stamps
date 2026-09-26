"""Record the current complete engineering suite separately from model quality."""

import argparse
import io
import json
import os
import platform
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from enterprise_state_validation import RecordedResult  # noqa: E402
from source_evidence import sha  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True, type=Path)
    out = parser.parse_args().out
    out.mkdir(exist_ok=False)
    log = io.StringIO()
    tests = unittest.TextTestRunner(stream=log, verbosity=2, resultclass=RecordedResult).run(
        unittest.defaultTestLoader.discover(str(ROOT / "tests")))
    (out / "tests.log").write_text(log.getvalue(), encoding="utf-8", newline="\n")
    example = subprocess.run([sys.executable, "examples/adaptive_acquisition.py", "--model",
        "evidence/acquisition-v1/mlp32-seed29.json"], cwd=ROOT, capture_output=True, text=True, timeout=30)
    sources = [*ROOT.glob("context_stamps/**/*.py"), *ROOT.glob("tests/*.py"), Path(__file__),
        ROOT / "experiments/enterprise_state_validation.py", ROOT / "experiments/source_evidence.py",
        ROOT / "examples/adaptive_acquisition.py"]
    payload = dict(schema=1, python=sys.version, platform=platform.platform(),
        source_base=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        source_hashes={p.relative_to(ROOT).as_posix(): sha(p.read_bytes()) for p in sorted(sources)},
        tests=tests.rows, discovered=tests.testsRun, skipped=len(tests.skipped), passed=tests.wasSuccessful(),
        test_log_sha256=sha((out / "tests.log").read_bytes()), cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
        example=dict(returncode=example.returncode, stdout=example.stdout, stderr=example.stderr))
    (out / "manifest.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({k: payload[k] for k in ("discovered", "skipped", "passed")}))
    if not tests.wasSuccessful() or tests.skipped or example.returncode:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

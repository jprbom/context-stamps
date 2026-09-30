"""Replay the authored numerical task example and its recorded source identity."""

import contextlib
import hashlib
import io
import json
import runpy
from pathlib import Path

from source_evidence import verify_sources

ROOT = Path(__file__).resolve().parents[1]


def verify():
    directory = ROOT/"evidence/numerical-tasks-v1"
    manifest = json.loads((directory/"manifest.json").read_bytes())
    for name, fingerprint in manifest["files"].items():
        path = (directory/name).resolve()
        if directory not in path.parents or hashlib.sha256(path.read_bytes()).hexdigest() != fingerprint:
            raise ValueError("numerical evidence artifact changed")
    verify_sources(manifest["source_hashes"])
    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        runpy.run_path(str(ROOT/"examples/numerical_tasks.py"), run_name="__main__")
    actual = json.loads(output.getvalue())
    expected = json.loads((directory/"example.json").read_bytes())
    if actual != expected or actual["model_calls"] != 0:
        raise ValueError("authored task execution replay differs")
    print("Declared numerical tasks, missing selector and source revocation replay passed; zero model calls.")


if __name__ == "__main__":
    verify()

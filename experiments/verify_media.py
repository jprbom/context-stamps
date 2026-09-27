"""Replay the authored WAV/lineage example; no model quality is measured."""

import contextlib
import hashlib
import io
import json
import runpy
from pathlib import Path

from source_evidence import verify_sources

ROOT = Path(__file__).resolve().parents[1]


def main():
    folder = ROOT / "evidence/media-v1"
    manifest = json.loads((folder / "manifest.json").read_bytes())
    for name, fingerprint in manifest["files"].items():
        if name not in ("example.json", "core-tests.txt"):
            raise ValueError("unexpected media evidence file")
        if hashlib.sha256((folder / name).read_bytes()).hexdigest() != fingerprint:
            raise ValueError("media evidence bytes changed")
    verify_sources(manifest["source_hashes"])
    output = io.StringIO()
    # Execute only the reviewed current example, never an archived source blob.
    with contextlib.redirect_stdout(output):
        runpy.run_path(str(ROOT / "examples/media_context.py"), run_name="__main__")
    actual = [json.loads(line) for line in output.getvalue().splitlines()]
    expected = json.loads((folder / "example.json").read_bytes())
    if actual != expected or actual[0]["model_calls"] != 0:
        raise ValueError("authored media example differs from retained result")
    print("Authored WAV lineage and revocation replay passed; no model/perception benchmark.")


if __name__ == "__main__":
    main()

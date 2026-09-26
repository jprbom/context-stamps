"""Replay recorded callback regression logs and exact source provenance."""

import hashlib
import json
import re
from pathlib import Path

from source_evidence import verify_sources

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "evidence/runtime-callbacks-v1"


def main():
    manifest = json.loads((OUT / "manifest.json").read_text(encoding="utf-8"))
    verify_sources(manifest["sources"])
    assert manifest["scope"] == "engineering regression; no model-quality or production-speed claim"
    for name, digest in manifest["files"].items():
        assert Path(name).name == name
        raw = (OUT / name).read_bytes()
        assert hashlib.sha256(raw).hexdigest() == digest, name
    for name, result in manifest["suites"].items():
        log = (OUT / name).read_text(encoding="utf-8")
        count = re.search(r"Ran (\d+) tests in", log)
        assert count and int(count.group(1)) == result["tests"]
        skipped = re.search(r"OK \(skipped=(\d+)\)", log)
        assert (int(skipped.group(1)) if skipped else 0) == result["skipped"]
        assert re.search(r"\nOK(?: \(skipped=\d+\))?\s*$", log)
        assert log.count("(test_runtime_callbacks.RuntimeCallbackTests.") == 11
    print("runtime-callbacks-v1: source and regression logs verified; no speedup inferred")


if __name__ == "__main__":
    main()

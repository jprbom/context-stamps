"""Replay the retained preparation metadata and its archived source bindings."""

import json
from pathlib import Path

from ruler_native import sha
from source_evidence import checked_path, verify_sources

ROOT = Path(__file__).resolve().parents[1]


def verify():
    folder = ROOT/"evidence/lme-relations-preparation-v1"
    manifest = json.loads((folder/"manifest.json").read_bytes())
    for name, digest in manifest["files"].items():
        if sha(checked_path(name, folder)) != digest:
            raise ValueError("superseded preparation changed")
    if manifest["model_calls"] != 0 or sha(checked_path(manifest["archive"])) != manifest["archive_sha256"]:
        raise ValueError("archived preparation identity changed")
    verify_sources(manifest["frozen_sources"])
    verify_sources(json.loads((folder/"registration.json").read_bytes())["sources"])
    return dict(superseded_preparation_verified=True, model_calls=0)


if __name__ == "__main__":
    print(json.dumps(verify()))

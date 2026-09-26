"""Replay the explicitly simulated cycle; never treat it as a model score."""

import gzip
import hashlib
import json
import math
import sys
import tempfile
from pathlib import Path

from source_evidence import verify_sources

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from examples.local_adaptation_cycle import fixture_report  # noqa: E402


def same_record(left, right):
    """Exact structure/types/decisions; allow only roundoff in finite floats."""
    if type(left) is not type(right):
        return False
    if type(left) is dict:
        return left.keys() == right.keys() and all(same_record(left[k], right[k]) for k in left)
    if type(left) is list:
        return len(left) == len(right) and all(same_record(a, b) for a, b in zip(left, right))
    if type(left) is float:
        return math.isfinite(left) and math.isfinite(right) and math.isclose(left, right, rel_tol=1e-12, abs_tol=1e-12)
    return left == right


def main():
    directory = ROOT/"evidence/local-monitor-v1"
    manifest = json.loads((directory/"manifest.json").read_bytes())
    verify_sources(manifest["source_hashes"])
    raw = (directory/"fixture.json.gz").read_bytes()
    if hashlib.sha256(raw).hexdigest() != manifest["result_sha256"]:
        raise ValueError("fixture evidence changed")
    recorded = json.loads(gzip.decompress(raw))
    with tempfile.TemporaryDirectory() as temporary:
        replay = fixture_report(temporary)
    if not same_record(json.loads(json.dumps(replay, allow_nan=False)), recorded):
        raise ValueError("simulated cycle did not reproduce")
    if (manifest["model_calls"] != 0 or not manifest["simulation_only"]
            or manifest["device_or_model_benefit_claimed"] or not recorded["restored_baseline"]):
        raise ValueError("simulation provenance or rollback failed")
    print(json.dumps({"reproduced": True, "evidence_class": recorded["evidence_class"], "model_calls": 0,
                      "monitored_tasks": len(recorded["events"]), "rollback": recorded["restored_baseline"]}))


if __name__ == "__main__":
    main()

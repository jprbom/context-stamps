"""Replay authored scorer checks and verify the published preparation record.

No model, network, source dataset or answer key is loaded by this verifier.
Dataset reproduction requires the separately documented publisher downloads.
Copyright (c) 2026 Prashant Jagtap. MIT License.
"""

import base64
import gzip
import hashlib
import json
from pathlib import Path

from chartqa_score import relaxed_correctness

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT/"evidence"/"chartqa-preparation-v1"


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def verify():
    manifest = json.loads((EVIDENCE/"manifest.json").read_bytes())
    for name, fingerprint in manifest["files"].items():
        path = (EVIDENCE/name).resolve()
        if EVIDENCE not in path.parents or sha(path.read_bytes()) != fingerprint:
            raise ValueError("published preparation artifact changed: "+name)
    for name, fingerprint in manifest["source_lf_sha256"].items():
        path = (ROOT/name).resolve()
        if ROOT not in path.parents or sha(path.read_bytes().replace(b"\r\n", b"\n")) != fingerprint:
            raise ValueError("registered experiment source changed: "+name)
    data = json.loads((EVIDENCE/"prepared-manifest.json").read_bytes())
    expected = dict(train=dict(images=64, questions=143), val=dict(images=32, questions=78),
                    test=dict(images=64, questions=145))
    if data["counts"] != expected or len(data["files"]) != 172:
        raise ValueError("prepared cohort inventory changed")
    if type(data["files"]) is not list or len({r["file"] for r in data["files"]}) != 172:
        raise ValueError("unique filename/digest records required")
    for record in data["files"]:
        if set(record) != {"file", "sha256"} or len(record["sha256"]) != 64:
            raise ValueError("prepared fingerprint format changed")
    if data["cross_split_exact_pixel_overlap"] != {"train/val": 0, "train/test": 0, "val/test": 0}:
        raise ValueError("prepared overlap report changed")
    with gzip.open(EVIDENCE/"source-snapshots.json.gz", "rb") as stream:
        raw = stream.read(1024*1024+1)
    if len(raw) > 1024*1024:
        raise ValueError("snapshot size limit")
    snapshots = json.loads(raw)
    for name, record in snapshots.items():
        if sha(base64.b64decode(record["base64"], validate=True)) != record["sha256"]:
            raise ValueError("historical preparation source changed: "+name)
    registration = json.loads((EVIDENCE/"registration.json").read_bytes())
    for name, fingerprint in registration["source_hashes"].items():
        if snapshots["native-success-v2/"+name]["sha256"] != fingerprint:
            raise ValueError("successful source snapshot does not match preparation")
    scorer = json.loads((EVIDENCE/"scorer-canaries.json").read_bytes())
    for row in scorer["canaries"]:
        if relaxed_correctness(row["target"], row["prediction"]) != row["reference"] or row["local"] != row["reference"]:
            raise ValueError("reference scorer canary differs")
    if len(scorer["canaries"]) != 21:
        raise ValueError("incomplete reference canaries")
    print(json.dumps(dict(preparation_fingerprints_verified=len(manifest["files"]),
        scorer_canaries=21, charts=160, questions=366, model_calls=0,
        scope="Published fingerprints and authored metric replay, not re-downloaded data or model evaluation")))


if __name__ == "__main__":
    verify()

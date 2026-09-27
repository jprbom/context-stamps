"""Publish authored local interface evidence, never third-party dataset bytes.

Copyright (c) 2026 Prashant Jagtap. MIT License.
"""

import argparse
import base64
import gzip
import json
from pathlib import Path

from chartqa_prepare import file_digest, outside, write_new
from chartqa_protocol import digest
from verify_chartqa_scalar import replay

ROOT = Path(__file__).resolve().parents[1]


def export(scalar, program):
    destination = ROOT/"evidence/chartqa-scalar-v2"
    summaries, archives = {}, {}
    for label, folder in (("scalar-v2", outside(scalar)), ("program-v3", outside(program))):
        complete = json.loads((folder/"complete.json").read_bytes())
        blobs = {name: (folder/name).read_bytes() for name in (*complete["files"], "complete.json")}
        summaries[label] = replay(blobs)
        archives[label] = {name: dict(sha256=digest(raw), base64=base64.b64encode(raw).decode()) for name, raw in blobs.items()}
    destination.mkdir(exist_ok=False)
    for name, records in archives.items():
        (destination/(name+".json.gz")).write_bytes(gzip.compress(json.dumps(records, sort_keys=True).encode(), mtime=0))
    write_new(destination/"summary.json", summaries)
    for name in ("earlier", "fresh", "new-position"):
        (destination/(name+".png")).write_bytes((outside(program)/(name+".png")).read_bytes())
    write_new(destination/"manifest.json", dict(evidence_class="authored_interface_development_not_model_learning",
        measured_calls=sum(r["calls"] for r in summaries.values()),
        files={p.name: file_digest(p) for p in destination.iterdir() if p.is_file()}))
    print(json.dumps(summaries))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scalar", type=Path, required=True)
    parser.add_argument("--program", type=Path, required=True)
    args = parser.parse_args()
    export(args.scalar, args.program)

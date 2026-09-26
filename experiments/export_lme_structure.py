"""Export measured outputs and short native references; no raw histories."""

import argparse
import gzip
import json
import shutil
from pathlib import Path

from ruler_native import sha, write_new

ROOT = Path(__file__).resolve().parents[1]


def export(run, output):
    output.mkdir(parents=True, exist_ok=False)
    records = json.loads((run/"scores.json").read_bytes())
    # Keep source references, but name them as view identities rather than keys.
    for row in records:
        row["sources"] = [{("view_id" if k == "key" else k): v for k, v in ref.items()} for ref in row["sources"]]
    payload = dict(scores=records, references=json.loads((run/"keys.json").read_bytes()))
    expanded = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode()
    (output/"records.json.gz").write_bytes(gzip.compress(expanded, mtime=0))
    names = ["registration.json", "prepared.json", "summary.json", "index.json", "projection-check.json",
             "qwen2.5-1.5b-completed.json", "qwen2.5-coder-7b-completed.json"]
    for name in names:
        shutil.copyfile(run/name, output/name)
    files = {p.name: sha(p) for p in output.iterdir() if p.is_file()}
    tool_names = ("export_lme_structure.py", "verify_lme_structure.py", "verify_structure_projection.py")
    write_new(output/"manifest.json", dict(files=files, expanded_bytes=len(expanded),
              tool_sources={"experiments/"+n: sha(ROOT/"experiments"/n) for n in tool_names}))
    print(json.dumps(dict(records=len(records), compressed_bytes=len((output/"records.json.gz").read_bytes()), expanded_bytes=len(expanded))))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    export(args.run, args.output)

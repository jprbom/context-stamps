"""Export measured outputs, source references and native keys; not raw histories."""

import argparse
import gzip
import json
import shutil
from pathlib import Path

from ruler_native import sha, write_new

ROOT = Path(__file__).resolve().parents[1]


def export(run, output):
    output.mkdir(parents=True, exist_ok=False)
    scores = json.loads((run/"scores.json").read_bytes())
    payload = dict(scores=scores, references=json.loads((run/"references.json").read_bytes()))
    expanded = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode()
    (output/"records.json.gz").write_bytes(gzip.compress(expanded, mtime=0))
    names = ("registration.json", "prepared.json", "summary.json", "index.json", "source-audit.json", "failure-review.json",
             "qwen2.5-1.5b-completed.json", "qwen2.5-coder-7b-completed.json")
    for name in names:
        shutil.copyfile(run/name, output/name)
    tools = ("export_lme_relations.py", "verify_lme_relations.py", "review_lme_relations.py", "verify_relation_sources.py")
    write_new(output/"manifest.json", dict(files={p.name: sha(p) for p in output.iterdir()}, expanded_bytes=len(expanded),
        tool_sources={"experiments/"+n: sha(ROOT/"experiments"/n) for n in tools}))
    print(json.dumps(dict(records=len(scores), expanded_bytes=len(expanded), compressed_bytes=(output/"records.json.gz").stat().st_size)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    export(args.run, args.output)

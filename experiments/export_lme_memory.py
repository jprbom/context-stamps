"""Export short public-benchmark predictions and native references, never prompts."""

import argparse
import gzip
import json
from pathlib import Path

from ruler_native import outside_repo, sha, write_new


def export(run, output):
    output = outside_repo(output)
    output.mkdir(parents=True, exist_ok=False)
    rows = json.loads((run/"scores.json").read_bytes())
    for phase in ("train", "holdout"):
        raw = [json.loads(s) for s in (run/f"{phase}-generations.jsonl").read_text(encoding="utf-8").splitlines()]
        complete = json.loads((run/f"{phase}-completed.json").read_bytes())
        selected = [{k: v for k, v in r.items() if k != "correct"} for r in rows if r["phase"] == phase and r["arm"] != "learned"]
        if (raw != selected or complete["records"] != len(raw)
                or sha(run/f"{phase}-generations.jsonl") != complete["sha256"]):
            raise ValueError("Raw generation log changed or incomplete")
    if any(len(r["prediction"].encode()) > 8192 for r in rows):
        raise ValueError("Oversized prediction requires separate disclosure review")
    inputs = json.loads((run/"inputs.json").read_bytes())
    prepared = json.loads((run/"prepared.json").read_bytes())
    if sha(run/"inputs.json") != prepared["inputs_sha256"] or sha(run/"keys.json") != prepared["keys_sha256"]:
        raise ValueError("Prepared inputs or keys changed")
    encoded = json.dumps(dict(scores=rows, references=json.loads((run/"keys.json").read_bytes()),
                              features=[dict(id=r["id"], features=r["features"]) for r in inputs]), ensure_ascii=False).encode()
    (output/"records.json.gz").write_bytes(gzip.compress(encoded, mtime=0))
    for name in ("registration", "prepared", "policy", "holdout-choices", "summary", "index"):
        write_new(output/f"{name}.json", json.loads((run/f"{name}.json").read_bytes()))
    write_new(output/"manifest.json", dict(schema=1, files={p.name: sha(p) for p in output.iterdir()},
        original_logs={p: sha(run/f"{p}-generations.jsonl") for p in ("train", "holdout")}, expanded_bytes=len(encoded),
        disclosure="Short public-benchmark predictions/references and source identities; no raw histories, prompts or private records"))
    print(json.dumps(dict(records=len(rows), expanded_bytes=len(encoded))))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    export(args.run, args.output)

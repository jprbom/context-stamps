"""Export bounded public-benchmark outputs, never long prompts or token IDs."""

import argparse
import gzip
import json
from pathlib import Path

from ruler_native import outside_repo, sha, write_new


def export(run, train, holdout, output):
    output = outside_repo(output)
    output.mkdir(parents=True, exist_ok=False)
    records = {}
    for phase in ("train", "holdout"):
        complete = json.loads((run/f"{phase}-completed.json").read_bytes())
        if sha(run/f"{phase}-generations.jsonl") != complete["sha256"]:
            raise ValueError("Generation log changed")
        rows = json.loads((run/f"{phase}-scored.json").read_bytes())
        raw = [json.loads(line) for line in (run/f"{phase}-generations.jsonl").read_text(encoding="utf-8").splitlines()]
        if len(rows) != complete["records"] or len(raw) != len(rows):
            raise ValueError("Incomplete retained records")
        for row, original in zip(rows, raw):
            if {k: v for k, v in row.items() if k not in ("native_score", "strict_correct")} != original:
                raise ValueError("Scored generation changed")
            if len(row["prediction"].encode()) > 4096:
                raise ValueError("Long prediction needs a separate rights review")
            if any(k in row for k in ("context", "question", "prompt", "prompts")):
                raise ValueError("Raw source field cannot be published")
        records[phase] = rows
    for name in ("references", "input-identities"):
        records[name] = json.loads((run/f"{name}.json").read_bytes())
    registration = json.loads((run/"registration.json").read_bytes())
    for name, data in (("training", train), ("holdout", holdout)):
        for field, filename in (("plan_sha256", "plan.json"), ("inventory_sha256", "inventory.json")):
            if sha(data/filename) != registration[name][field]:
                raise ValueError("Frozen data registration changed")
        records[name+"-data"] = dict(plan=json.loads((data/"plan.json").read_bytes()),
                                    inventory=json.loads((data/"inventory.json").read_bytes()))
    encoded = json.dumps(records, ensure_ascii=False).encode()
    (output/"records.json.gz").write_bytes(gzip.compress(encoded, mtime=0))
    for name in ("registration", "registered-trial", "policy", "promotion", "summary"):
        write_new(output/f"{name}.json", json.loads((run/f"{name}.json").read_bytes()))
    write_new(output/"policy-data.json", json.loads((run/"policy.json").read_bytes())["policy"])
    write_new(output/"manifest.json", dict(schema=1, files={p.name: sha(p) for p in output.iterdir() if p.is_file()},
        original_logs={phase: sha(run/f"{phase}-generations.jsonl") for phase in ("train", "holdout")},
        expanded_bytes=len(encoded), disclosure="Short public-benchmark outputs and references only; raw prompts/token IDs excluded",
        replay="Offline metrics/policy/gate replay; regenerating and verifying context requires the separate licensed source data"))
    print(json.dumps(dict(training=len(records["train"]), holdout=len(records["holdout"]), expanded_bytes=len(encoded))))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("run", "train", "holdout", "output"):
        parser.add_argument("--"+name, type=Path, required=True)
    args = parser.parse_args()
    export(args.run, args.train, args.holdout, args.output)

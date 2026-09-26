"""Export short predictions and metadata, excluding copyrighted long prompts.

Copyright (c) 2026 Prashant Jagtap. MIT License.
Inspect prediction text before publishing; this exporter enforces size only.
"""

import argparse
import gzip
import json
from pathlib import Path

from ruler_native import outside_repo, sha, write_new


def export(data, run, output):
    output = outside_repo(output)
    output.mkdir(parents=True, exist_ok=False)
    summary = json.loads((run/"summary.json").read_bytes())
    plan = json.loads((run/"plan.json").read_bytes())
    completed = json.loads((run/"completed.json").read_bytes())
    if sha(run/"generations.jsonl") != completed["generations_sha256"] or sha(run/"inputs.json") != plan["inputs_sha256"]:
        raise ValueError("Original generation/input bytes changed")
    inputs = json.loads((run/"inputs.json").read_bytes())
    metadata = [{k: v for k, v in row.items() if k != "prompts"} for row in inputs]
    inventory = json.loads((data/"inventory.json").read_bytes())
    references = []
    for entry in inventory:
        path = data/str(entry["length"])/entry["task"]/"keys.json"
        if sha(path) != entry["keys_sha256"]:
            raise ValueError("Native references changed")
        references.extend(json.loads(path.read_bytes()))
    fields = {"model", "created_at", "response", "done", "done_reason", "total_duration", "load_duration",
              "prompt_eval_count", "prompt_eval_duration", "eval_count", "eval_duration"}
    records = []
    for line in (run/"generations.jsonl").read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        result = row["response"]
        if result is not None:
            if len(result.get("response", "")) > 2048:
                raise ValueError("Long prediction requires a separate rights review")
            # Never export Ollama context token IDs: those may reconstruct source text.
            row["response"] = {k: v for k, v in result.items() if k in fields}
        records.append(row)
    raw = json.dumps(dict(inputs=metadata, references=references, generations=records), ensure_ascii=False).encode()
    (output/"records.json.gz").write_bytes(gzip.compress(raw, mtime=0))
    for name, value in (("summary.json", summary), ("registration.json", plan),
                        ("data-plan.json", json.loads((data/"plan.json").read_bytes())),
                        ("data-inventory.json", inventory)):
        write_new(output/name, value)
    write_new(output/"manifest.json", dict(schema=1,
        files={p.name: sha(p) for p in output.iterdir() if p.is_file()},
        original_generations_sha256=sha(run/"generations.jsonl"),
        scope="Short predictions/references and metadata; no prompts, essay text or context-token IDs",
        verification="Public replayer recomputes metrics from exported predictions. Full context-selection replay needs locally downloaded data."))
    print(json.dumps(dict(records=len(records), uncompressed_bytes=len(raw), output=str(output))))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    export(args.data, args.run, args.output)

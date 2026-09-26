"""Offline metric replay from short RULER predictions; no model or corpus fetch."""

import gzip
import hashlib
import json
import math
import statistics
from pathlib import Path

from ruler_context import native_score
from source_evidence import verify_sources

ROOT = Path(__file__).resolve().parents[1]


def validate(records, summary):
    inputs = records["inputs"]
    lookup = {row["id"]: row for row in inputs}
    refs = {row["id"]: row for row in records["references"]}
    generations = records["generations"]
    arms = ("full", "bm25", "tool_context", "runtime")
    if len(inputs) != 52 or len(lookup) != len(inputs) or len(refs) != len(inputs) or set(refs) != set(lookup):
        raise ValueError("Expected 52 uniquely paired samples/references")
    expected = {(key, arm) for key in lookup for arm in arms}
    if len(generations) != len(expected) or {(r["id"], r["arm"]) for r in generations} != expected:
        raise ValueError("Missing/duplicate treatment")
    scores, costs = {}, {}
    for row in generations:
        source = lookup[row["id"]]
        reference = refs[row["id"]]
        if row["task"] != source["task"] or row["length"] != source["length"]:
            raise ValueError("Task identity mismatch")
        response = row["response"] or {}
        if "context" in response or "prompts" in source or "question" in source:
            raise ValueError("Raw long context must not be exported")
        metadata = source["metadata"][row["arm"]]
        if not row["error"] and (not response.get("done") or response.get("prompt_eval_count") != metadata["input_tokens"]):
            raise ValueError("Successful record violates input-token parity")
        if not math.isfinite(row["wall_seconds"]) or row["wall_seconds"] < 0:
            raise ValueError("Invalid timing")
        key = (row["id"], row["arm"])
        scores[key] = 0. if row["error"] else native_score(response.get("response", ""), reference["expected_answer"], reference["match_type"])
        costs[key] = row["wall_seconds"]+metadata["preparation_seconds"]
    if summary["samples"] != len(inputs) or summary["calls"] != len(generations):
        raise ValueError("Summary denominator mismatch")
    if summary["errors"] != sum(bool(r["error"]) for r in generations):
        raise ValueError("Errors omitted")
    if summary["truncations"] != sum((r["response"] or {}).get("done_reason") == "length" for r in generations):
        raise ValueError("Truncations omitted")
    for arm in arms:
        group = [r for r in generations if r["arm"] == arm]
        values = [scores[r["id"], arm] for r in group]
        expected = dict(native_mean=statistics.mean(values), complete_credit=sum(v == 1 for v in values),
            input_tokens=sum(lookup[r["id"]]["metadata"][arm]["input_tokens"] for r in group),
            output_tokens=sum((r["response"] or {}).get("eval_count", 0) for r in group),
            total_seconds=sum(costs[r["id"], arm] for r in group))
        if expected != summary["overall"][arm]:
            raise ValueError("Overall metric/cost mismatch")
    cells = {(row["length"], row["task"]) for row in inputs}
    if len(summary["cells"]) != 26 or {(r["length"], r["task"]) for r in summary["cells"]} != cells:
        raise ValueError("All task/length cells required")
    for cell in summary["cells"]:
        selected = [r for r in inputs if (r["length"], r["task"]) == (cell["length"], cell["task"])]
        for arm in arms:
            expected = dict(n=len(selected), score=statistics.mean(scores[r["id"], arm] for r in selected),
                input_tokens=sum(r["metadata"][arm]["input_tokens"] for r in selected),
                total_seconds=sum(costs[r["id"], arm] for r in selected))
            if expected != cell["arms"][arm]:
                raise ValueError("Cell mismatch")
    direct = [native_score(row["direct_answer"] or "", refs[row["id"]]["expected_answer"],
                           refs[row["id"]]["match_type"]) for row in inputs]
    expected = dict(native_mean=statistics.mean(direct), complete_credit=sum(v == 1 for v in direct),
        abstentions=sum(row["direct_answer"] is None for row in inputs), model_calls=0)
    if summary["direct_control"] != expected:
        raise ValueError("Deterministic control mismatch")
    return dict(samples=len(inputs), calls=len(generations), errors=summary["errors"],
                truncations=summary["truncations"], native_metrics_match=True)


def verify(path):
    manifest = json.loads((path/"manifest.json").read_bytes())
    for name, expected in manifest["files"].items():
        if Path(name).name != name or hashlib.sha256((path/name).read_bytes()).hexdigest() != expected:
            raise ValueError("Evidence identity mismatch")
    registration = json.loads((path/"registration.json").read_bytes())
    verify_sources(registration["sources"])
    records = json.loads(gzip.decompress((path/"records.json.gz").read_bytes()))
    summary = json.loads((path/"summary.json").read_bytes())
    return validate(records, summary)


if __name__ == "__main__":
    print(json.dumps(verify(ROOT/"evidence/ruler-development-v1")))

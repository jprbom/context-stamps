"""Predeclared authored schema-grounding comparison; no public dataset questions.

Copyright (c) 2026 Prashant Jagtap. MIT License.
All variants and expected answers are registered before these local calls.
"""

import argparse
import json
import random
from pathlib import Path

from chartqa_canary import SOURCE_FILES, capture
from chartqa_prepare import file_digest, outside, write_new
from chartqa_protocol import MODEL, bind_table, digest, encoded, request_body
from local_eval import local_api, model_identity

ROOT = Path(__file__).resolve().parents[1]
CASES = (
    ("sum", "What is the sum of completed tasks for Alpha and Beta?", "60"),
    ("lookup", "How many tasks did Alpha complete?", "20"),
    ("ratio", "What is the ratio of completed tasks for Gamma to Alpha?", "3"),
    ("category", "Which category completed the most tasks?", "Gamma"),
)
VARIANTS = ("baseline", "grounded_schema", "json_only")


def variant_request(variant, mode, *, question, image=None, memory=None):
    if variant not in VARIANTS or mode not in ("direct", "memory"):
        raise ValueError("declared interface variant required")
    body = request_body(mode, question=question, image=image, memory=memory)
    if variant == "grounded_schema":
        body["messages"][0]["content"] += " The complete required response JSON schema is: " + encoded(body["format"]).decode()
    elif variant == "json_only":
        body["format"] = "json"
    return body


def run(prior, output):
    prior, output = outside(prior), outside(output)
    complete, registration = (json.loads((prior/name).read_bytes()) for name in ("complete.json", "registration.json"))
    if complete["passed"] or complete["model_calls"] != 4 or not complete["model_unchanged"]:
        raise ValueError("this diagnostic follows the retained four-call interface failure")
    identity = model_identity(MODEL)
    if identity != registration["model"]:
        raise ValueError("same installed local weights and configuration required")
    # The preceding canary owns the possible resident instance. Do not evict or
    # replace another model. The host must first verify its prior process ended.
    if any(r["name"] != MODEL or r["digest"] != identity["digest"] for r in local_api("/api/ps").get("models", [])):
        raise ValueError("unrelated resident model; wait without eviction")
    for name, expected in complete["files"].items():
        if file_digest(prior/name) != expected:
            raise ValueError("prior canary bytes changed")
    image = (prior/"authored-chart.png").read_bytes()
    table = json.loads((prior/"extract.parsed.json").read_bytes())["result"]["table"]
    source = dict(image_sha256=digest(image), byte_length=len(image), width=720, height=440)
    state, snapshot, memory, _ = bind_table(source, image, table, identity["digest"],
        digest(encoded(json.loads((prior/"extract.request.json").read_bytes()))))
    schedule = [(key, mode, variant) for key, _, _ in CASES for mode in ("direct", "memory") for variant in VARIANTS]
    random.Random(107).shuffle(schedule)
    output.mkdir(exist_ok=False)
    (output/"source").mkdir()
    hashes = {}
    for name in (*SOURCE_FILES, "experiments/chartqa_interface_probe.py"):
        raw = (ROOT/name).read_bytes()
        target = output/"source"/name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(raw)
        hashes[name] = digest(raw)
    write_new(output/"registration.json", dict(model=identity, server=local_api("/api/version"),
        prior_completion=file_digest(prior/"complete.json"), source_hashes=hashes, cases=CASES,
        variants=VARIANTS, call_order=schedule, maximum_calls=24, new_extractions=0,
        hypothesis="Including the response schema in the prompt may resolve schema-only answer artifacts.",
        interpretation="Authored interface development; not training or ChartQA evaluation; every variant retained."))
    results = []
    for key, mode, variant in schedule:
        _, question, expected = next(case for case in CASES if case[0] == key)
        if not state.is_current(snapshot):
            raise ValueError("source binding changed before inference")
        body = variant_request(variant, mode, question=question, image=image if mode == "direct" else None,
                               memory=memory if mode == "memory" else None)
        result = capture(output, key+"-"+mode+"-"+variant, body, mode)
        answer = (result["result"] or {}).get("answer")
        results.append(dict(case=key, mode=mode, variant=variant, answer=answer, expected=expected,
                            correct=not result["error"] and answer == expected, error=result["error"],
                            wall_seconds=result["wall_seconds"]))
        print(json.dumps(dict(completed=len(results), total=len(schedule))), flush=True)
    unchanged = model_identity(MODEL) == identity
    write_new(output/"complete.json", dict(model_unchanged=unchanged, results=results, model_calls=24,
        source_hashes=hashes, candidate_active=False,
        files={p.relative_to(output).as_posix(): file_digest(p) for p in output.rglob("*") if p.is_file()}))
    if not unchanged:
        raise ValueError("model identity changed during probe")
    print(json.dumps({variant: {mode: sum(r["correct"] for r in results if r["variant"] == variant and r["mode"] == mode)
                               for mode in ("direct", "memory")} for variant in VARIANTS}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prior", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.prior, args.output)

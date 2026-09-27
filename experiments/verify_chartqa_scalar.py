"""Offline replay of typed-scalar and program canaries, including failed calls.

Copyright (c) 2026 Prashant Jagtap. MIT License.
Historical source bytes are checked but never executed. No model is called.
"""

import base64
import gzip
import hashlib
import json
import math
from pathlib import Path

from chartqa_program_protocol import request_body as program_request
from chartqa_protocol import bind_table, digest, encoded
from chartqa_scalar_protocol import parse_response
from chartqa_scalar_protocol import request_body as scalar_request
from chartqa_score import score_answer

ROOT = Path(__file__).resolve().parents[1]


def load_archive(path):
    with gzip.open(path, "rb") as stream:
        content = stream.read(8*1024**2+1)
    if len(content) > 8*1024**2:
        raise ValueError("authored archive exceeds replay bound")
    records = json.loads(content)
    blobs = {}
    for name, record in records.items():
        if Path(name).is_absolute() or ".." in Path(name).parts:
            raise ValueError("invalid record identity")
        data = base64.b64decode(record["base64"], validate=True)
        if digest(data) != record["sha256"]:
            raise ValueError("historical record bytes changed")
        blobs[name] = data
    return blobs


def replay(blobs):
    def read(name):
        return json.loads(blobs[name])
    registered, complete = read("registration.json"), read("complete.json")
    version = registered["protocol"]
    if version not in ("typed-json-v2", "typed-json-program-v3"):
        raise ValueError("registered protocol required")
    make_request = scalar_request if version == "typed-json-v2" else program_request
    if not complete["model_unchanged"] or complete["candidate_active"]:
        raise ValueError("unchanged reader and inactive candidate required")
    if set(blobs) != set(complete["files"]) | {"complete.json"}:
        raise ValueError("archive contains missing or unregistered records")
    for name, fingerprint in complete["files"].items():
        if digest(blobs[name]) != fingerprint:
            raise ValueError("completion record identity differs")
    for name, fingerprint in registered["source_hashes"].items():
        if digest(blobs["source/"+name]) != fingerprint:
            raise ValueError("historical source identity differs")
    if registered["source_hashes"] != complete["source_hashes"]:
        raise ValueError("registration and completion source bindings differ")
    outcomes, costs, calls = [], {}, 0

    def check(stem, mode, body, question=None, table=None):
        nonlocal calls
        if read(stem+".request.json") != body:
            raise ValueError("request differs from registered protocol")
        raw = blobs[stem+".response.json"]
        observed = read(stem+".parsed.json")
        if observed["raw_sha256"] != digest(raw):
            raise ValueError("parsed result not bound to original reply")
        result, error = None, None
        try:
            result = parse_response(raw, mode, question=question, table=table)
        except (ValueError, TypeError, KeyError, UnicodeError) as exc:
            error = f"{type(exc).__name__}: {exc}"
        if result != observed["result"] or error != observed["error"]:
            raise ValueError("original result or failure changed")
        for name in ("wall_seconds", "processing_seconds"):
            if type(observed[name]) not in (int, float) or not math.isfinite(observed[name]) or observed[name] < 0:
                raise ValueError("finite nonnegative measured time required")
        outer = json.loads(raw)
        cost = costs.setdefault(mode, dict(calls=0, input_tokens=0, output_tokens=0, wall_seconds=0.0))
        cost["calls"] += 1
        cost["input_tokens"] += outer["prompt_eval_count"]
        cost["output_tokens"] += outer["eval_count"]
        cost["wall_seconds"] += observed["wall_seconds"]
        calls += 1
        return result, error, observed["wall_seconds"]

    for chart in registered["charts"]:
        key = chart["chart"]
        image = blobs[key+".png"]
        if digest(image) != chart["image_sha256"]:
            raise ValueError("authored image changed")
        body = make_request("extract", image=image)
        extraction, error, _ = check(key+"-extract", "extract", body)
        if error:
            raise ValueError("this archive requires completed extraction")
        table = extraction["table"]
        state, snapshot, memory, bundle = bind_table(dict(image_sha256=digest(image), byte_length=len(image), width=720, height=440),
            image, table, registered["model"]["digest"], digest(encoded(body)))
        required = {(c[0], m) for c in chart["cases"] for m in ("direct", "memory", "program")}
        if len(required) != 15 or len(chart["call_order"]) != 15 or set(map(tuple, chart["call_order"])) != required:
            raise ValueError("incomplete authored task coverage")
        for case, mode in chart["call_order"]:
            _, question, expected = next(c for c in chart["cases"] if c[0] == case)
            body = make_request(mode, question=question, image=image if mode == "direct" else None,
                memory=memory if mode != "direct" else None)
            result, error, wall = check(key+"-"+case+"-"+mode, mode, body, question, table)
            answer = (result or {}).get("answer")
            correct = not error and (answer is None if expected is None else score_answer([expected], answer)["relaxed"])
            outcomes.append(dict(chart=key, case=case, mode=mode, answer=answer, expected=expected,
                correct=correct, error=error, wall_seconds=wall))
        state.set_roles(bundle.source.key, ())
        if state.is_current(snapshot):
            raise ValueError("source revocation did not invalidate extraction")
    if outcomes != complete["results"] or calls != complete["model_calls"] or calls != registered["maximum_calls"]:
        raise ValueError("measured outcomes or request denominator differ")
    if complete["passed"] != all(r["correct"] for r in outcomes):
        raise ValueError("acceptance decision differs")
    return dict(protocol=version, passed=complete["passed"], calls=calls, costs=costs,
        answers_per_arm=len(outcomes)//3,
        correct={m: sum(r["correct"] for r in outcomes if r["mode"] == m) for m in ("direct", "memory", "program")},
        failures=[r for r in outcomes if not r["correct"]], model_calls_in_replay=0)


def verify():
    directory = ROOT/"evidence/chartqa-scalar-v2"
    manifest = json.loads((directory/"manifest.json").read_bytes())
    for name, fingerprint in manifest["files"].items():
        path = (directory/name).resolve()
        if directory not in path.parents or hashlib.sha256(path.read_bytes()).hexdigest() != fingerprint:
            raise ValueError("published artifact changed")
    results = {name: replay(load_archive(directory/(name+".json.gz"))) for name in ("scalar-v2", "program-v3")}
    if results != json.loads((directory/"summary.json").read_bytes()):
        raise ValueError("published summary differs from independent replay")
    print(json.dumps(results))


if __name__ == "__main__":
    verify()

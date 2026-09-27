"""Predeclared 32-call authored check of the typed-scalar interface.

Copyright (c) 2026 Prashant Jagtap. MIT License.
No public benchmark cases, training labels or paid providers are used.
"""

import argparse
import json
import os
import random
import time
from pathlib import Path

from chartqa_canary import SOURCE_FILES, create_chart
from chartqa_prepare import file_digest, outside, write_new
from chartqa_protocol import MODEL, bind_table, digest, encoded, local_chat
from chartqa_scalar_protocol import VERSION, parse_response, request_body
from chartqa_score import score_answer
from local_eval import local_api, model_identity

ROOT = Path(__file__).resolve().parents[1]
CHARTS = (
    ("earlier", ("Alpha", "Beta", "Gamma"), (20, 40, 60)),
    ("fresh", ("Nora", "Owen", "Pia"), (7, 13, 26)),
)


def cases(labels, values, lookup_index=0):
    a, b, c = labels
    return (
        ("sum", f"What is the sum of completed tasks for {a} and {b}?", str(values[0]+values[1])),
        ("lookup", f"How many tasks did {labels[lookup_index]} complete?", str(values[lookup_index])),
        ("ratio", f"What is the ratio of completed tasks for {c} to {b}?", str(values[2]/values[1])),
        ("category", "Which category completed the most tasks?", labels[values.index(max(values))]),
        ("absent", "How many tasks did Zeta complete?", None),
    )


def fresh_chart(path, font_path, *, labels=("Nora", "Owen", "Pia"), values=(7, 13, 26)):
    from PIL import Image, ImageDraw, ImageFont
    image = Image.new("RGB", (720, 440), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype(str(font_path), 24)
    draw.text((70, 20), "Authored fixture: completed tasks", fill="black", font=font)
    draw.line([(80, 80), (80, 350), (650, 350)], fill="black", width=2)
    for x, label, value in zip((170, 360, 550), labels, values):
        y = 350-value*min(9, 240/max(values))
        draw.rectangle((x-40, y, x+40, 350), fill="#577A6A")
        draw.text((x-16, y-32), str(value), fill="black", font=font)
        draw.text((x-36, 365), label, fill="black", font=font)
    image.save(path, format="PNG")


def capture(folder, name, body, mode, *, question=None, table=None):
    write_new(folder/(name+".request.json"), body)
    write_new(folder/(name+".started.json"), dict(utc=time.time(), pid=os.getpid()))
    started = time.perf_counter()
    try:
        raw = local_chat(body)
    except Exception as exc:
        write_new(folder/(name+".transport-failure.json"), dict(error=f"{type(exc).__name__}: {exc}",
            wall_seconds=time.perf_counter()-started, retry_allowed=False))
        raise
    wall = time.perf_counter()-started
    with (folder/(name+".response.json")).open("xb") as stream:
        stream.write(raw)
    tick, result, error = time.perf_counter(), None, None
    try:
        result = parse_response(raw, mode, question=question, table=table)
    except (ValueError, TypeError, KeyError, UnicodeError) as exc:
        error = f"{type(exc).__name__}: {exc}"
    outcome = dict(result=result, error=error, wall_seconds=wall,
        processing_seconds=time.perf_counter()-tick, raw_sha256=digest(raw))
    write_new(folder/(name+".parsed.json"), outcome)
    return outcome


def run(output, font, prior=None):
    version, make_request = VERSION, request_body
    charts = CHARTS
    output = outside(output)
    resident = local_api("/api/ps").get("models", [])
    if prior is None and resident:
        raise ValueError("resident local workload exists; wait without eviction")
    identity = model_identity(MODEL)
    if prior is not None:
        from chartqa_program_protocol import VERSION as program_version
        from chartqa_program_protocol import request_body as program_request
        prior = outside(prior)
        registered = json.loads((prior/"registration.json").read_bytes())
        finished = json.loads((prior/"complete.json").read_bytes())
        if registered["protocol"] != VERSION or registered["model"] != identity or not finished["model_unchanged"]:
            raise ValueError("completed same-model typed-scalar predecessor required")
        for name, expected in finished["files"].items():
            if file_digest(prior/name) != expected:
                raise ValueError("previous measured bytes changed")
        if any(r["name"] != MODEL or r["digest"] != identity["digest"] for r in resident):
            raise ValueError("unrelated resident workload; wait without eviction")
        version, make_request = program_version, program_request
        charts = (*CHARTS, ("new-position", ("Ravi", "Uma", "Tao"), (11, 44, 22)))
    if "vision" not in local_api("/api/show", dict(model=MODEL)).get("capabilities", []):
        raise ValueError("vision capability required")
    output.mkdir(exist_ok=False)
    hashes = {}
    for name in (*SOURCE_FILES, "experiments/chartqa_scalar_protocol.py", "experiments/chartqa_scalar_canary.py",
                 "experiments/chartqa_score.py", "experiments/chartqa_program_protocol.py"):
        raw = (ROOT/name).read_bytes()
        target = output/"source"/name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(raw)
        hashes[name] = digest(raw)
    create_chart(output/"earlier.png", font)
    fresh_chart(output/"fresh.png", font)
    if prior is not None:
        fresh_chart(output/"new-position.png", font, labels=charts[-1][1], values=charts[-1][2])
    registration = []
    for key, labels, values in charts:
        tasks = cases(labels, values, lookup_index=1 if key == "new-position" else 0)
        schedule = [(case[0], mode) for case in tasks for mode in ("direct", "memory", "program")]
        random.Random(109).shuffle(schedule)
        registration.append(dict(chart=key, cases=tasks, call_order=schedule,
            image_sha256=file_digest(output/(key+".png"))))
    write_new(output/"registration.json", dict(protocol=version, model=identity,
        server=local_api("/api/version"), source_hashes=hashes, font_sha256=file_digest(font),
        charts=registration, maximum_calls=16*len(charts),
        prior_experiment="chartqa-interface-v1" if prior is None else file_digest(prior/"complete.json"),
        purpose="Authored interface development; no ChartQA benchmark score or learned model update",
        acceptance="Every answer must match the authored reference or required abstention; no retries."))
    results, calls = [], 0
    for registered in registration:
        key = registered["chart"]
        image = (output/(key+".png")).read_bytes()
        body = make_request("extract", image=image)
        extraction = capture(output, key+"-extract", body, "extract")
        calls += 1
        if extraction["error"]:
            raise ValueError("extraction failed; retain this complete attempt without retries")
        table = extraction["result"]["table"]
        state, snapshot, memory, _ = bind_table(dict(image_sha256=digest(image), byte_length=len(image), width=720, height=440),
            image, table, identity["digest"], digest(encoded(body)))
        for case, mode in registered["call_order"]:
            _, question, expected = next(c for c in registered["cases"] if c[0] == case)
            if not state.is_current(snapshot):
                raise ValueError("source binding changed before request")
            body = make_request(mode, question=question, image=image if mode == "direct" else None,
                memory=memory if mode != "direct" else None)
            outcome = capture(output, key+"-"+case+"-"+mode, body, mode, question=question, table=table)
            calls += 1
            answer = (outcome["result"] or {}).get("answer")
            correct = not outcome["error"] and (answer is None if expected is None else score_answer([expected], answer)["relaxed"])
            results.append(dict(chart=key, case=case, mode=mode, answer=answer, expected=expected,
                correct=correct, error=outcome["error"], wall_seconds=outcome["wall_seconds"]))
            print(json.dumps(dict(completed=calls, total=16*len(charts))), flush=True)
    unchanged = identity == model_identity(MODEL)
    passed = unchanged and all(row["correct"] for row in results)
    write_new(output/"complete.json", dict(passed=passed, model_unchanged=unchanged, candidate_active=False,
        model_calls=calls, results=results, source_hashes=hashes,
        files={p.relative_to(output).as_posix(): file_digest(p) for p in output.rglob("*") if p.is_file()}))
    print(json.dumps(dict(passed=passed, correct={m: sum(r["correct"] for r in results if r["mode"] == m)
        for m in ("direct", "memory", "program")})), flush=True)
    if not passed:
        raise ValueError("authored interface acceptance failed; public benchmark run remains unqualified")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--font", type=Path, required=True)
    parser.add_argument("--prior", type=Path, help="Completed owned v2 run; enables the prospective v3 instruction check")
    args = parser.parse_args()
    run(args.output, args.font, args.prior)

"""Authored vision-interface canary, excluded from any benchmark or fitting.

Copyright (c) 2026 Prashant Jagtap. MIT License.
Uses one installed local model. Preserves every response before interpretation.
"""

import argparse
import json
import os
import time
from pathlib import Path

from chartqa_prepare import file_digest, outside, write_new
from chartqa_protocol import MODEL, bind_table, digest, encoded, local_chat, parse_response, request_body
from local_eval import local_api, model_identity

ROOT = Path(__file__).resolve().parents[1]
SOURCE_FILES = (
    "experiments/chartqa_canary.py", "experiments/chartqa_protocol.py", "experiments/chartqa_quant.py",
    "experiments/chartqa_prepare.py", "experiments/local_eval.py", "context_stamps/media_evidence.py",
    "context_stamps/context_state.py", "context_stamps/security.py",
)


def create_chart(path, font_path):
    from PIL import Image, ImageDraw, ImageFont

    image = Image.new("RGB", (720, 440), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype(str(font_path), 24)
    draw.text((70, 20), "Authored fixture: completed tasks", fill="black", font=font)
    draw.line([(80, 80), (80, 350), (650, 350)], fill="black", width=2)
    for x, label, value in ((170, "Alpha", 20), (360, "Beta", 40), (550, "Gamma", 60)):
        y = 350-value*4
        draw.rectangle((x-40, y, x+40, 350), fill="#427896")
        draw.text((x-16, y-32), str(value), fill="black", font=font)
        draw.text((x-42, 365), label, fill="black", font=font)
    image.save(path, format="PNG")


def capture(folder, name, body, mode, *, question=None, table=None):
    """Each call is attempted once. Network failures leave an explicit record."""
    write_new(folder/(name+".request.json"), body)
    started = time.perf_counter()
    write_new(folder/(name+".started.json"), dict(utc=time.time(), pid=os.getpid()))
    try:
        raw = local_chat(body)
    except Exception as exc:
        write_new(folder/(name+".transport-failure.json"), dict(error=f"{type(exc).__name__}: {exc}",
            wall_seconds=time.perf_counter()-started, retry_allowed=False))
        raise
    elapsed = time.perf_counter()-started
    with (folder/(name+".response.json")).open("xb") as stream:
        stream.write(raw)
    tick = time.perf_counter()
    error, result = None, None
    try:
        result = parse_response(raw, mode, question=question, table=table)
    except (ValueError, TypeError, KeyError, UnicodeError) as exc:
        error = f"{type(exc).__name__}: {exc}"
    outcome = dict(result=result, error=error, wall_seconds=elapsed,
                   processing_seconds=time.perf_counter()-tick, raw_sha256=digest(raw))
    write_new(folder/(name+".parsed.json"), outcome)
    return outcome


def run(output, font):
    output = outside(output)
    if local_api("/api/ps").get("models"):
        raise ValueError("resident local workload exists; do not evict it")
    identity = model_identity(MODEL)
    details = local_api("/api/show", dict(model=MODEL))
    if "vision" not in details.get("capabilities", []):
        raise ValueError("installed model must advertise vision")
    output.mkdir(exist_ok=False)
    (output/"source").mkdir()
    sources = {}
    for name in SOURCE_FILES:
        raw = (ROOT/name).read_bytes()
        target = output/"source"/name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(raw)
        sources[name] = digest(raw)
    create_chart(output/"authored-chart.png", font)
    image = (output/"authored-chart.png").read_bytes()
    question = "What is the sum of completed tasks for Alpha and Beta?"
    write_new(output/"registration.json", dict(model=identity, server=local_api("/api/version"),
        source_hashes=sources, font_sha256=file_digest(font), image_sha256=digest(image),
        question=question, authored_expected_answer="60", calls=4,
        purpose="Authored interface development only; excluded from model fitting and public task scores"))
    direct = capture(output, "direct", request_body("direct", question=question, image=image), "direct")
    extraction_body = request_body("extract", image=image)
    extraction = capture(output, "extract", extraction_body, "extract")
    if extraction["error"]:
        raise ValueError("extraction canary failed; keep the recorded attempt")
    table = extraction["result"]["table"]
    state, snapshot, memory, _ = bind_table(dict(image_sha256=digest(image), byte_length=len(image), width=720,
                                               height=440), image, table, identity["digest"], digest(encoded(extraction_body)))
    outcomes = dict(direct=direct, extract=extraction)
    for mode in ("memory", "program"):
        if not state.is_current(snapshot):
            raise ValueError("source binding changed before model call")
        outcomes[mode] = capture(output, mode, request_body(mode, question=question, memory=memory), mode,
                                 question=question, table=table)
    unchanged = identity == model_identity(MODEL)
    answer_results = {name: value["result"] and value["result"].get("answer") for name, value in outcomes.items() if name != "extract"}
    passed = unchanged and all(not v["error"] for v in outcomes.values()) and all(v == "60" for v in answer_results.values())
    write_new(output/"complete.json", dict(passed=passed, model_unchanged=unchanged, answers=answer_results,
        source_hashes=sources, candidate_active=False, model_calls=4,
        files={p.relative_to(output).as_posix(): file_digest(p) for p in output.rglob("*") if p.is_file()}))
    print(json.dumps(dict(canary_passed=passed, answers=answer_results, model_unchanged=unchanged)), flush=True)
    if not passed:
        raise ValueError("authored canary did not pass; no full study is authorized by this record")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--font", type=Path, required=True, help="Reviewed local TrueType font for the authored fixture")
    args = parser.parse_args()
    run(args.output, args.font)

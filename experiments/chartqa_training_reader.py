"""Collect paired local outcomes on the prepared public training split only.

Copyright (c) 2026 Prashant Jagtap. MIT License.
Diagnostic data collection, not deployment qualification or a held-out score.
The retained program canary failed semantically; this run does not reverse it.
"""

import argparse
import collections
import json
import os
import platform
import random
import re
import time
from pathlib import Path

from chartqa_prepare import file_digest, image_identity, outside, write_new
from chartqa_program_protocol import VERSION, request_body
from chartqa_protocol import MODEL, bind_table, digest, encoded
from chartqa_scalar_canary import capture
from local_eval import local_api, model_identity

ROOT = Path(__file__).resolve().parents[1]
SOURCES = (
    "experiments/chartqa_training_reader.py", "experiments/chartqa_training_policy.py",
    "experiments/chartqa_scalar_canary.py", "experiments/chartqa_scalar_protocol.py",
    "experiments/chartqa_program_protocol.py", "experiments/chartqa_protocol.py",
    "experiments/chartqa_quant.py", "experiments/chartqa_prepare.py", "experiments/chartqa_score.py",
    "experiments/chartqa_canary.py", "experiments/local_eval.py",
    "context_stamps/media_evidence.py", "context_stamps/context_state.py", "context_stamps/security.py",
    "context_stamps/local_policy.py", "context_stamps/local_learning.py", "context_stamps/experience.py",
)


def prepared_training(data):
    """Never open validation/test inputs or any answer file."""
    data = outside(data)
    manifest = json.loads((data/"manifest.json").read_bytes())
    path = data/"train.inputs.json"
    if file_digest(path) != manifest["files"][path.name]:
        raise ValueError("training input bytes changed")
    rows = json.loads(path.read_bytes())
    if len(rows) != 143 or len({r["id"] for r in rows}) != len(rows):
        raise ValueError("registered 143-question training cohort required")
    groups = collections.defaultdict(list)
    for row in rows:
        if row["split"] != "train" or not re.fullmatch(r"train-[ha]-[0-9]{5}", row["id"]):
            raise ValueError("training-only reader")
        path = (data/row["image_file"]).resolve()
        if data not in path.parents or file_digest(path) != manifest["files"][row["image_file"]]:
            raise ValueError("registered image path and bytes required")
        identity = image_identity(path.read_bytes())
        if any(identity[k] != row[k] for k in identity):
            raise ValueError("image metadata differs from decoded source")
        groups[row["image_sha256"]].append(row)
    if len(groups) != 64 or any(not 2 <= len(rows) <= 4 for rows in groups.values()):
        raise ValueError("complete registered repeated-chart cohort required")
    return groups


def run(data, output):
    data, output = outside(data), outside(output)
    groups = prepared_training(data)
    if local_api("/api/ps").get("models"):
        raise ValueError("resident local workload exists; wait without eviction")
    identity = model_identity(MODEL)
    if "vision" not in local_api("/api/show", dict(model=MODEL)).get("capabilities", []):
        raise ValueError("vision capability required")
    output.mkdir(exist_ok=False)
    sources = {}
    for name in SOURCES:
        raw = (ROOT/name).read_bytes()
        target = output/"source"/name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(raw)
        sources[name] = digest(raw)
    schedule = list(sorted(groups))
    random.Random(113).shuffle(schedule)
    write_new(output/"registration.json", dict(protocol=VERSION, model=identity, source_hashes=sources,
        server=local_api("/api/version"), python=platform.python_version(), pid=os.getpid(), utc=time.time(),
        data_manifest=file_digest(data/"manifest.json"), input_digest=file_digest(data/"train.inputs.json"),
        images=64, questions=143, maximum_calls=493, schedule=schedule,
        reader_order="extract once per chart, then shuffle all question/mode pairs with seed 113+chart index",
        modes=["direct", "memory", "program"], candidate_active=False,
        purpose="Training diagnostics and candidate fitting only; validation/test inputs and all keys excluded from reader",
        prior_canary="Authored v3 14/15 programs: valid arithmetic for absent category. Still failed; no gate overwritten.",
        gate="Successful extraction required for dependent arms; failures retained. No answer-quality acceptance or activation.",
        costs="Charge extraction and state preparation to each memory treatment; direct never pays extraction. No retries.",
        training_plan="Fit predefined CPU cell policies from independent train labels after collection. Chart-group cross-validation only; no held-out qualification."))
    records, calls = [], 0
    wall_start = time.perf_counter()
    for index, key in enumerate(schedule):
        rows = groups[key]
        folder = output/f"chart-{index:03d}"
        folder.mkdir()
        image = (data/rows[0]["image_file"]).read_bytes()
        body = request_body("extract", image=image)
        extraction = capture(folder, "extract", body, "extract")
        calls += 1
        state, snapshot, memory = None, None, None
        build_error, build_seconds = None, 0.0
        if not extraction["error"]:
            tick = time.perf_counter()
            try:
                state, snapshot, memory, _ = bind_table(rows[0], image, extraction["result"]["table"],
                    identity["digest"], digest(encoded(body)))
            except (ValueError, TypeError, KeyError) as exc:
                build_error = f"{type(exc).__name__}: {exc}"
            build_seconds = time.perf_counter()-tick
        write_new(folder/"memory.json", dict(image_sha256=key, memory=memory,
            build_seconds=build_seconds, error=build_error, tasks=rows))
        order = [(row, mode) for row in rows for mode in ("direct", "memory", "program")]
        random.Random(113+index).shuffle(order)
        for row, mode in order:
            name = row["id"]+"-"+mode
            if mode != "direct" and (memory is None or not state.is_current(snapshot)):
                outcome = dict(result=None, error="unavailable_extraction", wall_seconds=0.0,
                    processing_seconds=0.0, raw_sha256=None, no_model_call=True)
                write_new(folder/(name+".parsed.json"), outcome)
            else:
                body = request_body(mode, question=row["question"], image=image if mode == "direct" else None,
                    memory=memory if mode != "direct" else None)
                outcome = capture(folder, name, body, mode, question=row["question"],
                    table=memory["table"] if mode != "direct" else None)
                calls += 1
                if mode != "direct" and not state.is_current(snapshot):
                    raise ValueError("source changed during request; retain partial run without retry")
            records.append(dict(id=row["id"], mode=mode, image_sha256=key, folder=folder.name,
                answer=(outcome["result"] or {}).get("answer"), error=outcome["error"],
                called=not outcome.get("no_model_call", False)))
        write_new(folder/"complete.json", dict(image_sha256=key, questions=len(rows),
            files={p.name: file_digest(p) for p in folder.iterdir() if p.is_file()}))
        print(json.dumps(dict(charts_completed=index+1, charts_total=64, records=len(records), model_calls=calls)), flush=True)
    unchanged = identity == model_identity(MODEL)
    write_new(output/"complete.json", dict(model_unchanged=unchanged, source_hashes=sources,
        model_calls=calls, maximum_calls=493, records=records, candidate_active=False,
        run_wall_seconds=time.perf_counter()-wall_start,
        files={p.relative_to(output).as_posix(): file_digest(p) for p in output.rglob("*") if p.is_file()}))
    if not unchanged:
        raise ValueError("model identity changed during run")
    print(json.dumps(dict(complete=True, records=len(records), model_calls=calls)), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.data, args.output)

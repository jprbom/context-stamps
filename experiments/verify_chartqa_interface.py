"""Replay the retained authored visual-interface failure without a model call."""

import base64
import gzip
import hashlib
import json
from pathlib import Path

from chartqa_protocol import bind_table, digest, encoded, parse_response

ROOT = Path(__file__).resolve().parents[1]


def verify():
    directory = ROOT/"evidence"/"chartqa-interface-v1"
    manifest = json.loads((directory/"manifest.json").read_bytes())
    for name, expected in manifest["files"].items():
        path = (directory/name).resolve()
        if directory not in path.parents or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError("interface artifact changed")
    with gzip.open(directory/"records.json.gz", "rb") as stream:
        content = stream.read(1024*1024+1)
    if len(content) > 1024*1024:
        raise ValueError("interface archive limit")
    records = json.loads(content)
    def raw(name):
        record = records[name]
        data = base64.b64decode(record["base64"], validate=True)
        if digest(data) != record["sha256"]:
            raise ValueError("retained interface record differs")
        return data
    for name in records:
        raw(name)
    complete, registration = json.loads(raw("complete.json")), json.loads(raw("registration.json"))
    if complete["passed"] or not complete["model_unchanged"] or complete["candidate_active"]:
        raise ValueError("recorded failed inactive canary required")
    for name, fingerprint in complete["files"].items():
        if digest(raw(name)) != fingerprint:
            raise ValueError("native completion fingerprint differs")
    table = parse_response(raw("extract.response.json"), "extract")["table"]
    answers = {}
    for mode in ("direct", "memory", "program"):
        result = parse_response(raw(mode+".response.json"), mode, question=registration["question"], table=table)
        if result != json.loads(raw(mode+".parsed.json"))["result"]:
            raise ValueError("response interpretation changed")
        answers[mode] = result["answer"]
    if answers != complete["answers"] or answers["program"] != "60" or any(answers[m] == "60" for m in ("direct", "memory")):
        raise ValueError("retained interface failure pattern changed")
    image = raw("authored-chart.png")
    state, snapshot, memory, bundle = bind_table(dict(image_sha256=digest(image), byte_length=len(image), width=720, height=440),
        image, table, registration["model"]["digest"], digest(encoded(json.loads(raw("extract.request.json")))))
    if not state.is_current(snapshot) or memory["kind"] != "MODEL_OUTPUT":
        raise ValueError("extraction lineage changed")
    state.set_roles(bundle.source.key, ())
    if state.is_current(snapshot):
        raise ValueError("source revocation did not invalidate memory")
    with gzip.open(directory/"probe-records.json.gz", "rb") as stream:
        content = stream.read(1024*1024+1)
    if len(content) > 1024*1024:
        raise ValueError("probe archive limit")
    probe = json.loads(content)
    def probe_raw(name):
        value = base64.b64decode(probe[name]["base64"], validate=True)
        if digest(value) != probe[name]["sha256"]:
            raise ValueError("probe record changed")
        return value
    for name in probe:
        probe_raw(name)
    finished = json.loads(probe_raw("complete.json"))
    registered = json.loads(probe_raw("registration.json"))
    if not finished["model_unchanged"] or finished["candidate_active"] or finished["model_calls"] != 24:
        raise ValueError("complete inactive 24-call probe required")
    for name, fingerprint in finished["files"].items():
        if digest(probe_raw(name)) != fingerprint:
            raise ValueError("probe completion identity changed")
    expected = {case[0]: case[2] for case in registered["cases"]}
    required = {(case, mode, variant) for case in expected for mode in ("direct", "memory")
                for variant in ("baseline", "grounded_schema", "json_only")}
    if len(finished["results"]) != 24 or {(r["case"], r["mode"], r["variant"]) for r in finished["results"]} != required:
        raise ValueError("probe control coverage differs")
    counts = {variant: {mode: 0 for mode in ("direct", "memory")} for variant in registered["variants"]}
    for row in finished["results"]:
        stem = row["case"]+"-"+row["mode"]+"-"+row["variant"]
        answer, error = None, None
        try:
            answer = parse_response(probe_raw(stem+".response.json"), row["mode"])["answer"]
        except (ValueError, TypeError, KeyError, UnicodeError) as exc:
            error = f"{type(exc).__name__}: {exc}"
        if answer != row["answer"] or error != row["error"] or row["correct"] != (not error and answer == expected[row["case"]]):
            raise ValueError("probe response interpretation changed")
        counts[row["variant"]][row["mode"]] += row["correct"]
    print(json.dumps(dict(replayed_actual_requests=4, correct_direct=0, correct_memory=0, correct_program=1,
        probe_requests=24, probe_correct=counts, model_calls_in_replay=0,
        scope="Authored interface cases; not ChartQA accuracy or a trained improvement")))


if __name__ == "__main__":
    verify()

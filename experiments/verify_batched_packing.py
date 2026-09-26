"""Replay CPU timings and bind compiled prompts to the frozen reader study.

Recorded parity/timing assertions are reproducible research logs, not independent
attestation. Rerunning the full comparison requires the pinned local source index.
"""

import gzip
import io
import json
import math
from pathlib import Path

from profile_batched_packing import ARMS, digest, summarize
from ruler_native import sha
from source_evidence import checked_path, verify_sources

ROOT = Path(__file__).resolve().parents[1]


def study_contexts():
    path = ROOT/"evidence/lme-relations-v2/records.json.gz"
    with gzip.GzipFile(fileobj=io.BytesIO(path.read_bytes())) as stream:
        raw = stream.read(16*1024*1024+1)
    if len(raw) > 16*1024*1024:
        raise ValueError("study expansion limit")
    contexts = {}
    for row in json.loads(raw)["scores"]:
        value = dict(prompt_sha256=row["prompt_sha256"], input_tokens=row["input_tokens"],
                     selected_sources_sha256=digest(row["sources"]), selected=len(row["sources"]))
        key = (row["id"], row["arm"])
        if key in contexts and contexts[key] != value:
            raise ValueError("reader study contexts disagree")
        contexts[key] = value
    return contexts


def validate(rows, plan, summary, contexts):
    ids = plan["ids"]
    if len(ids) != 72 or len(set(ids)) != 72 or plan["repeats"] != 2 or plan["workers"] != 8 or plan["batch_size"] != 8:
        raise ValueError("registered CPU comparison changed")
    if plan["labels_used"] or plan["model_calls"] != 0 or plan["pairs"] != 432 or plan["arms"] != list(ARMS):
        raise ValueError("unsupported training or model-call claim")
    expected = {(i, a, r) for i in ids for a in ARMS for r in range(2)}
    if len(rows) != 432 or {(r["id"], r["arm"], r["repeat"]) for r in rows} != expected:
        raise ValueError("complete paired repeats required")
    repeated = {}
    for row in rows:
        if row["equal"] is not True or sorted(row["order"]) != ["baseline", "batched"]:
            raise ValueError("context mismatch or invalid paired order")
        context = contexts[(row["id"], row["arm"])]
        if any(row[k] != value for k, value in context.items()):
            raise ValueError("compiled context differs from the frozen reader study")
        key = (row["id"], row["arm"])
        ranked = row["ranked_views_sha256"]
        if len(ranked) != 64 or any(c not in "0123456789abcdef" for c in ranked):
            raise ValueError("ranked-view digest required")
        if key in repeated and repeated[key] != ranked:
            raise ValueError("ranked views differ between repeats")
        repeated[key] = ranked
        for method in ("baseline", "batched"):
            values = row[method]
            if any(type(v) not in (int, float) or not math.isfinite(v) or v < 0 for v in values.values()):
                raise ValueError("invalid measured CPU/wall time")
            if values["wall_seconds"]+1e-6 < values["search_seconds"]+values["pack_seconds"]:
                raise ValueError("whole compilation cannot omit a sequential stage")
    if summary != summarize(rows):
        raise ValueError("CPU comparison aggregates changed")
    return dict(comparison_pairs=432, timed_compilations=864, contexts=216, repeats=2, all_contexts_equal=True,
                new_model_calls=0, new_quality_claim=False)


def verify(path):
    manifest = json.loads((path/"manifest.json").read_bytes())
    for name, expected in manifest["files"].items():
        if sha(checked_path(name, path)) != expected:
            raise ValueError("CPU evidence hash mismatch")
    if sha(checked_path(manifest["archive"])) != manifest["archive_sha256"]:
        raise ValueError("exploratory source archive changed")
    plan = json.loads((path/"registration.json").read_bytes())
    if sha(ROOT/"evidence/lme-relations-v2/registration.json") != plan["study_registration_sha256"]:
        raise ValueError("reader study registration changed")
    verify_sources(plan["sources"])
    verify_sources(manifest["tools"])
    for i in (1, 2, 3):
        verify_sources(json.loads((path/f"probes/attempt-{i}.json").read_bytes())["source_hashes"])
    raw = (path/"records.jsonl").read_bytes()
    if len(raw) > 4*1024*1024 or sha(path/"records.jsonl") != json.loads((path/"completion.json").read_bytes())["records_sha256"]:
        raise ValueError("CPU records changed or exceed bounds")
    canaries = json.loads((path/"tokenizer-canaries.json").read_bytes())
    if len(canaries) != 170 or not all(r["equal"] is True for r in canaries):
        raise ValueError("native tokenizer parity is incomplete")
    return validate([json.loads(s) for s in raw.splitlines()], plan,
                    json.loads((path/"summary.json").read_bytes()), study_contexts())


if __name__ == "__main__":
    print(json.dumps(verify(ROOT/"evidence/batched-packing-v1")))

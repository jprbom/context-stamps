"""Offline native-score, paired-comparison and request-timing replay."""

import gzip
import io
import json
import math
from pathlib import Path

from lme_memory import native_scorer, score, summary
from lme_relations import ARMS, MODELS, OPTIONS
from review_lme_relations import diagnostics
from ruler_native import sha
from source_evidence import checked_path, verify_sources

ROOT = Path(__file__).resolve().parents[1]


def validate(payload, registration, summaries, review, namespace):
    ids = registration["ids"]
    keys = {k["id"]: k for k in payload["references"]}
    if len(ids) != 72 or len(set(ids)) != 72 or len(keys) != len(payload["references"]) or set(keys) != set(ids):
        raise ValueError("unique complete registered references required")
    rows = payload["scores"]
    if len(rows) != 432 or {(r["model"], r["id"], r["arm"]) for r in rows} != {(m, i, a) for m in MODELS for i in ids for a in ARMS}:
        raise ValueError("complete paired experiment required")
    if registration["options"] != OPTIONS or any(registration["models"][m]["digest"] != MODELS[m] for m in MODELS):
        raise ValueError("registered model configuration changed")
    for r in rows:
        if r["correct"] != (not r["error"] and score(namespace, r["prediction"], keys[r["id"]])):
            raise ValueError("native credit mismatch")
        if any(k in r for k in ("prompt", "prompts", "context", "question")):
            raise ValueError("raw prompts excluded from public output")
        if not 0 <= r["input_tokens"] <= 6144 or not 0 <= r["output_tokens"] <= 256:
            raise ValueError("token ceiling violated")
        for name in ("request_wall_seconds", "generation_wall_seconds", "retrieval_seconds", "pack_seconds"):
            if type(r[name]) not in (int, float) or not math.isfinite(r[name]) or r[name] < 0:
                raise ValueError("invalid observed timing")
        if r["request_wall_seconds"]+1e-6 < r["generation_wall_seconds"]+r["retrieval_seconds"]+r["pack_seconds"]:
            raise ValueError("whole request cannot be shorter than its measured sequential stages")
        if not r["scope_checked"] or r["sufficient_context_certified"]:
            raise ValueError("incorrect scope/sufficiency claim")
    expected = {m: {a: summary([r for r in rows if r["model"] == m and r["arm"] == a]) for a in ARMS} for m in MODELS}
    if summaries != expected or review["models"] != diagnostics(rows, namespace, keys) or review["activation_qualified"]:
        raise ValueError("aggregate results or inactive status changed")
    return dict(native_scores_replayed=432, readers=2, questions=72, actual_request_timings_replayed=True, activation_qualified=False)


def verify(path):
    manifest = json.loads((path/"manifest.json").read_bytes())
    for name, digest in manifest["files"].items():
        if sha(checked_path(name, path)) != digest:
            raise ValueError("evidence hash mismatch")
    registration = json.loads((path/"registration.json").read_bytes())
    verify_sources(registration["sources"])
    verify_sources(manifest["tool_sources"])
    with gzip.GzipFile(fileobj=io.BytesIO((path/"records.json.gz").read_bytes())) as stream:
        raw = stream.read(16*1024*1024+1)
    if len(raw) != manifest["expanded_bytes"] or len(raw) > 16*1024*1024:
        raise ValueError("bounded evidence expansion failed")
    return validate(json.loads(raw), registration, json.loads((path/"summary.json").read_bytes()),
                    json.loads((path/"failure-review.json").read_bytes()),
                    native_scorer(ROOT/"evidence/lme-memory-v1/upstream/qa_eval_metrics.py"))


if __name__ == "__main__":
    print(json.dumps(verify(ROOT/"evidence/lme-relations-v2")))

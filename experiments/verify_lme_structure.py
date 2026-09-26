"""Bounded offline replay of the two-reader structural-memory experiment."""

import gzip
import io
import json
from pathlib import Path

from lme_memory import native_scorer, score, summary
from lme_structure import ARMS, MODELS, OPTIONS
from ruler_native import sha
from source_evidence import checked_path, verify_sources

ROOT = Path(__file__).resolve().parents[1]


def validate(payload, registration, summaries, namespace):
    ids = registration["ids"]
    keys = {k["id"]: k for k in payload["references"]}
    if len(ids) != 72 or len(set(ids)) != 72 or len(payload["references"]) != 72 or set(keys) != set(ids):
        raise ValueError("complete unique registered references required")
    rows = payload["scores"]
    if len(rows) != 288 or {(r["model"], r["id"], r["arm"]) for r in rows} != {(m, i, a) for m in MODELS for i in ids for a in ARMS}:
        raise ValueError("complete paired model/arm population required")
    if registration["options"] != OPTIONS or any(registration["models"][m]["digest"] != MODELS[m] for m in MODELS):
        raise ValueError("registered reader or configuration changed")
    for row in rows:
        correct = not row["error"] and score(namespace, row["prediction"], keys[row["id"]])
        if row["correct"] != correct:
            raise ValueError("native credit mismatch")
        if any(k in row for k in ("prompt", "prompts", "context", "question")):
            raise ValueError("raw text excluded from public evidence")
        if not 0 <= row["input_tokens"] <= 6144 or not 0 <= row["output_tokens"] <= 256:
            raise ValueError("token cap violated")
        if not row["scope_checked"] or row["sufficient_context_certified"]:
            raise ValueError("scope/sufficiency claim changed")
    expected = {}
    for model in MODELS:
        arms = {a: {r["id"]: r for r in rows if r["model"] == model and r["arm"] == a} for a in ARMS}
        result = {a: summary(list(arms[a].values())) for a in ARMS}
        result["gains"] = [i for i in keys if arms["structure"][i]["correct"] and not arms["raw"][i]["correct"]]
        result["regressions"] = [i for i in keys if arms["raw"][i]["correct"] and not arms["structure"][i]["correct"]]
        expected[model] = result
    if summaries != expected:
        raise ValueError("aggregate metrics or discordant questions changed")
    return dict(native_scores_replayed=len(rows), questions=72, readers=2, activation_qualified=False)


def verify(directory):
    manifest = json.loads((directory/"manifest.json").read_bytes())
    for name, digest in manifest["files"].items():
        if sha(checked_path(name, directory)) != digest:
            raise ValueError("evidence hash mismatch")
    registration = json.loads((directory/"registration.json").read_bytes())
    verify_sources(registration["sources"])
    verify_sources(manifest["tool_sources"])
    with gzip.GzipFile(fileobj=io.BytesIO((directory/"records.json.gz").read_bytes())) as stream:
        raw = stream.read(8*1024*1024+1)
    if len(raw) != manifest["expanded_bytes"] or len(raw) > 8*1024*1024:
        raise ValueError("bounded evidence expansion failed")
    return validate(json.loads(raw), registration, json.loads((directory/"summary.json").read_bytes()),
                    native_scorer(ROOT/"evidence/lme-memory-v1/upstream/qa_eval_metrics.py"))


if __name__ == "__main__":
    print(json.dumps(verify(ROOT/"evidence/lme-structure-v1")))

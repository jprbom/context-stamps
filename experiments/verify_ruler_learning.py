"""Recompute local policy, all retained scores and protected rejection offline."""

import gzip
import hashlib
import io
import json
import math
from pathlib import Path

from ruler_local_learning import ARMS, grade, summarize
from source_evidence import verify_sources

from context_stamps.local_learning import (
    EvaluationCohort,
    LearningLimits,
    PairedOutcome,
    assess_cohorts,
    revision,
)
from context_stamps.local_policy import PolicyObservation, fit_cell_policy

ROOT = Path(__file__).resolve().parents[1]


def validate(records, registration, plan, policy, promotion, summary):
    paired_records = {}
    train_ids, hold_ids = {}, {}
    for phase, count, arms, ids, source_name in (
        ("train", 312, ("reader", "direct"), train_ids, "training"),
        ("holdout", 208, ARMS, hold_ids, "holdout"),
    ):
        sources = records["input-identities"][source_name]
        ids.update({r["id"]: r["input_sha256"] for r in sources})
        keys = {r["id"]: r for r in records["references"][source_name]}
        rows = records[phase]
        if len(sources) != count or len(ids) != count or len(keys) != count or set(keys) != set(ids):
            raise ValueError("All unique inputs and references required")
        if len(rows) != count*len(arms) or {(r["id"], r["arm"]) for r in rows} != {(key, arm) for key in ids for arm in arms}:
            raise ValueError("Missing or duplicate treatment")
        rescored = grade(rows, keys)
        if rescored != rows:
            raise ValueError("Native or strict scores do not match retained predictions")
        lookup = {r["id"]: r for r in sources}
        for row in rows:
            if (row["length"], row["task"]) != (lookup[row["id"]]["length"], lookup[row["id"]]["task"]):
                raise ValueError("Task identity mismatch")
            if any(field in row for field in ("question", "prompt", "prompts", "context")):
                raise ValueError("Raw context excluded from publication")
            if any(type(row[k]) is not int or row[k] < 0 for k in ("input_tokens", "output_tokens", "model_calls")):
                raise ValueError("Invalid resource counts")
            if not math.isfinite(row["wall_seconds"]) or row["wall_seconds"] < 0:
                raise ValueError("Invalid latency")
            if row["verified"] and (row["action"] != "direct" or row["model_calls"] != 0
                                    or row["input_tokens"] != 0 or row["output_tokens"] != 0):
                raise ValueError("Verified exact path must not claim model inference")
        expected = {arm: summarize([r for r in rows if r["arm"] == arm]) for arm in arms}
        if summary[source_name] != expected:
            raise ValueError("Aggregate metric mismatch")
        if phase == "holdout":
            paired_records = {(r["id"], r["arm"]): r for r in rows}
    if len(set(train_ids.values())) != 312 or len(set(hold_ids.values())) != 208 or set(train_ids.values()) & set(hold_ids.values()):
        raise ValueError("Exact input overlap")
    observations = tuple(PolicyObservation(train_ids[r["id"]], r["cell"], r["arm"], r["strict_correct"],
                         float(r["input_tokens"]+r["output_tokens"])) for r in records["train"])
    fitting = {k: v for k, v in registration["fitting"].items() if k != "cells"}
    fitted = fit_cell_policy(observations, binding=registration["binding"], baseline_action="reader", **fitting)
    if fitted.revision != policy["revision"] or revision(policy["policy"]) != fitted.revision:
        raise ValueError("Fitted policy mismatch")
    if revision([r.__dict__ for r in observations]) != revision(policy["observations"]):
        raise ValueError("Training observations mismatch")
    if json.loads(json.dumps(fitted.choices)) != summary["policy_choices"] or summary["policy_revision"] != fitted.revision:
        raise ValueError("Policy summary mismatch")
    paired = []
    for source in records["input-identities"]["holdout"]:
        old, new = (paired_records[source["id"], arm] for arm in ("reader", "learned"))
        paired.append(PairedOutcome(hold_ids[source["id"]], revision([old, new]), old["strict_correct"], new["strict_correct"],
            float(old["input_tokens"]+old["output_tokens"]), float(new["input_tokens"]+new["output_tokens"]),
            new["wall_seconds"]*1000., None))
    if revision([r.__dict__ for r in paired]) != revision(promotion["observations"]):
        raise ValueError("Paired promotion observations mismatch")
    cohorts = tuple(EvaluationCohort(c["name"], tuple(c["clusters"])) for c in plan["cohorts"])
    result = assess_cohorts(tuple(paired), LearningLimits(**plan["limits"]), cohorts, alpha=plan["round_alpha"])
    result["plan_revision"] = revision(plan)
    if result != promotion["report"] or result["eligible"] or summary["promoted"]:
        raise ValueError("Protected non-activation result mismatch")
    if promotion["active_revision"] != promotion["baseline_revision"] or plan["candidate"] != fitted.revision:
        raise ValueError("Incorrect activation/candidate identity")
    update = sum(r["input_tokens"]+r["output_tokens"] for r in records["train"])
    if update != summary["update_tokens"] or plan["limits"]["update_cost"] != update:
        raise ValueError("Training cost omitted")
    expected_cells = []
    for length, task in sorted({(r["length"], r["task"]) for r in records["holdout"]}):
        expected_cells.append(dict(length=length, task=task, arms={arm: summarize([r for r in records["holdout"] if
            (r["length"], r["task"], r["arm"]) == (length, task, arm)]) for arm in ARMS}))
    if expected_cells != summary["cells"]:
        raise ValueError("Task/length metrics mismatch")
    return dict(training_inputs=312, holdout_inputs=208, policy_reproduced=True,
                all_metrics_reproduced=True, candidate_inactive=True)


def verify(path):
    def read(name):
        return json.loads((path/name).read_bytes())
    manifest = read("manifest.json")
    for name, expected in manifest["files"].items():
        if Path(name).name != name or hashlib.sha256((path/name).read_bytes()).hexdigest() != expected:
            raise ValueError("Evidence identity mismatch")
    registration = read("registration.json")
    verify_sources(registration["sources"])
    with gzip.GzipFile(fileobj=io.BytesIO((path/"records.json.gz").read_bytes())) as stream:
        raw = stream.read(8*1024*1024+1)
    if len(raw) > 8*1024*1024 or len(raw) != manifest["expanded_bytes"]:
        raise ValueError("Bounded evidence expansion mismatch")
    records = json.loads(raw)
    for key in ("training-data", "holdout-data"):
        verify_sources({"experiments/ruler_native.py": records[key]["plan"]["runner_sha256"]})
    return validate(records, registration, read("registered-trial.json"), read("policy.json"),
                    read("promotion.json"), read("summary.json"))


if __name__ == "__main__":
    print(json.dumps(verify(ROOT/"evidence/ruler-local-learning-v1")))

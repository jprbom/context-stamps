"""Replay native memory scores and the fitted local policy without a model call."""

import gzip
import hashlib
import io
import json
from pathlib import Path

from lme_memory import ARMS, grade, native_scorer, summary
from source_evidence import verify_sources

from context_stamps.linear_policy import UtilityPair, fit_linear_policy

ROOT = Path(__file__).resolve().parents[1]


def validate(payload, registration, policy_record, choices, summaries, namespace):
    keys = {r["id"]: r for r in payload["references"]}
    if len(keys) != 294:
        raise ValueError("Complete deterministic text population required")
    train_ids, hold_ids = set(registration["train_ids"]), set(registration["holdout_ids"])
    if len(train_ids) != 72 or len(hold_ids) != 222 or train_ids & hold_ids or train_ids | hold_ids != set(keys):
        raise ValueError("Registered disjoint question split changed")
    rows = payload["scores"]
    for phase, ids, arms in (("train", train_ids, ARMS), ("holdout", hold_ids, ARMS+("learned",))):
        subset = [r for r in rows if r["phase"] == phase]
        if (len(subset) != len(ids)*len(arms)
                or {(r["id"], r["arm"]) for r in subset} != {(i, a) for i in ids for a in arms}):
            raise ValueError("Missing or duplicate measured/policy treatment")
        if subset != grade(subset, keys, namespace):
            raise ValueError("Native scores changed")
        if summaries[phase] != {a: summary([r for r in subset if r["arm"] == a]) for a in arms}:
            raise ValueError("Aggregate results changed")
        for r in subset:
            if any(field in r for field in ("prompt", "prompts", "context", "question")):
                raise ValueError("Raw histories/prompts excluded from public export")
            if not 0 <= r["input_tokens"] <= 6144 or not 0 <= r.get("output_tokens", 0) <= 256:
                raise ValueError("Registered token ceiling violated")
            if r["sufficient_context_certified"] or not r["scope_checked"]:
                raise ValueError("Incorrect context qualification claim")
    features = {r["id"]: tuple(r["features"]) for r in payload["features"]}
    if set(features) != set(keys) or len(payload["features"]) != len(keys):
        raise ValueError("Complete unique input features required")
    lookup = {(r["id"], r["arm"]): r for r in rows if r["arm"] != "learned"}
    observations = []
    for saved in policy_record["training_observations"]:
        i = saved["cluster_id"]
        if i not in train_ids:
            raise ValueError("Holdout contamination of fitting")
        a, b = lookup[i, "state"], lookup[i, "linked"]
        observations.append(UtilityPair(i, features[i], a["correct"], b["correct"],
                                       a["input_tokens"]+a.get("output_tokens", 0), b["input_tokens"]+b.get("output_tokens", 0)))
    if len(observations) != len(train_ids) or json.loads(json.dumps([r.__dict__ for r in observations])) != policy_record["training_observations"]:
        raise ValueError("Paired training observations changed")
    saved_policy = policy_record["policy"]
    fitted = fit_linear_policy(tuple(observations), binding=saved_policy["binding"], baseline_action="state", candidate_action="linked",
                               cost_cap=registration["cost_cap"], failure_penalty=registration["failure_penalty"], ridge=registration["ridge"])
    # BLAS implementations may differ at final rounding; choices must match exactly.
    if (len(saved_policy["coefficients"]) != len(fitted.coefficients)
            or saved_policy["baseline_action"] != fitted.baseline_action
            or saved_policy["candidate_action"] != fitted.candidate_action
            or any(abs(a-b) > 1e-10 for a, b in zip(fitted.coefficients+(fitted.intercept,), tuple(saved_policy["coefficients"])+(saved_policy["intercept"],)))):
        raise ValueError("Refitted policy differs")
    expected = {i: fitted.choose(features[i], binding=fitted.binding, allowed_actions=("state", "linked")) for i in hold_ids}
    if choices != expected or policy_record["active"]:
        raise ValueError("Holdout choices or inactive state changed")
    for row in [r for r in rows if r["arm"] == "learned"]:
        selected = lookup[row["id"], choices[row["id"]]]
        if row != selected | dict(arm="learned", selected_arm=selected["arm"]):
            raise ValueError("Policy-selected outcome differs from measured arm")
    return dict(native_scores_replayed=len(rows), fitted_policy_replayed=True,
                training_questions=72, heldout_questions=222, active=False)


def verify(path):
    def read(name):
        return json.loads((path/name).read_bytes())
    manifest = read("manifest.json")
    for name, expected in manifest["files"].items():
        p = Path(name)
        if p.is_absolute() or p.drive or ".." in p.parts or hashlib.sha256((path/p).read_bytes()).hexdigest() != expected:
            raise ValueError("Evidence hash mismatch")
    registration = read("registration.json")
    verify_sources(registration["sources"])
    with gzip.GzipFile(fileobj=io.BytesIO((path/"records.json.gz").read_bytes())) as stream:
        raw = stream.read(16*1024*1024+1)
    if len(raw) != manifest["expanded_bytes"] or len(raw) > 16*1024*1024:
        raise ValueError("Unexpected evidence expansion")
    return validate(json.loads(raw), registration, read("policy.json"), read("holdout-choices.json"),
                    read("summary.json"), native_scorer(path/"upstream/qa_eval_metrics.py"))


if __name__ == "__main__":
    print(json.dumps(verify(ROOT/"evidence/lme-memory-v1")))

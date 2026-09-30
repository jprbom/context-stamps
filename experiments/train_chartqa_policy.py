"""Fit CPU policies after complete paired training collection; holdouts stay closed.

Copyright (c) 2026 Prashant Jagtap. MIT License.
Four-fold chart-group diagnostics are for candidate selection, not qualification.
"""

import argparse
import collections
import json
import random
import time
from dataclasses import asdict
from pathlib import Path

from chartqa_prepare import file_digest, outside, write_new
from chartqa_scalar_protocol import parse_response
from chartqa_score import score_answer
from chartqa_training_policy import ACTIONS, PENALTIES, environment, fit, input_cell


def measured_call(folder, stem):
    path = folder/(stem+".response.json")
    parsed = json.loads((folder/(stem+".parsed.json")).read_bytes())
    if parsed.get("no_model_call"):
        if path.exists() or parsed["error"] != "unavailable_extraction":
            raise ValueError("explicit unavailable extraction required")
        return dict(tokens=0, wall_seconds=0.0, processing_seconds=0.0, calls=0), parsed
    raw = json.loads(path.read_bytes())
    for name in ("prompt_eval_count", "eval_count"):
        if type(raw.get(name)) is not int or raw[name] < 0:
            raise ValueError("measured nonnegative token counts required")
    return dict(tokens=raw["prompt_eval_count"]+raw["eval_count"], wall_seconds=parsed["wall_seconds"],
        processing_seconds=parsed["processing_seconds"], calls=1), parsed


def plan_metrics(rows, shared, choices):
    ids = {r["id"] for r in rows}
    if set(choices) != ids:
        raise ValueError("exactly one action per question required")
    selected = [r for r in rows if choices[r["id"]] == r["mode"]]
    if len(selected) != len(ids):
        raise ValueError("exactly one action per question required")
    groups = collections.defaultdict(list)
    for row in selected:
        groups[row["image_sha256"]].append(row)
    charged = {r["image_sha256"] for r in selected if r["mode"] != "direct"}
    # One extraction per used chart, including failed extraction. Mixed routes
    # cannot pay only a per-question fraction of this unavoidable setup cost.
    costs = {key: sum(r["query_cost"][key] for r in selected)+sum(shared[g][key] for g in charged)
             for key in ("tokens", "wall_seconds", "processing_seconds", "calls")}
    return dict(questions=len(selected), correct=sum(r["correct"] for r in selected),
        stripped_exact=sum(r["stripped_exact"] for r in selected),
        errors=sum(r["error"] is not None for r in selected), abstentions=sum(r["answer"] is None for r in selected),
        charts=len(groups), complete_charts=sum(all(r["correct"] for r in records) for records in groups.values()),
        extractions_charged=len(charged), costs=costs,
        subgroup={str(origin): dict(questions=sum(r["origin"] == origin for r in selected),
            correct=sum(r["correct"] and r["origin"] == origin for r in selected)) for origin in (0, 1)})


def choose(policy, rows, binding):
    return {r["id"]: policy.choose(input_cell(r["question"]), binding=binding, allowed_actions=ACTIONS) for r in rows}


def load_records(data, runs):
    manifest = json.loads((data/"manifest.json").read_bytes())
    registration = json.loads((runs/"registration.json").read_bytes())
    complete = json.loads((runs/"complete.json").read_bytes())
    if (registration["data_manifest"] != file_digest(data/"manifest.json") or not complete["model_unchanged"]
            or len(complete["records"]) != 429 or complete["source_hashes"] != registration["source_hashes"]):
        raise ValueError("complete fixed-reader training run required")
    for name, fingerprint in complete["files"].items():
        path = (runs/name).resolve()
        if runs not in path.parents or file_digest(path) != fingerprint:
            raise ValueError("training run fingerprint changed")
    for name in ("train.inputs.json", "train.keys.json"):
        if file_digest(data/name) != manifest["files"][name]:
            raise ValueError("training data changed")
    inputs = {r["id"]: r for r in json.loads((data/"train.inputs.json").read_bytes())}
    # No validation or test key is opened by this script.
    keys = {r["id"]: r["labels"] for r in json.loads((data/"train.keys.json").read_bytes())}
    pairs = {(r["id"], r["mode"]) for r in complete["records"]}
    if pairs != {(key, action) for key in inputs for action in ACTIONS} or set(keys) != set(inputs):
        raise ValueError("complete matched actions and references required")
    shared, rows = {}, []
    for record in complete["records"]:
        source = inputs[record["id"]]
        folder = runs/record["folder"]
        memory = json.loads((folder/"memory.json").read_bytes())
        group = record["image_sha256"]
        if group != source["image_sha256"] or memory["image_sha256"] != group:
            raise ValueError("image group binding changed")
        if group not in shared:
            cost, extraction = measured_call(folder, "extract")
            cost["processing_seconds"] += memory["build_seconds"]
            shared[group] = cost
        query_cost, parsed = measured_call(folder, record["id"]+"-"+record["mode"])
        if not parsed.get("no_model_call"):
            result, error = None, None
            try:
                result = parse_response((folder/(record["id"]+"-"+record["mode"]+".response.json")).read_bytes(),
                    record["mode"], question=source["question"], table=(memory["memory"] or {}).get("table"))
            except (ValueError, TypeError, KeyError, UnicodeError) as exc:
                error = f"{type(exc).__name__}: {exc}"
            if result != parsed["result"] or error != parsed["error"]:
                raise ValueError("saved model interpretation differs on replay")
        answer = (parsed["result"] or {}).get("answer")
        if answer != record["answer"] or parsed["error"] != record["error"]:
            raise ValueError("completion projection differs from recorded response")
        score = score_answer(keys[record["id"]], answer)
        rows.append(dict(record, question=source["question"], origin=source["origin"],
            correct=score["relaxed"] and not parsed["error"], stripped_exact=score["stripped_exact"] and not parsed["error"],
            tokens=query_cost["tokens"]+(shared[group]["tokens"] if record["mode"] != "direct" else 0),
            query_cost=query_cost))
    calls = sum(r["query_cost"]["calls"] for r in rows)+sum(c["calls"] for c in shared.values())
    if calls != complete["model_calls"]:
        raise ValueError("model request denominator differs")
    return rows, shared, keys


def train(data, runs, output):
    data, runs, output = map(outside, (data, runs, output))
    complete = json.loads((runs/"complete.json").read_bytes())
    if len(complete["records"]) != 429 or not complete["model_unchanged"]:
        raise ValueError("do not fit partial or changed-reader runs")
    registered = json.loads((runs/"registration.json").read_bytes())
    root = Path(__file__).resolve().parents[1]
    for name in ("experiments/chartqa_training_policy.py", "context_stamps/local_policy.py"):
        if file_digest(root/name) != registered["source_hashes"][name]:
            raise ValueError("predeclared policy source changed")
    output.mkdir(exist_ok=False)
    write_new(output/"registration.json", dict(reader=file_digest(runs/"registration.json"),
        completion=file_digest(runs/"complete.json"), source=file_digest(Path(__file__)),
        folds=4, split_seed=127, penalties=PENALTIES, reader_weights_changed=False,
        cost_fit="Conservative cold-query cost: full extraction per dependent training observation",
        cost_evaluation="Actual selected query costs plus extraction/build once per chart with any dependent route",
        selection="Chart-group CV candidate must preserve question and complete-chart credit versus fold-selected fixed control, and reduce tokens without longer request time",
        limits="Training-only exploratory selection. No independent validation, retention or deployment qualification. No activation."))
    rows, shared, keys = load_records(data, runs)
    write_new(output/"paired-training.json", dict(rows=rows, shared=shared, keys=keys))
    binding = environment(file_digest(runs/"registration.json"))
    groups = sorted(shared)
    random.Random(127).shuffle(groups)
    fold_ids = {group: i % 4 for i, group in enumerate(groups)}
    folds, cv_choices = [], {str(p): {} for p in PENALTIES}
    cv_fixed = {}
    started = time.perf_counter()
    for fold in range(4):
        training = [r for r in rows if fold_ids[r["image_sha256"]] != fold]
        evaluation = [r for r in rows if fold_ids[r["image_sha256"]] == fold]
        fixed = {action: plan_metrics(training, shared, {r["id"]: action for r in training}) for action in ACTIONS}
        best = min(ACTIONS, key=lambda a: (-fixed[a]["correct"], -fixed[a]["complete_charts"], fixed[a]["costs"]["tokens"], a))
        cv_fixed.update({r["id"]: best for r in evaluation})
        fitted = {}
        for penalty in PENALTIES:
            policy = fit(training, binding=binding, penalty=penalty)
            cv_choices[str(penalty)].update(choose(policy, evaluation, binding))
            fitted[str(penalty)] = asdict(policy)
        folds.append(dict(fold=fold, training_charts=len({r["image_sha256"] for r in training}),
            evaluation_charts=len({r["image_sha256"] for r in evaluation}), fixed_selected=best, policies=fitted))
    fitted = {str(p): fit(rows, binding=binding, penalty=p) for p in PENALTIES}
    fit_seconds = time.perf_counter()-started
    controls = {action: plan_metrics(rows, shared, {r["id"]: action for r in rows}) for action in ACTIONS}
    controls["fixed_rule"] = plan_metrics(rows, shared, {r["id"]: "program" if input_cell(r["question"]) in ("arithmetic", "extreme") else "direct" for r in rows})
    controls["cv_selected_fixed"] = plan_metrics(rows, shared, cv_fixed)
    candidates = {penalty: plan_metrics(rows, shared, choices) for penalty, choices in cv_choices.items()}
    baseline = controls["cv_selected_fixed"]
    qualified = [p for p, result in candidates.items() if result["correct"] >= baseline["correct"]
        and result["complete_charts"] >= baseline["complete_charts"]
        and result["costs"]["tokens"] < baseline["costs"]["tokens"]
        and result["costs"]["wall_seconds"] <= baseline["costs"]["wall_seconds"]]
    selected = min(qualified, key=lambda p: (-candidates[p]["correct"], candidates[p]["costs"]["tokens"], float(p))) if qualified else None
    for penalty, policy in fitted.items():
        write_new(output/("policy-"+penalty+".json"), asdict(policy))
    write_new(output/"diagnostics.json", dict(controls=controls, cv=candidates, folds=folds, fold_ids=fold_ids,
        cv_choices=cv_choices, cv_fixed=cv_fixed, full_fit_choices={p: choose(policy, rows, binding) for p, policy in fitted.items()},
        fit_seconds=fit_seconds, trained_policies=len(fitted), retained_fold_policies=12,
        selection=dict(penalty=selected, candidate_active=False, independent_validation_complete=False),
        notes=["Extraction costs included once per chart for each evaluated plan.",
               "CV is a training diagnostic; shared fold training and selection prevent an independent-test claim.",
               "No base-reader parameters trained. Policy has at most four context-cell choices.",
               "Peak memory and energy not measured; no target-edge qualification."]))
    print(json.dumps(dict(controls=controls, cv=candidates, selected=selected, fit_seconds=fit_seconds)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    train(args.data, args.runs, args.output)

"""Export only aggregate ChartQA TRAIN diagnostics from pinned local records.

Copyright (c) 2026 Prashant Jagtap. MIT License.
No images, questions, answers, case IDs, response text, or dataset labels are exported.
"""

import argparse
import collections
import hashlib
import json
from pathlib import Path

from chartqa_prepare import file_digest
from chartqa_training_policy import ACTIONS, input_cell
from train_chartqa_policy import load_records, plan_metrics


def compact(metrics):
    return {key: metrics[key] for key in (
        "questions", "correct", "stripped_exact", "complete_charts", "charts",
        "errors", "abstentions", "extractions_charged", "costs", "subgroup")}


def summary(data, runs, policy):
    data, runs, policy = (Path(path).resolve() for path in (data, runs, policy))
    registration = json.loads((runs/"registration.json").read_bytes())
    complete = json.loads((runs/"complete.json").read_bytes())
    fit_registration = json.loads((policy/"registration.json").read_bytes())
    diagnostics = json.loads((policy/"diagnostics.json").read_bytes())
    if (fit_registration["reader"] != file_digest(runs/"registration.json")
            or fit_registration["completion"] != file_digest(runs/"complete.json")
            or fit_registration["source"] != file_digest(Path(__file__).with_name("train_chartqa_policy.py"))
            or diagnostics["selection"]["candidate_active"]
            or diagnostics["selection"]["independent_validation_complete"]):
        raise ValueError("policy output is not bound to the frozen training run")
    rows, shared, _ = load_records(data, runs)
    by_question = collections.defaultdict(dict)
    by_cell = collections.defaultdict(lambda: collections.Counter())
    for row in rows:
        by_question[row["id"]][row["mode"]] = row
    if len(by_question) != 143 or len(shared) != 64:
        raise ValueError("unexpected train cohort")
    comparisons = collections.Counter()
    for arms in by_question.values():
        if set(arms) != set(ACTIONS):
            raise ValueError("matched arms required")
        cell = input_cell(arms["direct"]["question"])
        by_cell[cell]["questions"] += 1
        for action in ACTIONS:
            by_cell[cell][action] += int(arms[action]["correct"])
        direct = arms["direct"]["correct"]
        comparisons["any_arm_correct"] += int(any(row["correct"] for row in arms.values()))
        comparisons["all_arms_wrong"] += int(not any(row["correct"] for row in arms.values()))
        for action in ("memory", "program"):
            comparisons[action+"_rescues_direct"] += int(not direct and arms[action]["correct"])
            comparisons[action+"_loses_direct"] += int(direct and not arms[action]["correct"])
    controls = {name: compact(value) for name, value in diagnostics["controls"].items()}
    cv = {name: compact(value) for name, value in diagnostics["cv"].items()}
    for action in ACTIONS:
        choices = {key: action for key in by_question}
        if compact(plan_metrics(rows, shared, choices)) != controls[action]:
            raise ValueError("control metric differs on replay")
    if compact(plan_metrics(rows, shared, diagnostics["cv_fixed"])) != controls["cv_selected_fixed"]:
        raise ValueError("fixed-fold control differs on replay")
    for penalty, choices in diagnostics["cv_choices"].items():
        if compact(plan_metrics(rows, shared, choices)) != cv[penalty]:
            raise ValueError("cross-validation metric differs on replay")
    if any(value != "direct" for choices in diagnostics["cv_choices"].values()
           for value in choices.values()):
        raise ValueError("this recorded CV did not select only direct")
    manifest = json.loads((data/"manifest.json").read_bytes())
    return {
        "schema": 1,
        "owner": "Prashant Jagtap",
        "cohort": "ChartQA public TRAIN subset; 64 charts, 143 questions, 2-4 questions per chart",
        "dataset_revision": manifest["revision"],
        "dataset_license": "ChartQA upstream GPL-3.0; dataset bytes are not redistributed here",
        "reader_model": {key: registration["model"][key] for key in ("name", "digest")},
        "reader_quantization": registration["model"]["details"]["quantization_level"],
        "reader_protocol": registration["protocol"],
        "model_unchanged": complete["model_unchanged"],
        "model_calls": complete["model_calls"],
        "paired_records": len(rows),
        "reader_run_seconds": complete["run_wall_seconds"],
        "reader_registration_sha256": file_digest(runs/"registration.json"),
        "reader_completion_sha256": file_digest(runs/"complete.json"),
        "policy_diagnostics_sha256": file_digest(policy/"diagnostics.json"),
        "summary_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "scorer": "native relaxed 5% numeric tolerance or stripped string match; exact also recorded",
        "cost_accounting": "Each selected query plus extraction/build once per chart if a dependent route is used; warmup, idle power and model load excluded",
        "controls": controls,
        "cv": cv,
        "cells": {cell: dict(counts) for cell, counts in sorted(by_cell.items())},
        "paired_comparisons": dict(comparisons),
        "folds": 4,
        "fold_group": "chart image; 16 evaluation charts per fold",
        "policy_candidates": diagnostics["trained_policies"],
        "retained_fold_policies": diagnostics["retained_fold_policies"],
        "selected_penalty": diagnostics["selection"]["penalty"],
        "candidate_active": False,
        "independent_validation_complete": False,
        "limitations": [
            "Training-only exploratory diagnostics; no untouched validation or test score.",
            "The direct route wins on this cohort; no learned visual route qualifies.",
            "Base reader weights were unchanged; fitted CPU policies are not a fine-tuned model.",
            "Near-duplicate images and base-model pretraining exposure cannot be ruled out.",
            "No peak memory, energy, target-device latency, or production scale measurement.",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    encoded = (json.dumps(summary(args.data, args.runs, args.policy), indent=2, sort_keys=True)+"\n").encode()
    if args.write:
        with args.summary.open("xb") as output:
            output.write(encoded)
    elif args.summary.read_bytes() != encoded:
        raise ValueError("published aggregate summary differs from pinned local evidence")
    print("ChartQA TRAIN aggregate summary verified; no case-level data exported.")


if __name__ == "__main__":
    main()

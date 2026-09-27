"""Fit and select a local abstention candidate without opening development keys.

Copyright (c) 2026 Prashant Jagtap. MIT License.
Training uses native independently labelled outcomes, not model self-ratings.
Selection is exploratory calibration, not a statistical deployment guarantee.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
from ruler_native import outside_repo, sha, write_new
from techqa_native import abstention, load_native, metrics
from techqa_policy import RIDGES, THRESHOLDS, features, fit, predict


def load_phase(data, runs, phase, model):
    manifest = json.loads((data/"manifest.json").read_bytes())
    registration = json.loads((runs/"registration.json").read_bytes())
    prepared = json.loads((runs/"prepared.json").read_bytes())
    if phase not in ("fit", "calibration"):
        raise ValueError("this training path cannot open development keys")
    if (registration["data_manifest"] != sha(data/"manifest.json")
            or prepared["registration"] != sha(runs/"registration.json")
            or prepared["compiled"][phase] != sha(runs/(phase+".compiled.json"))):
        raise ValueError("registered preparation changed")
    key_path = data/(phase+".keys.json")
    if sha(key_path) != manifest["files"][key_path.name]:
        raise ValueError("label bytes changed")
    keys = json.loads(key_path.read_bytes())
    compiled = {row["id"]: row for row in json.loads((runs/(phase+".compiled.json")).read_bytes())}
    run_dir = runs/(phase+"-"+model)
    complete = json.loads((run_dir/"complete.json").read_bytes())
    if (not complete["model_unchanged"] or complete["records"] != len(keys)*2
            or complete["source_hashes"] != registration["source_hashes"]):
        raise ValueError("complete paired fixed-model run required")
    results, direct_ids = {}, set()
    for name, expected in complete["files"].items():
        if sha(run_dir/name) != expected:
            raise ValueError("saved run changed")
        if name.endswith("-cited.json"):
            row = json.loads((run_dir/name).read_bytes())
            if row["id"] in results or row.get("mode") != "cited":
                raise ValueError("duplicate cited result")
            results[row["id"]] = row
        elif name.endswith("-direct.json"):
            row = json.loads((run_dir/name).read_bytes())
            if row["id"] in direct_ids or row.get("mode") != "direct":
                raise ValueError("duplicate direct result")
            direct_ids.add(row["id"])
    if set(keys) != set(results) or set(keys) != set(compiled) or direct_ids != set(keys):
        raise ValueError("complete key/prediction/input coverage required")
    return keys, compiled, results


def accepted(row):
    return (not row.get("error") and row.get("final_scope_check")
            and (row.get("check") or {}).get("status") == "source_bound"
            and row.get("bound_prediction") is not None and not row.get("truncated"))


def train(data, runs, output, model):
    data, runs, output = map(outside_repo, (data, runs, output))
    reader_plan = json.loads((runs/"registration.json").read_bytes())
    if any(sha(Path(__file__).with_name(name)) != expected for name, expected in reader_plan["policy_source_hashes"].items()):
        raise ValueError("policy or analysis source changed after reader registration")
    manifest = json.loads((data/"manifest.json").read_bytes())
    native = load_native(Path(manifest["data"])/"techqa_evaluation.py")
    fk, fc, fr = load_phase(data, runs, "fit", model)
    ck, cc, cr = load_phase(data, runs, "calibration", model)
    fit_ids = [key for key in sorted(fk) if accepted(fr[key])]
    cal_ids = sorted(ck)
    cal_candidates = {key: cr[key]["bound_prediction"] if accepted(cr[key]) else abstention() for key in cal_ids}
    baseline, _ = metrics(native, ck, cal_candidates)
    # Native utility: answering is worth span F1 on positives, -1 on negatives,
    # relative to explicit abstention. Invalid source bindings cannot be admitted.
    _, f1 = metrics(native, fk, {key: fr[key]["bound_prediction"] if accepted(fr[key]) else abstention() for key in fk})
    x = [features(fc[key], fr[key]) for key in fit_ids]
    y = [f1[key] if fk[key]["ANSWERABLE"] == "Y" else -1.0 for key in fit_ids]
    cx = [features(cc[key], cr[key]) for key in cal_ids]
    output.mkdir(parents=True, exist_ok=False)
    source_names = ("train_techqa_policy.py", "techqa_policy.py", "techqa_native.py", "techqa_context.py")
    write_new(output/"registration.json", dict(
        source_hashes={name: sha(Path(__file__).with_name(name)) for name in source_names},
        data_manifest=sha(data/"manifest.json"), reader_registration=sha(runs/"registration.json"),
        completed_runs={phase: sha(runs/(phase+"-"+model)/"complete.json") for phase in ("fit", "calibration")},
        model=model, ridges=RIDGES, thresholds=THRESHOLDS, eligible_fit=len(fit_ids),
        minimum_fit=20, development_keys_opened=False,
        selection="preserve calibration positive F1 exactly; reduce false-positive count and improve native F1; otherwise retain fixed citation check",
        active=False))
    tick = time.perf_counter()
    grid, candidates = [], []
    if len(fit_ids) >= 20:
        for ridge in RIDGES:
            policy = fit(x, y, ridge)
            scores = predict(policy, cx)
            quantized = predict(policy, cx, quantized=True)
            write_new(output/(f"ridge-{ridge:g}.json"), policy)
            for threshold in THRESHOLDS:
                predictions = {key: cal_candidates[key] if scores[i] >= threshold else abstention() for i, key in enumerate(cal_ids)}
                measured, _ = metrics(native, ck, predictions)
                eligible = (measured["positive_f1"] >= baseline["positive_f1"]-1e-12
                            and measured["false_positive_count"] < baseline["false_positive_count"]
                            and measured["QA_F1"] > baseline["QA_F1"]+1e-12)
                row = dict(ridge=ridge, threshold=threshold, eligible=eligible, metrics=measured,
                           int16_max_score_difference=float(np.max(np.abs(scores-quantized))),
                           int16_gate_flips=int(np.sum((scores >= threshold) != (quantized >= threshold))))
                grid.append(row)
                if eligible:
                    candidates.append(row)
    chosen = min(candidates, key=lambda row: (-row["metrics"]["QA_F1"], row["ridge"], row["threshold"])) if candidates else None
    selected = dict(mode="fixed_citation_check", active=False, reason="no learned candidate met calibration preservation rule")
    if chosen:
        selected = dict(mode="learned_expected_utility", active=False, ridge=chosen["ridge"], threshold=chosen["threshold"],
                        checkpoint=f"ridge-{chosen['ridge']:g}.json", checkpoint_sha256=sha(output/f"ridge-{chosen['ridge']:g}.json"),
                        inference="FP64; int16 storage is diagnostic only", reason="calibration-selected; needs untouched evaluation and retention qualification")
    write_new(output/"selection.json", selected)
    write_new(output/"training.json", dict(eligible_fit=len(fit_ids), fit_total=len(fk), calibration_total=len(ck),
                                            baseline=baseline, grid=grid, selected=selected, seconds=time.perf_counter()-tick,
                                            development_keys_opened=False, candidate_activated=False,
                                            selection_sha256=sha(output/"selection.json"),
                                            fit_labels="public native ground truth; no model self-judging",
                                            uncertainty="shared documents and adaptive calibration; no formal risk guarantee"))
    print(json.dumps(dict(selected=selected, eligible_fit=len(fit_ids), baseline=baseline)), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("data", "runs", "output"):
        parser.add_argument("--"+name, type=Path, required=True)
    parser.add_argument("--model", choices=("small", "modern"), required=True)
    args = parser.parse_args()
    train(args.data, args.runs, args.output, args.model)

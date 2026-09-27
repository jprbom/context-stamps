"""Native development evaluation only after frozen local predictions complete.

Copyright (c) 2026 Prashant Jagtap. MIT License.
This is the native public development split, not the retired official test set.
"""

import argparse
import collections
import json
import statistics
import time
from pathlib import Path

import numpy as np
from ruler_native import outside_repo, sha, write_new
from techqa_context import digest, terms
from techqa_native import abstention, load_native, metrics
from techqa_policy import features, predict
from train_techqa_policy import accepted


def evaluate(data, runs, trained, output, model):
    data, runs, trained, output = map(outside_repo, (data, runs, trained, output))
    manifest = json.loads((data/"manifest.json").read_bytes())
    registration = json.loads((runs/"registration.json").read_bytes())
    if any(sha(Path(__file__).with_name(name)) != expected for name, expected in registration["policy_source_hashes"].items()):
        raise ValueError("policy or analysis source changed after reader registration")
    prepared = json.loads((runs/"prepared.json").read_bytes())
    freeze = json.loads((runs/"development-policy-freeze.json").read_bytes())
    if (prepared["registration"] != sha(runs/"registration.json")
            or prepared["compiled"]["development"] != sha(runs/"development.compiled.json")
            or registration["data_manifest"] != sha(data/"manifest.json")
            or freeze != dict(selection=sha(trained/"selection.json"), training=sha(trained/"training.json"),
                              registration=sha(trained/"registration.json"))):
        raise ValueError("frozen preparation or policy changed")
    run_dir = runs/("development-"+model)
    complete = json.loads((run_dir/"complete.json").read_bytes())
    if (complete["records"] != 620 or not complete["model_unchanged"]
            or complete["source_hashes"] != registration["source_hashes"]):
        raise ValueError("complete paired frozen-model predictions required before opening keys")
    results = {"direct": {}, "cited": {}}
    for name, expected in complete["files"].items():
        if sha(run_dir/name) != expected:
            raise ValueError("prediction evidence changed")
        if name.endswith(("-cited.json", "-direct.json")):
            row = json.loads((run_dir/name).read_bytes())
            if row["id"] in results[row["mode"]]:
                raise ValueError("duplicate prediction")
            results[row["mode"]][row["id"]] = row
    compiled = {r["id"]: r for r in json.loads((runs/"development.compiled.json").read_bytes())}
    if any(set(rows) != set(compiled) for rows in results.values()):
        raise ValueError("every declared input requires both outcomes")
    # Keys are opened only after all 620 outputs and policy bindings verify.
    key_path = data/"development.keys.json"
    if sha(key_path) != manifest["files"][key_path.name]:
        raise ValueError("native keys changed")
    keys = json.loads(key_path.read_bytes())
    if set(keys) != set(compiled):
        raise ValueError("exact development key coverage required")
    native = load_native(Path(manifest["data"])/"techqa_evaluation.py")
    selection = json.loads((trained/"selection.json").read_bytes())
    training_registration = json.loads((trained/"registration.json").read_bytes())
    ids = sorted(keys)
    arms = {name: {} for name in ("direct_raw", "direct_exact_span", "cited_raw", "cited_checked", "learned_gate", "abstain_all")}
    policy, scores, quant_scores = None, None, None
    if selection["mode"] == "learned_expected_utility":
        checkpoint = trained/selection["checkpoint"]
        if sha(checkpoint) != selection["checkpoint_sha256"]:
            raise ValueError("selected checkpoint changed")
        policy = json.loads(checkpoint.read_bytes())
        tick = time.perf_counter()
        x = [features(compiled[qid], results["cited"][qid]) for qid in ids]
        scores = predict(policy, x)
        policy_seconds = time.perf_counter()-tick
        quant_scores = predict(policy, x, quantized=True)
    else:
        policy_seconds = 0.0
    invalid = dict(doc_id="__INVALID_LOCAL_PREDICTION__", score=1.0, start_offset=0, end_offset=1)
    for i, qid in enumerate(ids):
        for mode in ("direct", "cited"):
            row = results[mode][qid]
            # A hallucinated/unlocatable raw answer is an incorrect answer, not
            # a correct abstention. Compiler/parser failures also receive zero.
            raw = invalid if row.get("error") or row.get("truncated") else (row.get("prediction") or (invalid if row.get("answer") is not None else abstention()))
            arms[mode+"_raw"][qid] = raw
        direct, cited = results["direct"][qid], results["cited"][qid]
        arms["direct_exact_span"][qid] = (invalid if direct.get("error") else
                                           (direct.get("prediction") if not direct.get("truncated") else None) or abstention())
        checked = cited["bound_prediction"] if accepted(cited) else abstention()
        if compiled[qid].get("compile_error"):
            checked = invalid
        arms["cited_checked"][qid] = checked
        arms["learned_gate"][qid] = checked if scores is None or scores[i] >= selection["threshold"] else abstention()
        if compiled[qid].get("compile_error"):
            arms["learned_gate"][qid] = invalid
        arms["abstain_all"][qid] = abstention()
    summaries, per_question = {}, {}
    for name, predictions in arms.items():
        summaries[name], per_question[name] = metrics(native, keys, predictions)
    rng = np.random.default_rng(83)
    clusters = collections.defaultdict(list)
    for i, qid in enumerate(ids):
        clusters[digest(" ".join(terms(compiled[qid].get("question", qid))))].append(i)
    groups = list(clusters.values())
    samples = rng.integers(0, len(groups), size=(5000, len(groups)))
    group_lengths = np.array([len(group) for group in groups])
    contrasts = {}
    for left, right in (("cited_raw", "direct_raw"), ("cited_checked", "direct_exact_span"), ("learned_gate", "cited_checked")):
        delta = np.array([per_question[left][qid]-per_question[right][qid] for qid in ids])
        group_sums = np.array([delta[group].sum() for group in groups])
        means = group_sums[samples].sum(axis=1)/group_lengths[samples].sum(axis=1)
        contrasts[left+"__minus__"+right] = dict(mean=float(delta.mean()),
                                                descriptive_95_percentile=np.quantile(means, [.025, .975]).tolist(),
                                                gains=int(np.sum(delta > 0)), losses=int(np.sum(delta < 0)))
    costs = {}
    for mode, rows in results.items():
        values = list(rows.values())
        walls = [r["wall_seconds"] for r in values]
        costs[mode] = dict(input_tokens=sum(r.get("input_tokens", 0) for r in values),
                           output_tokens=sum(r.get("output_tokens", 0) or 0 for r in values),
                           request_wall_seconds=sum(walls), request_wall_median=statistics.median(walls),
                           request_wall_p95=float(np.quantile(walls, .95)),
                           processing_seconds=sum(r.get("process_seconds", 0) for r in values),
                           errors=sum(bool(r.get("error")) for r in values),
                           truncations=sum(bool(r.get("truncated")) for r in values),
                           raw_proposals=sum(r.get("answer") is not None for r in values),
                           model_calls=sum(r.get("called", False) for r in values))
    coverage = dict(answerable=0, complete_gold_span_selected=0, gold_doc_selected=0)
    for qid in ids:
        key = keys[qid]
        if key["ANSWERABLE"] != "Y":
            continue
        coverage["answerable"] += 1
        selected = compiled[qid].get("selected", [])
        coverage["gold_doc_selected"] += any(r["doc_id"] == key["DOCUMENT"] for r in selected)
        coverage["complete_gold_span_selected"] += any(r["doc_id"] == key["DOCUMENT"] and r["start"] <= int(key["START_OFFSET"]) and r["end"] >= int(key["END_OFFSET"]) for r in selected)
    output.mkdir(parents=True, exist_ok=False)
    write_new(output/"predictions.json", arms)
    write_new(output/"per-question.json", per_question)
    write_new(output/"summary.json", dict(
        model=model, trained_for=training_registration["model"], cross_model_diagnostic=model != training_registration["model"],
        metrics=summaries, contrasts=contrasts, costs=costs, selection=selection, support_diagnostic=coverage,
        bootstrap=dict(unit="normalized question cluster", clusters=len(groups), replicates=5000, seed=83,
                       scope="descriptive; shared documents still prevent an IID guarantee"),
        native_label_integrity=manifest["label_integrity"], input_duplicate_audit=manifest["duplicate_audit"],
        compile_seconds_shared=sum(r["compile_seconds"] for r in compiled.values()),
        policy_decision_seconds=policy_seconds, policy_feature_construction_excluded=False,
        int16_gate_flips=None if scores is None else int(np.sum((scores >= selection["threshold"]) != (quant_scores >= selection["threshold"]))),
        policy_activation=False, base_weights_changed=False,
        limitations=["public native development split", "shared documents; bootstrap descriptive, not IID significance",
                     "unknown base-model pretraining exposure", "no retention/device qualification",
                     "post-generation gate cannot reduce the already incurred reader token cost",
                     "exact quotes do not establish semantic entailment"],
        evidence_hashes=dict(data_manifest=sha(data/"manifest.json"), reader_plan=sha(runs/"registration.json"),
                             completed_run=sha(run_dir/"complete.json"), trained=sha(trained/"training.json"),
                             keys=sha(key_path), analysis_source=sha(__file__))))
    print(json.dumps(dict(model=model, metrics=summaries, coverage=coverage)), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("data", "runs", "trained", "output"):
        parser.add_argument("--"+name, type=Path, required=True)
    parser.add_argument("--model", choices=("small", "modern"), required=True)
    args = parser.parse_args()
    evaluate(args.data, args.runs, args.trained, args.output, args.model)

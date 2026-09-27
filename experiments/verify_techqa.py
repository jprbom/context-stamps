"""Independent offset, ridge-fit, selection and cost replay for local TechQA.

Copyright (c) 2026 Prashant Jagtap. MIT License.
No network, model calls or executable historical-source loading. The verifier
does not attest that a quoted span entails an answer or rerun the original GPU.
"""

import argparse
import base64
import collections
import gzip
import hashlib
import json
import math
import statistics
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ARMS = ("direct_raw", "direct_exact_span", "cited_raw", "cited_checked", "learned_gate", "abstain_all")
ABSTAIN = dict(doc_id="", score=0.0, start_offset=-1, end_offset=-1)
INVALID = dict(doc_id="__INVALID_LOCAL_PREDICTION__", score=1.0, start_offset=0, end_offset=1)


def load(path):
    raw = path.read_bytes()
    if path.suffix == ".gz":
        with gzip.open(path, "rb") as stream:
            raw = stream.read(16*1024**2+1)
        if len(raw) > 16*1024**2:
            raise ValueError("bounded evidence archive required")
    return json.loads(raw)


def close(actual, expected):
    if type(expected) is dict:
        if set(actual) != set(expected):
            raise ValueError("replay field coverage differs")
        for key in expected:
            close(actual[key], expected[key])
    elif type(expected) is list:
        if len(actual) != len(expected):
            raise ValueError("replay length differs")
        for a, b in zip(actual, expected):
            close(a, b)
    elif type(expected) in (float, int) and type(actual) in (float, int):
        if not math.isclose(actual, expected, abs_tol=1e-9, rel_tol=1e-9):
            raise ValueError(f"numeric replay differs: {actual} versus {expected}")
    elif actual != expected:
        raise ValueError("replay value differs")


def point_score(key, prediction):
    if prediction is None:
        return 0.0, 0
    positive = key["ANSWERABLE"] == "Y"
    document = key["DOCUMENT"] if positive else ""
    if document.strip() != prediction["doc_id"].strip():
        return 0.0, 0
    start, end = (int(key["START_OFFSET"]), int(key["END_OFFSET"])) if positive else (-1, -1)
    pstart, pend = prediction["start_offset"], prediction["end_offset"]
    if end == start or pend == pstart:
        return float(end-start == pend-pstart), 1
    overlap = max(0, min(end, pend)-max(start, pstart))
    return 2*overlap/(end-start+pend-pstart), 1


def measure(keys, predictions):
    if set(keys) != set(predictions):
        raise ValueError("complete declared key coverage required")
    values = {key: point_score(keys[key], predictions[key]) for key in keys}
    positive = [key for key in keys if keys[key]["ANSWERABLE"] == "Y"]
    negative = [key for key in keys if key not in positive]
    answered = [key for key in keys if predictions[key]["doc_id"]]
    def native_fields(ids):
        return dict(QA_F1=100*sum(values[k][0] for k in ids)/len(ids),
                    IR_Precision=100*sum(values[k][1] for k in ids)/len(ids), Total_Questions=len(ids))
    result = native_fields(list(keys))
    for prefix in ("HasAns_", "HasAns_Top_1_"):
        result.update({prefix+k: v for k, v in native_fields(positive).items()})
    result.update(positive_count=len(positive), negative_count=len(negative), answered_count=len(answered),
        positive_f1=sum(values[k][0] for k in positive)/len(positive),
        false_positive_count=sum(bool(predictions[k]["doc_id"]) for k in negative),
        abstained_positive_count=sum(not predictions[k]["doc_id"] for k in positive),
        answered_f1=sum(values[k][0] for k in answered)/len(answered) if answered else None)
    return result, {key: value[0] for key, value in values.items()}


def accepted(row):
    return (not row["error"] and row["final_scope_check"]
            and (row["check"] or {}).get("status") == "source_bound"
            and row["bound_prediction"] is not None and not row["truncated"])


def replay_canaries():
    keys = {"yes": dict(ANSWERABLE="Y", DOCUMENT="doc", START_OFFSET=10, END_OFFSET=20),
            "no": dict(ANSWERABLE="N", DOCUMENT="-", START_OFFSET="-", END_OFFSET="-")}
    exact = dict(doc_id="doc", score=1.0, start_offset=10, end_offset=20)
    cases = (("exact", exact, ABSTAIN), ("wrong_document", dict(exact, doc_id="other"), ABSTAIN),
             ("partial_span", dict(exact, end_offset=15), ABSTAIN),
             ("no_overlap", dict(exact, start_offset=30, end_offset=40), ABSTAIN),
             ("abstain_all", ABSTAIN, ABSTAIN), ("answer_negative", exact, exact))
    result = {name: measure(keys, dict(yes=yes, no=no))[0]["QA_F1"] for name, yes, no in cases}
    if any(point_score(key, None)[0] != 0 for key in keys.values()):
        raise ValueError("missing predictions must score zero")
    return result


def reconstruct(rows, contexts):
    arms = {name: {} for name in ARMS}
    for qid in contexts:
        direct, cited = rows["direct"][qid], rows["cited"][qid]
        for mode, row in (("direct", direct), ("cited", cited)):
            raw = INVALID if row["error"] or row["truncated"] else row["prediction"] or (INVALID if row["answer_present"] else ABSTAIN)
            arms[mode+"_raw"][qid] = raw
        arms["direct_exact_span"][qid] = INVALID if direct["error"] else ((direct["prediction"] if not direct["truncated"] else None) or ABSTAIN)
        checked = cited["bound_prediction"] if accepted(cited) else ABSTAIN
        if contexts[qid]["compile_error"]:
            checked = INVALID
        arms["cited_checked"][qid] = checked
        arms["learned_gate"][qid] = checked  # This recorded study selected the fixed check.
        arms["abstain_all"][qid] = ABSTAIN
    return arms


def verify(directory):
    directory = directory.resolve()
    manifest = load(directory/"manifest.json")
    for name, expected in manifest["files"].items():
        path = (directory/name).resolve()
        if directory not in path.parents or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError("published artifact changed: "+name)
    for name, expected in manifest.get("figures", {}).items():
        path = (ROOT/name).resolve()
        if ROOT not in path.parents or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError("published figure changed: "+name)
    archive = load(directory/"historical-sources.json.gz")
    for fingerprint in archive["hashes"].values():
        if hashlib.sha256(base64.b64decode(archive["content_by_sha256"][fingerprint], validate=True)).hexdigest() != fingerprint:
            raise ValueError("measured source snapshot differs")
    independent_canaries = replay_canaries()
    for case in load(directory/"native-canaries.json")["cases"]:
        if case["name"] == "missing_is_zero":
            if not case["passed"]:
                raise ValueError("native missing-prediction canary failed")
        else:
            close(independent_canaries[case["name"]], case["observed"])
            close(independent_canaries[case["name"]], case["expected"])
    targets, contexts = load(directory/"targets.json.gz"), load(directory/"contexts.json.gz")
    records = load(directory/"reader-records.json.gz")
    if len(records) != 1774 or len({(r["phase"], r["mode"], r["id"]) for r in records}) != 1774:
        raise ValueError("all measured reader records required")
    by_phase, arms_by_phase = {}, {}
    for phase, count in (("fit", 400), ("calibration", 177), ("development", 310)):
        rows = {mode: {r["id"]: r for r in records if r["phase"] == phase and r["mode"] == mode} for mode in ("direct", "cited")}
        if len(targets[phase]) != count or set(contexts[phase]) != set(targets[phase]) or any(set(v) != set(targets[phase]) for v in rows.values()):
            raise ValueError("complete phase and control coverage required")
        by_phase[phase] = rows
        arms_by_phase[phase] = reconstruct(rows, contexts[phase])
    summary, selection = load(directory/"summary.json"), load(directory/"selection.json")
    if selection["mode"] != "fixed_citation_check" or selection["active"] or summary["policy_activation"] or summary["base_weights_changed"]:
        raise ValueError("recorded inactive fixed selection changed")
    arms = arms_by_phase["development"]
    close(arms, load(directory/"predictions.json.gz"))
    actual = {}
    for name in ARMS:
        measured, actual[name] = measure(targets["development"], arms[name])
        close(measured, summary["metrics"][name])
    close(actual, load(directory/"per-question.json.gz"))
    training = load(directory/"training.json")
    if len(training["grid"]) != 18 or training["development_keys_opened"] or training["candidate_activated"]:
        raise ValueError("complete pre-development inactive calibration record required")
    fit_keys, cal_keys = targets["fit"], targets["calibration"]
    fit_rows, cal_rows = by_phase["fit"]["cited"], by_phase["calibration"]["cited"]
    eligible = [key for key in sorted(fit_keys) if accepted(fit_rows[key])]
    x = np.asarray([fit_rows[key]["observable_features"] for key in eligible], dtype=np.float64)
    y = np.asarray([point_score(fit_keys[key], fit_rows[key]["bound_prediction"])[0] if fit_keys[key]["ANSWERABLE"] == "Y" else -1.0 for key in eligible])
    if x.shape != (129, 12) or training["eligible_fit"] != 129:
        raise ValueError("registered eligible training population differs")
    mean, scale = x.mean(axis=0), np.maximum(x.std(axis=0), 1e-6)
    z = np.column_stack((np.ones(len(x)), np.clip((x-mean)/scale, -8, 8)))
    cal_ids = sorted(cal_keys)
    cx = np.asarray([cal_rows[key]["observable_features"] for key in cal_ids])
    cz = np.column_stack((np.ones(len(cx)), np.clip((cx-mean)/scale, -8, 8)))
    baseline, _ = measure(cal_keys, arms_by_phase["calibration"]["cited_checked"])
    close(baseline, training["baseline"])
    for ridge in (1, 10, 100):
        saved = load(directory/f"ridge-{ridge}.json")
        # Independent augmented least-squares formulation, not the fitter's
        # normal-equation solver. The intercept is deliberately unpenalized.
        penalty = np.diag([0.0]+[math.sqrt(ridge)]*12)
        fitted = np.linalg.lstsq(np.vstack((z, penalty)), np.r_[y, np.zeros(13)], rcond=None)[0]
        close(fitted.tolist(), saved["weights"])
        close(mean.tolist(), saved["mean"])
        close(scale.tolist(), saved["scale"])
        scores = np.clip(cz@np.asarray(saved["weights"]), -1, 1)
        quantized = np.clip(cz@(np.asarray(saved["int16_weights"])*saved["int16_scale"]), -1, 1)
        for grid in (g for g in training["grid"] if g["ridge"] == ridge):
            predictions = {key: arms_by_phase["calibration"]["cited_checked"][key] if scores[i] >= grid["threshold"] else ABSTAIN for i, key in enumerate(cal_ids)}
            measured, _ = measure(cal_keys, predictions)
            close(measured, grid["metrics"])
            admissible = (measured["positive_f1"] >= baseline["positive_f1"]-1e-12
                and measured["false_positive_count"] < baseline["false_positive_count"] and measured["QA_F1"] > baseline["QA_F1"]+1e-12)
            if admissible or grid["eligible"]:
                raise ValueError("recorded calibration rejection no longer holds")
            close(float(np.max(np.abs(scores-quantized))), grid["int16_max_score_difference"])
            close(int(np.sum((scores >= grid["threshold"]) != (quantized >= grid["threshold"]))), grid["int16_gate_flips"])
    for mode, records_by_id in by_phase["development"].items():
        rows = list(records_by_id.values())
        walls = [r["wall_seconds"] for r in rows]
        cost = dict(input_tokens=sum(r["input_tokens"] for r in rows), output_tokens=sum(r["output_tokens"] or 0 for r in rows),
            request_wall_seconds=sum(walls), request_wall_median=statistics.median(walls), request_wall_p95=float(np.quantile(walls, .95)),
            processing_seconds=sum(r["process_seconds"] for r in rows), errors=sum(bool(r["error"]) for r in rows),
            truncations=sum(bool(r["truncated"]) for r in rows), raw_proposals=sum(r["answer_present"] for r in rows),
            model_calls=sum(r["called"] for r in rows))
        close(cost, summary["costs"][mode])
    ids = sorted(targets["development"])
    clusters = collections.defaultdict(list)
    for i, qid in enumerate(ids):
        clusters[contexts["development"][qid]["cluster"]].append(i)
    groups = list(clusters.values())
    draws = np.random.default_rng(83).integers(0, len(groups), size=(5000, len(groups)))
    lengths = np.asarray([len(group) for group in groups])
    for contrast, expected in summary["contrasts"].items():
        left, right = contrast.split("__minus__")
        delta = np.asarray([actual[left][key]-actual[right][key] for key in ids])
        sums = np.asarray([sum(delta[group]) for group in groups])
        means = sums[draws].sum(axis=1)/lengths[draws].sum(axis=1)
        close(dict(mean=float(delta.mean()), descriptive_95_percentile=np.quantile(means, [.025, .975]).tolist(),
                   gains=int(np.sum(delta > 0)), losses=int(np.sum(delta < 0))), expected)
    print(json.dumps(dict(verified_records=1774, native_development_questions=310, independently_refitted_policies=3,
        rejected_calibration_settings=18, candidate_active=False, model_calls=0)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", type=Path, default=ROOT/"evidence"/"techqa-local-v1")
    verify(parser.parse_args().evidence)

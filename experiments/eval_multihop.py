"""Native MuSiQue scoring after frozen local predictions, plus resource review.

Copyright (c) 2026 Prashant Jagtap. MIT License.
The upstream scorer is downloaded separately and hash-checked before import.
Selected-context support is a retrieval diagnostic, not model-cited support.
"""

import argparse
import importlib
import json
import statistics
import sys
from pathlib import Path

import numpy as np
from ruler_native import outside_repo, sha, write_new

ARMS = ("full", "bm25", "pointwise", "diffusion")


def read(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def abstained(text):
    return text.strip().upper() == "UNKNOWN"


def load_native(source, expected):
    if any(sha(source/name) != value for name, value in expected.items()):
        raise ValueError("reviewed native scoring source changed")
    if (source/"metrics/__init__.py").exists() or (source/"__init__.py").exists():
        raise ValueError("unexpected package initializer")
    if any(n == "metrics" or n.startswith("metrics.") for n in sys.modules):
        raise ValueError("run native evaluation in a fresh process")
    sys.path.insert(0, str(source.resolve()))
    # Dotted filename is not an ordinary package; load only the reviewed file.
    from importlib.util import module_from_spec, spec_from_file_location
    spec = spec_from_file_location("musique_native_eval", source/"evaluate_v1.0.py")
    native = module_from_spec(spec)
    spec.loader.exec_module(native)
    metric = importlib.import_module("metrics.answer")
    return native, metric


def bootstrap(delta):
    rng = np.random.default_rng(71)
    array = np.asarray(delta, dtype=float)
    values = array[rng.integers(len(array), size=(5000, len(array)))].mean(axis=1)
    return dict(mean=float(array.mean()), percentile95=[float(v) for v in np.quantile(values, [.025, .975])],
                unit="one source-separated question pair", resamples=5000)


def score(data, run, source, output):
    data, run, source, output = map(outside_repo, (data, run, source, output))
    plan = json.loads((run/"registration.json").read_bytes())
    completed = json.loads((run/"complete.json").read_bytes())
    if completed["records"] != 512 or sha(run/"generations.jsonl") != completed["sha256"]:
        raise ValueError("complete frozen reader log required")
    preparation = json.loads((data/"registration.json").read_bytes())
    files = json.loads((data/"prepared.json").read_bytes())["files"]
    if any(sha(data/n) != h for n, h in files.items()):
        raise ValueError("prepared data changed")
    native, metric = load_native(source, preparation["scorer_hashes"])
    keys = read(data/"evaluation-keys.jsonl")
    inputs = {r["key"]: r for r in read(data/"evaluation-inputs.jsonl")}
    logs = read(run/"generations.jsonl")
    lookup = {(r["arm"], r["key"]): r for r in logs}
    if len(lookup) != 512 or set(lookup) != {(a, k["key"]) for a in ARMS for k in keys}:
        raise ValueError("duplicate, missing or unknown paired outputs")
    if plan["inputs"] != sha(data/"evaluation-inputs.jsonl"):
        raise ValueError("reader inputs differ from scoring inputs")
    output.mkdir(parents=True, exist_ok=False)
    write_new(output/"registration.json", dict(scorer_sha256=sha(__file__), generation_sha256=completed["sha256"],
                                               native_sources=preparation["scorer_hashes"],
                                               abstention="strip whitespace and case-fold; exact UNKNOWN only",
                                               grouping="64 source-separated answerable/unanswerable question pairs",
                                               primary_candidate="diffusion", controls=["full", "bm25", "pointwise"]))
    gold_path = output/"native-gold.jsonl"
    gold_path.write_text("\n".join(json.dumps(k["original"]) for k in keys)+"\n", encoding="utf-8")
    reports, per_case, per_group = {}, [], {}
    for arm in ARMS:
        native_predictions, case_rows, groups = [], [], {}
        for key in keys:
            row = lookup[arm, key["key"]]
            raw = key["original"]
            answerable = not abstained(row["prediction"])
            prediction = row["prediction"] if answerable else ""
            if row["error"]:
                prediction, answerable = "", False
            support = {p["idx"] for p in raw["paragraphs"] if p["is_supporting"]}
            selected = set(row["selected_indices"])
            if not selected <= {p["idx"] for p in inputs[key["key"]]["paragraphs"]}:
                raise ValueError("unknown selected source")
            aliases = [raw["answer"]]+raw["answer_aliases"]
            f1 = max(metric.compute_f1(a, prediction) for a in aliases)
            em = max(metric.compute_exact(a, prediction) for a in aliases)
            complete = support <= selected
            native_predictions.append(dict(id=raw["id"], predicted_answer=prediction,
                                            predicted_support_idxs=sorted(selected), predicted_answerable=answerable))
            item = dict(key=key["key"], group=key["group"], arm=arm, answerable=raw["answerable"],
                        predicted_answerable=answerable, prediction=row["prediction"], answer=raw["answer"],
                        f1=f1 if raw["answerable"] else None, em=em if raw["answerable"] else None,
                        selected_complete_support=complete if raw["answerable"] else None,
                        support_f1=2*len(support&selected)/(len(support)+len(selected)) if raw["answerable"] else None,
                        **{k: row[k] for k in ("input_tokens", "output_tokens", "generation_wall_seconds", "compile_seconds", "truncated", "error")})
            case_rows.append(item)
            groups.setdefault(key["group"], []).append(item)
        pred_path = output/(arm+"-native-predictions.jsonl")
        pred_path.write_text("\n".join(json.dumps(r) for r in native_predictions)+"\n", encoding="utf-8")
        native_scores = native.evaluate(str(pred_path), str(gold_path))
        positive = [r for r in case_rows if r["answerable"]]
        negative = [r for r in case_rows if not r["answerable"]]
        per_group[arm] = {}
        for group, items in groups.items():
            pos = next(r for r in items if r["answerable"])
            neg = next(r for r in items if not r["answerable"])
            sufficient = pos["predicted_answerable"] and not neg["predicted_answerable"]
            per_group[arm][group] = dict(f1=pos["f1"], em=pos["em"], grouped_f1=pos["f1"]*sufficient,
                                        complete_pair=int(bool(pos["em"] and sufficient)),
                                        complete_support=int(pos["selected_complete_support"]))
        totals = {n: sum(r[n] for r in case_rows) for n in ("input_tokens", "output_tokens", "generation_wall_seconds", "compile_seconds")}
        totals["total_model_tokens"] = totals["input_tokens"]+totals["output_tokens"]
        totals["median_request_seconds"] = statistics.median(r["generation_wall_seconds"] for r in case_rows)
        totals["p95_request_seconds"] = float(np.quantile([r["generation_wall_seconds"] for r in case_rows], .95))
        reports[arm] = dict(native=native_scores, native_support_interpretation="support F1 of selected context, not model-selected citations",
                            answerable_cases=len(positive), unanswerable_cases=len(negative),
                            exact_answers=sum(r["em"] for r in positive),
                            answer_f1=sum(r["f1"] for r in positive)/len(positive),
                            complete_support=sum(r["selected_complete_support"] for r in positive),
                            unanswerable_false_positives=sum(r["predicted_answerable"] for r in negative),
                            complete_pairs=sum(r["complete_pair"] for r in per_group[arm].values()),
                            truncations=sum(r["truncated"] for r in case_rows), errors=sum(bool(r["error"]) for r in case_rows), **totals)
        per_case.extend(case_rows)
    comparisons = {}
    for control in ("full", "bm25", "pointwise"):
        groups = sorted(per_group[control])
        comparisons[control] = {m: bootstrap([per_group["diffusion"][g][m]-per_group[control][g][m] for g in groups])
                                for m in ("f1", "em", "grouped_f1", "complete_pair", "complete_support")}
        comparisons[control]["exact_answer_gains"] = [g for g in groups if per_group["diffusion"][g]["em"] > per_group[control][g]["em"]]
        comparisons[control]["exact_answer_regressions"] = [g for g in groups if per_group["diffusion"][g]["em"] < per_group[control][g]["em"]]
    result = dict(arms=reports, paired_comparisons=comparisons, candidate_active=False,
                  boundary="One filtered development subset and reader. No retention, power, complete process-tree memory or edge qualification.")
    write_new(output/"summary.json", result)
    write_new(output/"case-scores.json", per_case)
    write_new(output/"group-scores.json", per_group)
    print(json.dumps(reports, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("data", "run", "source", "output"):
        parser.add_argument("--"+name, type=Path, required=True)
    args = parser.parse_args()
    score(args.data, args.run, args.source, args.output)

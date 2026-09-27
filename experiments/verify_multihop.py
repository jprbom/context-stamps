"""Offline replay of published predictions, targets and selector features.

Native scorer parity is checked against retained controlled examples. This
does not re-attest dataset truth or reproduce a GPU generation bit-for-bit.
"""

import collections
import gzip
import hashlib
import json
import math
import re
import statistics
import string
from pathlib import Path

import numpy as np
from relation_ranker import probabilities
from source_evidence import verify_sources

ROOT = Path(__file__).resolve().parents[1]
ARMS = ("full", "bm25", "pointwise", "diffusion")


def load(path):
    raw = path.read_bytes()
    return json.loads(gzip.decompress(raw) if path.suffix == ".gz" else raw)


def norm(text):
    text = "".join(c for c in text.lower() if c not in string.punctuation)
    return " ".join(re.sub(r"\b(a|an|the)\b", " ", text).split())


def answer_scores(prediction, answers):
    prediction = norm(prediction)
    scores = []
    for answer in answers:
        answer = norm(answer)
        p, a = prediction.split(), answer.split()
        common = sum((collections.Counter(p) & collections.Counter(a)).values())
        f1 = 2*common/(len(p)+len(a)) if p and a else float(p == a)
        scores.append((int(answer == prediction), f1))
    return max(s[0] for s in scores), max(s[1] for s in scores)


def support_f1(prediction, gold):
    p, g = set(prediction), set(gold)
    return 2*len(p & g)/(len(p)+len(g)) if p or g else 1.


def native_replay(golds, predictions):
    groups = collections.defaultdict(list)
    answers, supports = [], []
    for gold, prediction in zip(golds, predictions):
        if gold["id"] != prediction["id"]:
            raise ValueError("native pair order mismatch")
        em, f1 = answer_scores(prediction["predicted_answer"], [gold["answer"]]+gold["answer_aliases"])
        support = [p["idx"] for p in gold["paragraphs"] if p["is_supporting"]]
        sf1 = support_f1(prediction["predicted_support_idxs"], support)
        if gold["answerable"]:
            answers.append((em, f1))
            supports.append(sf1)
        groups[gold["id"]].append((gold["answerable"], prediction["predicted_answerable"], f1, sf1))
    grouped_answers, grouped_supports = [], []
    for group in groups.values():
        if sorted(r[0] for r in group) != [False, True]:
            raise ValueError("one complete question pair required")
        positive = next(r for r in group if r[0])
        sufficient = all(r[0] == r[1] for r in group)
        grouped_answers.append(positive[2]*sufficient)
        grouped_supports.append(positive[3]*sufficient)
    return dict(answer_em=round(statistics.mean(v[0] for v in answers), 3),
                answer_f1=round(statistics.mean(v[1] for v in answers), 3),
                support_f1=round(statistics.mean(supports), 3),
                group_answer_sufficiency_f1=round(statistics.mean(grouped_answers), 3),
                group_support_sufficiency_f1=round(statistics.mean(grouped_supports), 3))


def verify(directory):
    manifest = load(directory/"manifest.json")
    for name, expected in manifest["files"].items():
        path = (directory/name).resolve()
        if not path.is_relative_to(directory.resolve()) or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError("published artifact integrity failed")
    verify_sources(manifest["source_hashes"])
    for entry in load(directory/"native-canaries.json")["cases"]:
        if native_replay(entry["gold"], entry["predictions"]) != entry["native"]:
            raise ValueError("native scoring canary differs")
    inventories = load(directory/"partition-inventory.json.gz")
    union = {part: {f: set().union(*(set(g[f]) for g in groups.values())) for f in ("titles", "texts", "seeds")}
             for part, groups in inventories.items()}
    for a, b in (("training", "calibration"), ("training", "evaluation"), ("calibration", "evaluation")):
        if any(union[a][f] & union[b][f] for f in union[a]):
            raise ValueError("partition source/seed overlap")
    for partition in ("calibration", "evaluation"):
        seen = {f: set() for f in union[partition]}
        for group in inventories[partition].values():
            for field in seen:
                if seen[field] & set(group[field]):
                    raise ValueError("evaluation groups share sources or seed IDs")
                seen[field].update(group[field])
    targets = load(directory/"targets.json")
    if len(targets) != 128 or len({r["group"] for r in targets}) != 64:
        raise ValueError("complete reserved target cohort required")
    target_by_key = {r["key"]: r for r in targets}
    feat = load(directory/"evaluation-features.json.gz")
    policies = {name: load(directory/(name+".json")) for name in ("pointwise", "diffusion")}
    selected = {}
    for key, case in feat.items():
        if key not in target_by_key:
            raise ValueError("unexpected evaluation feature case")
        for arm in ARMS:
            if arm == "full":
                order = list(range(len(case["indices"])))
            else:
                values = case["bm25"] if arm == "bm25" else probabilities(case["x"], case["transition"], policies[arm]["weights"], policies[arm]["alpha"])
                order = sorted(range(len(values)), key=lambda i: (-values[i], case["hashes"][i], case["indices"][i]))[:6]
            selected[arm, key] = {case["indices"][i] for i in order}
    for model in ("small", "reference"):
        records = load(directory/f"{model}/generations.json.gz")
        summary = load(directory/f"{model}/summary.json")
        if len(records) != 512 or len({(r["arm"], r["key"]) for r in records}) != 512:
            raise ValueError("missing or duplicate reader results")
        if {(r["arm"], r["key"]) for r in records} != {(a, k) for a in ARMS for k in target_by_key}:
            raise ValueError("all reserved reader cases and controls required")
        by_arm = {}
        for arm in ARMS:
            rows = [r for r in records if r["arm"] == arm]
            golds, predictions, exact, complete, fp = [], [], 0, 0, 0
            group_values = collections.defaultdict(dict)
            for row in rows:
                key = target_by_key[row["key"]]
                if set(row["selected_indices"]) != selected[arm, row["key"]]:
                    raise ValueError("selector replay differs")
                if row["error"] is not None:
                    raise ValueError("this completed study must retain zero protocol errors")
                predicted_answerable = row["prediction"].strip().upper() != "UNKNOWN"
                answer = row["prediction"] if predicted_answerable else ""
                support = key["support"]
                em, f1 = answer_scores(answer, [key["answer"]]+key["aliases"])
                if key["answerable"]:
                    exact += em
                    complete += set(support) <= set(row["selected_indices"])
                    group_values[key["group"]].update(em=em, f1=f1, pos=predicted_answerable,
                                                      complete_support=int(set(support) <= set(row["selected_indices"])))
                else:
                    fp += predicted_answerable
                    group_values[key["group"]]["neg"] = predicted_answerable
                golds.append(dict(id=key["group"], answer=key["answer"], answer_aliases=key["aliases"],
                                  answerable=key["answerable"], paragraphs=[dict(idx=i, is_supporting=True) for i in support]))
                predictions.append(dict(id=key["group"], predicted_answer=answer, predicted_answerable=predicted_answerable,
                                        predicted_support_idxs=row["selected_indices"]))
            expected = summary["arms"][arm]
            for group in group_values.values():
                pos, neg = group.pop("pos"), group.pop("neg")
                sufficient = pos and not neg
                group.update(grouped_f1=group["f1"]*sufficient,
                             complete_pair=int(bool(group["em"] and sufficient)))
            by_arm[arm] = group_values
            if native_replay(golds, predictions) != expected["native"]:
                raise ValueError("native result replay differs")
            if (exact, complete, fp) != (expected["exact_answers"], expected["complete_support"], expected["unanswerable_false_positives"]):
                raise ValueError("quality count differs")
            for field in ("input_tokens", "output_tokens", "generation_wall_seconds", "compile_seconds"):
                if not math.isclose(sum(r[field] for r in rows), expected[field], rel_tol=1e-12, abs_tol=1e-12):
                    raise ValueError("resource totals differ")
            values = dict(total_model_tokens=sum(r["input_tokens"]+r["output_tokens"] for r in rows),
                          median_request_seconds=statistics.median(r["generation_wall_seconds"] for r in rows),
                          p95_request_seconds=float(np.quantile([r["generation_wall_seconds"] for r in rows], .95)),
                          complete_pairs=sum(v["complete_pair"] for v in group_values.values()),
                          answer_f1=statistics.mean(v["f1"] for v in group_values.values()),
                          truncations=sum(r["truncated"] for r in rows))
            if any(not math.isclose(v, expected[k], rel_tol=1e-12, abs_tol=1e-12) for k, v in values.items()):
                raise ValueError("quality/resource aggregates differ")
        for control in ("full", "bm25", "pointwise"):
            expected = summary["paired_comparisons"][control]
            groups = sorted(by_arm[control])
            for metric in ("f1", "em", "grouped_f1", "complete_pair", "complete_support"):
                delta = np.asarray([by_arm["diffusion"][g][metric]-by_arm[control][g][metric] for g in groups])
                rng = np.random.default_rng(71)
                boot = delta[rng.integers(len(delta), size=(5000, len(delta)))].mean(axis=1)
                if (not math.isclose(float(delta.mean()), expected[metric]["mean"], abs_tol=1e-12)
                        or not np.allclose(np.quantile(boot, [.025, .975]), expected[metric]["percentile95"], rtol=1e-12, atol=1e-12)):
                    raise ValueError("paired descriptive interval differs")
            for label, sign in (("gains", 1), ("regressions", -1)):
                ids = [g for g in groups if sign*(by_arm["diffusion"][g]["em"]-by_arm[control][g]["em"]) > 0]
                if ids != expected["exact_answer_"+label]:
                    raise ValueError("gain/regression list differs")
    return dict(reader_requests=1024, question_pairs=64, native_metric_canaries=9,
                selector_and_score_replay=True, candidate_active=False)


if __name__ == "__main__":
    print(json.dumps(verify(ROOT/"evidence/multihop-v1")))

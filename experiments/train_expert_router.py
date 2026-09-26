"""Fit from training judgments, choose threshold on tuning, gate on calibration."""

import argparse
import gzip
import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from run_hybrid_retrieval import read, save, sha  # noqa: E402
from train_controller_v2 import baseline_records, load_partitions, paired_bound  # noqa: E402

from context_stamps.expert_router import LinearExpertRouter  # noqa: E402

OUT = ROOT / "evidence/runtime-v1"


def route_features(features):
    """Only observable candidate scores; no teacher output or relevance label."""
    f = np.asarray(features, dtype=np.float64)
    if f.ndim != 2 or f.shape[1] != 6 or not 10 <= len(f) <= 256 or not np.isfinite(f).all():
        raise ValueError("bounded candidate features required")
    dense, hybrid = np.argsort(-f[:, 0], kind="stable"), np.argsort(-f[:, 2], kind="stable")
    values = [len(f) / 256, len(set(dense[:10]) & set(hybrid[:10])) / 10]
    for col in (0, 2, 3):
        ranked = np.sort(f[:, col])[::-1]
        values.extend([ranked[0], ranked[0] - ranked[1], ranked[0] - ranked[9], ranked.std()])
    return [float(v) for v in values]


def dataset_rows(partitions):
    records = baseline_records(partitions)
    scores = {(r["dataset"], r["query_id"], r["method"]): r["ndcg10"] for r in records}
    rows = []
    for data in partitions:
        for i, qid in enumerate(data["meta"]["query_ids"]):
            rows.append(dict(dataset=data["name"], query_id=qid,
                features=route_features(data["F"][i][data["mask"][i]]),
                hybrid=scores[data["name"], qid, "hybrid"], fusion=scores[data["name"], qid, "fusion"]))
    return rows


def main(work):
    OUT.mkdir(exist_ok=True)
    train = dataset_rows(load_partitions(work, "train"))
    tune = dataset_rows(load_partitions(work, "tune"))
    calibration = dataset_rows(load_partitions(work, "calibration"))
    x = np.asarray([r["features"] for r in train])
    y = np.asarray([r["hybrid"] - r["fusion"] for r in train])
    means, scales = x.mean(0), np.maximum(x.std(0), 1e-6)
    z = np.column_stack([np.ones(len(x)), (x - means) / scales])
    # Equal total influence per training domain; no fitting on held-out labels.
    counts = {n: sum(r["dataset"] == n for r in train) for n in {r["dataset"] for r in train}}
    importance = np.asarray([len(train) / (len(counts) * counts[r["dataset"]]) for r in train])
    penalty = np.eye(z.shape[1]) * 10
    penalty[0, 0] = 0
    coef = np.linalg.solve(z.T @ (z * importance[:, None]) + penalty, z.T @ (y * importance))
    params = dict(means=tuple(map(float, means)), scales=tuple(map(float, scales)),
                  weights=tuple(map(float, coef[1:])), intercept=float(coef[0]))
    trials = []
    for threshold in (0., .01, .02, .04, .08, .16, 2.):
        model = LinearExpertRouter(**params, threshold=threshold)
        accepted = []
        for row in tune:
            score, _ = model.predict(row["features"])
            accepted.append(score is not None and score >= threshold)
        deltas = {n: float(np.mean([(r["hybrid"] - r["fusion"]) if a else 0 for r, a in zip(tune, accepted) if r["dataset"] == n])) for n in counts}
        trials.append(dict(threshold=threshold, cheap=sum(accepted), delta_by_domain=deltas, eligible=min(deltas.values()) >= 0))
    best = max((r for r in trials if r["eligible"]), key=lambda r: (r["cheap"], r["threshold"]))
    model = LinearExpertRouter(**params, threshold=best["threshold"])
    gates, approved = {}, []
    old_selection = read(ROOT / "evidence/controller-v2/selection.json")["decisions"]
    for name, old in old_selection.items():
        rows = [r for r in calibration if r["dataset"] == name]
        if old["deployed"] != "fusion" or not rows:
            gates[name] = dict(approved=False, reason="preserve_existing_dense_scope")
            continue
        predictions = [model.predict(r["features"])[0] for r in rows]
        cheap = [p is not None and p >= best["threshold"] for p in predictions]
        bound = paired_bound([r["hybrid"] if c else r["fusion"] for r, c in zip(rows, cheap)], [r["fusion"] for r in rows])
        passed = sum(cheap) >= 30 and bound["mean"] >= 0 and bound["lower"] >= -.001
        gates[name] = dict(approved=passed, cheap=sum(cheap), n=len(rows), delta=bound)
        if passed:
            approved.append(name)
    deployed = LinearExpertRouter(**params, threshold=best["threshold"], approved_scopes=tuple(approved))
    save(OUT / "router.json", asdict(deployed))
    save(OUT / "router-training.json", dict(train=len(train), tune=len(tune), calibration=len(calibration),
        trials=trials, selected_threshold=best["threshold"], gates=gates,
        source_sha256={p: sha(ROOT / p) for p in ("experiments/train_expert_router.py", "context_stamps/expert_router.py")},
        limitation="All calibration/test collections were previously inspected; this is an engineering regression round."))
    (OUT / "router-development.json.gz").write_bytes(gzip.compress(json.dumps(dict(train=train, tune=tune, calibration=calibration), separators=(",", ":")).encode(), mtime=0))
    print(json.dumps(dict(selected=best, gates=gates)), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--work", type=Path, required=True)
    main(p.parse_args().work)

"""Balanced metric-aware training, strong controls and frozen held-out evaluation.

Copyright (c) 2026 Prashant Jagtap. MIT License. No external inference endpoints.
"""

import argparse
import gzip
import json
import math
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))
from run_hybrid_retrieval import read, save, sha  # noqa: E402
from train_controller import batch, evaluate, scored_records  # noqa: E402

from context_stamps.contractive_controller import (  # noqa: E402
    ContractiveRanker,
    centered_distillation,
    load_contractive,
    ndcg_pair_loss,
    save_contractive,
)

OUT = ROOT / "evidence/controller-v2"
NAMES = ("scifact", "nfcorpus", "arguana", "scidocs", "fiqa")
BASELINES = ("dense", "hybrid", "bm25", "teacher", "fusion")


def load_partitions(work, split):
    manifest = read(OUT / "data-manifest.json")
    rows = []
    for info in manifest["partitions"]:
        if info["split"] != split:
            continue
        name = info["dataset"]
        path = work / ("controller-v2-data/" + name + "-" + split + ".npz")
        dpath = work / (("controller-v2-data/" if name == "fiqa" else "controller-v1-data/") + name + "-D.npy")
        if (sha(path) != info["array_sha256"] or sha(path.with_suffix(".json")) != info["metadata_sha256"]
                or sha(dpath) != info["document_sha256"]):
            raise ValueError("prepared data hash mismatch")
        arrays = dict(np.load(path, allow_pickle=False))
        arrays.update(D=np.load(dpath, allow_pickle=False), meta=read(path.with_suffix(".json")), name=name)
        if arrays["meta"]["query_ids"] != manifest["assignments"][name][split]:
            raise ValueError("query assignment mismatch")
        rows.append(arrays)
    return rows


def normalize_lists(values, mask):
    count = mask.sum(1, keepdims=True)
    mean = (values * mask).sum(1, keepdims=True) / count
    std = np.sqrt((((values - mean) ** 2) * mask).sum(1, keepdims=True) / count)
    return (values - mean) / np.maximum(std, 1e-9)


def baseline_records(partitions):
    result = []
    for data in partitions:
        mask = data["mask"]
        for method in ("teacher", "fusion"):
            score = data["teacher"].copy()
            if method == "fusion":
                score = .5 * normalize_lists(score, mask) + .5 * normalize_lists(data["F"][..., 2], mask)
            result.extend(scored_records(data, np.where(mask, score, -np.inf), method))
        for i, original in enumerate(data["meta"]["records"]):
            rel = {data["meta"]["document_ids"][j]: float(v) for j, v, valid in
                   zip(data["candidates"][i], data["Y"][i], mask[i]) if valid}
            for method in ("dense", "hybrid", "bm25"):
                ids = original["ranked_ids"][method]
                result.append(dict(dataset=data["name"], query_id=original["query_id"], method=method,
                    **original["metrics"][method], ranked_ids=ids, ranked_gains=[rel[j] for j in ids],
                    ideal=original["ideal"], total_relevant=original["total_relevant"]))
    return result


def values(records, name, method):
    selected = sorted((r for r in records if r["dataset"] == name and r["method"] == method),
                      key=lambda r: r["query_id"])
    return np.asarray([r["ndcg10"] for r in selected])


def paired_bound(a, b):
    if len(a) != len(b) or not len(a):
        raise ValueError("paired nonempty samples required")
    delta = np.asarray(a) - np.asarray(b)
    rng = np.random.default_rng(20260926)
    samples = np.concatenate([delta[rng.integers(0, len(delta), (500, len(delta)))].mean(1) for _ in range(20)])
    return dict(n=len(delta), mean=float(delta.mean()), lower=float(np.percentile(samples, 100 * .05 / 3)),
                upper=float(np.percentile(samples, 100 * (1 - .05 / 3))),
                lower_95=float(np.percentile(samples, 2.5)), upper_95=float(np.percentile(samples, 97.5)))


def balanced_indexes(training, rng, batches, per_domain=8):
    """Equal domain exposure; with replacement, no positive-only filtering."""
    for _ in range(batches):
        yield [(data, rng.integers(0, len(data["Q"]), per_domain)) for data in training]


def main(work):
    protocol = read(OUT / "protocol.json")
    if read(OUT / "data-manifest.json")["protocol_sha256"] != sha(OUT / "protocol.json"):
        raise ValueError("protocol changed after data preparation")
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for recorded RTX run")
    models = OUT / "models"
    models.mkdir(exist_ok=True)
    training, tuning = load_partitions(work, "train"), load_partitions(work, "tune")
    if {d["name"] for d in training} != {"scifact", "nfcorpus"}:
        raise ValueError("unexpected training domain")
    batches = math.ceil(sum(len(d["Q"]) for d in training) / protocol["batch_size"])
    source_paths = ["experiments/train_controller_v2.py", "context_stamps/contractive_controller.py",
                    "experiments/train_controller.py", "experiments/run_hybrid_retrieval.py"]
    run_config = dict(protocol_sha256=sha(OUT / "protocol.json"), data_manifest_sha256=sha(OUT / "data-manifest.json"),
        source_sha256={p: sha(ROOT / p) for p in source_paths}, batches_per_epoch=batches,
        sampling="8 queries per domain with replacement per batch; all lists eligible; equal-domain loss mean",
        optimizer="AdamW, lr0.0003, weight_decay0.01, gradient norm clip1.0, BF16 forward/FP32 loss",
        recurrence="Uniformly sample depth2/4/8 once per optimizer step; inference4; research stress32",
        gate="One-sided Bonferroni alpha0.05/3 for each of dense/hybrid/teacher; identical method is exact equality and exempt. At least50 calibration queries. Other failed comparisons fall back to dense.",
        timing="Data preparation/teacher labeling recorded separately. Training clock includes tune checks and checkpoint I/O.")
    config_path = OUT / "run-config.json"
    if config_path.exists() and read(config_path) != run_config:
        raise ValueError("run configuration changed; create a new experiment")
    save(config_path, run_config)
    trials_path = OUT / "training.json"
    trials = read(trials_path) if trials_path.exists() else []
    for recipe in protocol["recipes"]:
        for seed in protocol["seeds"]:
            trial = recipe + "-" + str(seed)
            checkpoint = models / (trial + ".safetensors")
            previous = next((r for r in trials if r["trial"] == trial), None)
            if previous:
                if sha(checkpoint) != previous["checkpoint_sha256"]:
                    raise ValueError("checkpoint changed")
                continue
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            rng = np.random.default_rng(seed)
            model = ContractiveRanker(recurrent=recipe != "balanced_ndcg").cuda()
            optimizer = torch.optim.AdamW(model.parameters(), lr=protocol["learning_rate"], weight_decay=.01)
            _, best = evaluate(model, tuning, trial)
            save_contractive(model, checkpoint)
            history, best_epoch = [dict(epoch=0, tune_ndcg10=best)], 0
            torch.cuda.reset_peak_memory_stats()
            start = time.perf_counter()
            for epoch in range(1, protocol["epochs"] + 1):
                model.train()
                losses = []
                for groups in balanced_indexes(training, rng, batches):
                    optimizer.zero_grad(set_to_none=True)
                    depth = int(rng.choice([2, 4, 8]))
                    total = 0.
                    for data, indexes in groups:
                        inputs = batch(data, indexes, "cuda")
                        y = torch.as_tensor(data["Y"][indexes], device="cuda")
                        teacher = torch.as_tensor(data["teacher"][indexes], device="cuda")
                        with torch.autocast("cuda", dtype=torch.bfloat16):
                            score = model(*inputs, steps=depth).float()
                        mask = inputs[3]
                        residual = (score - inputs[2][..., 2]).masked_fill(~mask, 0)
                        loss = ndcg_pair_loss(score, y, mask) + .01 * (residual.square().sum(1) / mask.sum(1)).mean()
                        if recipe == "distilled":
                            loss = loss + .1 * centered_distillation(score, teacher, mask)
                        if not torch.isfinite(loss):
                            raise FloatingPointError("non-finite training loss")
                        (loss / len(groups)).backward()
                        total += loss.item() / len(groups)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
                    optimizer.step()
                    losses.append(total)
                _, score = evaluate(model, tuning, trial)
                history.append(dict(epoch=epoch, loss=float(np.mean(losses)), tune_ndcg10=score))
                if score > best:
                    best, best_epoch = score, epoch
                    save_contractive(model, checkpoint)
                print(trial, "epoch", epoch, "tune", round(score, 5), flush=True)
            trials.append(dict(trial=trial, recipe=recipe, seed=seed, selected_epoch=best_epoch,
                tune_ndcg10=best, seconds=time.perf_counter() - start, history=history,
                parameters=sum(p.numel() for p in model.parameters()), checkpoint_sha256=sha(checkpoint),
                peak_allocated_bytes=torch.cuda.max_memory_allocated()))
            save(trials_path, trials)
            del model, optimizer
    selected = max(trials, key=lambda r: r["tune_ndcg10"])["trial"]
    model = load_contractive(models / (selected + ".safetensors")).cuda()
    tune_records = baseline_records(tuning) + evaluate(model, tuning, "student")[0]
    eligible = ("dense", "hybrid", "teacher", "fusion", "student")
    tune_means = {n: {m: float(values(tune_records, n, m).mean()) for m in eligible} for n in ("scifact", "nfcorpus")}
    chosen = {n: max(scores, key=scores.get) for n, scores in tune_means.items()}
    global_method = max(eligible, key=lambda m: np.mean([r[m] for r in tune_means.values()]))
    chosen["fiqa"] = global_method
    # No test arrays or metrics are accessed until selection and calibration are written.
    calibration = load_partitions(work, "calibration")
    cal_records = baseline_records(calibration) + evaluate(model, calibration, "student")[0]
    decisions = {}
    for name, method in chosen.items():
        a = values(cal_records, name, method)
        comparisons = {m: paired_bound(a, values(cal_records, name, m)) for m in ("dense", "hybrid", "teacher")}
        passed = len(a) >= 50 and all(m == method or r["lower"] > 0 for m, r in comparisons.items())
        decisions[name] = dict(candidate=method, deployed=method if passed else "dense", comparisons=comparisons,
                               qualified=bool(passed), queries=len(a))
    for name in ("arguana", "scidocs"):
        decisions[name] = dict(candidate=global_method, deployed="dense", qualified=False, reason="no scoped calibration")
    save(OUT / "selection.json", dict(selected_student=selected, tune_scores=tune_means,
        global_tune_method=global_method, decisions=decisions, limits="Existing SciFact/NFCorpus calibration has been inspected in v1. FiQA calibration is fresh locally. Test data never selects a model, recipe, seed or gate."))
    selection_hash = sha(OUT / "selection.json")
    test = load_partitions(work, "test")
    records = baseline_records(test)
    for row in trials:
        candidate = load_contractive(models / (row["trial"] + ".safetensors")).cuda()
        records.extend(evaluate(candidate, test, row["trial"])[0])
        del candidate
    records.extend(dict(r, method="student") for r in list(records) if r["method"] == selected)
    for name, decision in decisions.items():
        records.extend(dict(r, method="gated") for r in list(records)
                       if r["dataset"] == name and r["method"] == decision["deployed"])
    recurrent = max((r for r in trials if r["recipe"] == "contractive"), key=lambda r: r["tune_ndcg10"])["trial"]
    recurrent_model = load_contractive(models / (recurrent + ".safetensors")).cuda()
    for depth in (2, 8, 32):
        records.extend(evaluate(recurrent_model, test, "depth-" + str(depth), steps=depth)[0])
    summary = {}
    for name in NAMES:
        summary[name] = {}
        for method in sorted({r["method"] for r in records}):
            subset = [r for r in records if r["dataset"] == name and r["method"] == method]
            row = {key: float(np.mean([r[key] for r in subset])) for key in ("ndcg10", "recall10", "mrr10")}
            row["delta_vs_dense"] = paired_bound(values(records, name, method), values(records, name, "dense"))
            row["delta_vs_hybrid"] = paired_bound(values(records, name, method), values(records, name, "hybrid"))
            row["delta_vs_teacher"] = paired_bound(values(records, name, method), values(records, name, "teacher"))
            summary[name][method] = row
    save(OUT / "summary.json", summary)
    payload = dict(test=records, tune=tune_records, calibration=cal_records,
        candidate_diagnostics=[dict(dataset=d["name"], query_id=r["query_id"],
            old_candidates=r["old_candidates"], new_candidates=r["new_candidates"])
            for d in test for r in d["meta"]["records"]])
    (OUT / "rankings.json.gz").write_bytes(gzip.compress(json.dumps(payload, separators=(",", ":"), allow_nan=False).encode(), mtime=0))
    coverage = {d["name"]: {which: {key: float(np.mean([r[which][key] for r in d["meta"]["records"]]))
                 for key in ("recall", "oracle_ndcg10")} for which in ("old_candidates", "new_candidates")} for d in test}
    save(OUT / "candidate-coverage.json", coverage)
    # Bounded-state stress evidence, separate from accuracy.
    stresses = []
    with torch.inference_mode():
        for data in test:
            inputs = batch(data, np.arange(min(32, len(data["Q"]))), "cuda")
            u, p = recurrent_model.initial_state(*inputs)
            reference = u
            for _ in range(32):
                reference = recurrent_model.advance(u, p, reference)
            state = u
            for depth in range(1, 9):
                state = recurrent_model.advance(u, p, state)
                if depth in (2, 4, 8):
                    stresses.append(dict(dataset=data["name"], queries=len(u), depth=depth,
                        max_row_l2_error_to_32=float((state - reference).norm(dim=-1).max().item())))
    save(OUT / "recurrence.json", dict(trial=recurrent, stresses=stresses,
        limits="First32 queries per collection; fixed point proximity does not establish relevance, factuality or safety."))
    save(OUT / "runtime.json", dict(gpu=torch.cuda.get_device_name(), torch=torch.__version__, cuda=torch.version.cuda,
        python=platform.python_version(), numpy=np.__version__, threads=torch.get_num_threads(),
        training_seconds=sum(r["seconds"] for r in trials),
        teacher_preparation_seconds=sum(r["seconds"] for r in read(OUT / "data-manifest.json")["partitions"]),
        limits="Teacher preparation excludes initial FiQA encoding/download; training includes tuning but excludes final test evaluation."))
    if selection_hash != sha(OUT / "selection.json"):
        raise ValueError("selection changed during test")
    save(OUT / "manifest.json", dict(selection_sha256=selection_hash, selected_student=selected,
        run_config_sha256=sha(config_path), query_counts={d["name"]: len(d["Q"]) for d in test},
        methods=sorted({r["method"] for r in records}), source_sha256=run_config["source_sha256"]))
    print(json.dumps({n: {m: round(v["ndcg10"], 5) for m, v in rows.items() if m in (*BASELINES, "student", "gated")}
                      for n, rows in summary.items()}, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work", required=True, type=Path)
    os.environ["HF_HUB_OFFLINE"] = "1"
    main(parser.parse_args().work)

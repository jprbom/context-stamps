"""Local RTX training of judged-pool retention, with frozen regression controls.

Copyright (c) 2026 Prashant Jagtap. MIT License.
No new corpus download, model generation, teacher supervision or paid endpoint.
"""

import argparse
import gzip
import json
import platform
import subprocess
import sys
import threading
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from profile_context_training import sha, telemetry  # noqa: E402

from context_stamps.acquisition import (  # noqa: E402
    FEATURE_REVISION,
    TARGET_REVISION,
    AcquisitionModel,
    StopPolicy,
    Trajectory,
    assess_policy,
    prefix_features,
)

SOURCES = ("context_stamps/acquisition.py", "context_stamps/decisions/calibration.py",
           "experiments/train_acquisition.py", "experiments/profile_context_training.py")


def save(path, payload):
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def register(out):
    out.mkdir(parents=True, exist_ok=False)
    protocol = dict(schema=1, created_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        source_base=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        sources={p: sha(ROOT / p) for p in SOURCES},
        data_manifest_sha256=sha(ROOT / "evidence/controller-v2/data-manifest.json"),
        feature_revision=FEATURE_REVISION, target_revision=TARGET_REVISION,
        candidates="Original v2 dense96/hybrid96/BM25-64 union, hybrid-score order, no positive injection.",
        features="33 prefix statistics from six available retrieval scores. Entire pool's retrieval scores available before planning. No query IDs, judgments, teacher, document embeddings or future payloads in features.",
        target="All nonzero judged relevance gain in the fixed candidate pool retained, with at least one judged-positive candidate. No-positive pools have completeness false and gain targets zero. Not answer sufficiency or global recall.",
        heads=["pool_completeness_score", "remaining_judged_gain_fraction", "next_batch_judged_gain_fraction"],
        train=["scifact-train", "nfcorpus-train"], tune=["scifact-tune", "nfcorpus-tune"],
        calibration=["scifact-calibration", "nfcorpus-calibration", "fiqa-calibration"],
        test=["scifact-test", "nfcorpus-test", "arguana-test", "scidocs-test", "fiqa-test"],
        data_status="All datasets, including calibration and test, were previously used by this project. Disjoint for this fit, but regression/development evidence only; no untouched-final or production certification claim.",
        architectures=[0, 32], seeds=[7, 29, 61], epochs=80, batch_size=4096, learning_rate=.003,
        weight_decay=.001, loss="Binary cross entropy completeness + 0.25 * sum of squared remaining/next gain errors; each domain equally weighted, each query equally weighted within domain, each prefix equally weighted within query.",
        selection="Each checkpoint selected by equal-domain tune loss, including epoch0. Select one architecture/seed by tune loss only. Freeze all weights before loading calibration/test. Report six training/tuning controls and selected-model regression; deploy none by default.",
        thresholds=[.8, .9, .95, .99], alpha=.05, maximum_error=.05,
        calibration_rule="One error per early-stopped query trajectory; one-sided Clopper-Pearson, alpha divided by all 12 selected-model threshold/scope pairs. Minimize mean retained candidates among passing thresholds per scope; absent or failing scope uses all candidates. IDs do not establish independent queries.",
        controls=["fixed-8", "fixed-32", "fixed-128", "full-pool", "raw-selected-.90", "risk-gated"],
        reporting="Candidate counts and judged recall only. Selection-only batch1 CPU timing excludes retrieval, payload I/O, tokenizer, reader, network. No token/cost/end-to-end latency reduction inferred.",
        release="Opt-in experiment; existing qualified retrieval routes unchanged. Original code MIT; derived public-data coefficients/evidence CC BY-SA4 with attribution. No raw text or external pretrained weights published.")
    save(out / "protocol.json", protocol)
    print(json.dumps({"registered": out.name, "protocol_sha256": sha(out / "protocol.json")}), flush=True)


def load_partition(work, name, manifest):
    dataset, split = name.rsplit("-", 1)
    info = next(p for p in manifest["partitions"] if p["dataset"] == dataset and p["split"] == split)
    path = work / "controller-v2-data" / (name + ".npz")
    meta_path = path.with_suffix(".json")
    if sha(path) != info["array_sha256"] or sha(meta_path) != info["metadata_sha256"]:
        raise ValueError("pinned prepared data hash mismatch")
    meta = json.loads(meta_path.read_text())
    if meta["query_ids"] != manifest["assignments"][dataset][split]:
        raise ValueError("query assignment mismatch")
    with np.load(path, allow_pickle=False) as data:
        f, y, mask = data["F"], data["Y"], data["mask"]
    if not np.isfinite(f).all() or not np.isfinite(y).all() or (y < 0).any():
        raise ValueError("invalid prepared features or labels")
    x, targets, weights, records = [], [], [], []
    for i, qid in enumerate(meta["query_ids"]):
        valid = np.flatnonzero(mask[i])
        rows = tuple(tuple(float(v) for v in row) for row in f[i, valid])
        order, lengths, features = prefix_features(rows)
        relevance = y[i, valid][list(order)].astype(float)
        cumulative = np.cumsum(relevance)
        total = float(cumulative[-1])
        truth, labels, gains = [], [], []
        for j, k in enumerate(lengths):
            gain = float(cumulative[k - 1])
            complete = total > 0 and gain >= total - 1e-9
            next_gain = float(cumulative[lengths[min(j + 1, len(lengths) - 1)] - 1]) - gain
            truth.append(complete)
            labels.append([float(complete), (total - gain) / max(1., total), next_gain / max(1., total)])
            gains.append(gain)
        records.append(dict(query_id=qid, dataset=dataset, split=split, offset=len(x), count=len(lengths),
            lengths=lengths, complete=truth, gains=gains, pool_gain=total,
            positive_counts=[int(np.count_nonzero(relevance[:k])) for k in lengths],
            total_relevant=meta["records"][i]["total_relevant"],
            full_pool_positive_count=int(np.count_nonzero(relevance))))
        x.extend(features)
        targets.extend(labels)
        weights.extend([1 / len(lengths)] * len(lengths))
    return dict(name=name, x=np.asarray(x, np.float32), y=np.asarray(targets, np.float32),
                w=np.asarray(weights, np.float32), records=records, source=info)


def combine(parts, device, mean=None, scale=None):
    x = np.concatenate([p["x"] for p in parts])
    weights = np.concatenate([p["w"] / len(p["records"]) / len(parts) for p in parts])
    if mean is None:
        mean = (x.astype(float) * weights[:, None]).sum(0)
        scale = np.sqrt(((x - mean) ** 2 * weights[:, None]).sum(0)).clip(1e-6)
    z = ((x - mean) / scale).clip(-8, 8).astype(np.float32)
    return (torch.as_tensor(z, device=device), torch.as_tensor(np.concatenate([p["y"] for p in parts]), device=device),
            torch.as_tensor(weights, device=device), mean, scale)


def loss_rows(logits, targets):
    return (nn.functional.binary_cross_entropy_with_logits(logits[:, 0], targets[:, 0], reduction="none")
            + .25 * (logits[:, 1:].sigmoid() - targets[:, 1:]).square().sum(1))


def portable(net, mean, scale, width):
    layers = [layer for layer in net if isinstance(layer, nn.Linear)]
    hidden, output = (layers[0], layers[1]) if width else (None, layers[0])
    def matrix(tensor):
        return tuple(tuple(float(v) for v in row) for row in tensor.detach().cpu().tolist())
    def vector(tensor):
        return tuple(float(v) for v in tensor.detach().cpu().tolist())
    return AcquisitionModel(tuple(float(v) for v in mean), tuple(float(v) for v in scale), matrix(hidden.weight) if hidden else (),
        vector(hidden.bias) if hidden else (), matrix(output.weight), vector(output.bias))


def predictions(model, part):
    started = time.perf_counter()
    values = [model.predict(tuple(float(v) for v in row)) for row in part["x"]]
    duration = time.perf_counter() - started
    records = [dict(r, predictions=values[r["offset"]:r["offset"] + r["count"]]) for r in part["records"]]
    return records, duration


def metrics(rows, indexes):
    count = len(rows)
    retained = [r["lengths"][i] for r, i in zip(rows, indexes)]
    early = [i < r["count"] - 1 for r, i in zip(rows, indexes)]
    errors = [e and not r["complete"][i] for r, i, e in zip(rows, indexes, early)]
    return dict(queries=count, early_stops=sum(early), false_early_stops=sum(errors),
        conditional_error=sum(errors) / sum(early) if any(early) else None,
        mean_retained=float(np.mean(retained)), mean_pool=float(np.mean([r["lengths"][-1] for r in rows])),
        retained_fraction=sum(retained) / sum(r["lengths"][-1] for r in rows),
        mean_pool_gain_recall=float(np.mean([r["gains"][i] / r["pool_gain"] if r["pool_gain"] else 0
                                            for r, i in zip(rows, indexes)])),
        mean_global_judged_recall=float(np.mean([r["positive_counts"][i] / max(1, r["total_relevant"])
                                                for r, i in zip(rows, indexes)])),
        complete_pool_queries=sum(r["complete"][i] for r, i in zip(rows, indexes)),
        no_positive_pool_queries=sum(r["pool_gain"] == 0 for r in rows))


def train(work, out):
    protocol = json.loads((out / "protocol.json").read_text())
    if any(sha(ROOT / p) != value for p, value in protocol["sources"].items()):
        raise ValueError("source changed after registration")
    manifest_path = ROOT / "evidence/controller-v2/data-manifest.json"
    if sha(manifest_path) != protocol["data_manifest_sha256"]:
        raise ValueError("data manifest changed")
    with (out / "started.json").open("x", encoding="utf-8") as stream:
        json.dump({"started": time.time(), "protocol_sha256": sha(out / "protocol.json")}, stream)
    if not torch.cuda.is_available():
        raise RuntimeError("registered training requires CUDA")
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    manifest = json.loads(manifest_path.read_text())
    started = time.perf_counter()
    training = [load_partition(work, n, manifest) for n in protocol["train"]]
    tuning = [load_partition(work, n, manifest) for n in protocol["tune"]]
    tx, ty, tw, mean, scale = combine(training, "cuda")
    vx, vy, vw, _, _ = combine(tuning, "cuda", mean, scale)
    samples, trials, portable_models = [], [], {}
    stop = threading.Event()
    monitor = threading.Thread(target=telemetry, args=(stop, samples), daemon=True)
    monitor.start()
    try:
        for width in protocol["architectures"]:
            for seed in protocol["seeds"]:
                begin = time.perf_counter()
                torch.manual_seed(seed)
                net = nn.Sequential(nn.Linear(33, width), nn.Tanh(), nn.Linear(width, 3)) if width else nn.Sequential(nn.Linear(33, 3))
                net = net.cuda()
                optimizer = torch.optim.AdamW(net.parameters(), lr=protocol["learning_rate"], weight_decay=protocol["weight_decay"])
                history, best, best_state, best_epoch = [], float("inf"), None, 0
                for epoch in range(protocol["epochs"] + 1):
                    if epoch:
                        net.train()
                        permutation = torch.randperm(len(tx), device="cuda")
                        for start in range(0, len(tx), protocol["batch_size"]):
                            indices = permutation[start:start + protocol["batch_size"]]
                            optimizer.zero_grad(set_to_none=True)
                            loss = (loss_rows(net(tx[indices]), ty[indices]) * tw[indices]).sum() * (len(tx) / len(indices))
                            loss.backward()
                            nn.utils.clip_grad_norm_(net.parameters(), 5.)
                            optimizer.step()
                    net.eval()
                    with torch.inference_mode():
                        tune_loss = float((loss_rows(net(vx), vy) * vw).sum())
                    history.append(tune_loss)
                    if tune_loss < best:
                        best, best_epoch = tune_loss, epoch
                        best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
                net.load_state_dict(best_state)
                model = portable(net, mean, scale, width)
                name = f"{'mlp32' if width else 'linear'}-seed{seed}"
                model.save(out / (name + ".json"))
                with torch.inference_mode():
                    reference = net(vx[:64]).sigmoid().cpu().numpy()
                source_features = np.concatenate([p["x"] for p in tuning])[:64]
                actual = np.asarray([model.predict(tuple(float(v) for v in row)) for row in source_features])
                parity_error = float(np.abs(reference - actual).max())
                if parity_error > 1e-5:
                    raise RuntimeError("portable CPU / CUDA parity failed")
                row = dict(name=name, width=width, seed=seed, parameters=sum(p.numel() for p in net.parameters()),
                    selected_epoch=best_epoch, tune_loss=best, history=history, seconds=time.perf_counter() - begin,
                    model_revision=model.revision, file_sha256=sha(out / (name + ".json")),
                    portable_cuda_max_error=parity_error)
                trials.append(row)
                portable_models[name] = model
                print(json.dumps({k: row[k] for k in ("name", "selected_epoch", "tune_loss", "seconds")}), flush=True)
                del net, optimizer, best_state
        selected = min(trials, key=lambda r: (r["tune_loss"], r["name"]))["name"]
        save(out / "selection.json", dict(selected=selected, trials=trials, frozen_before_calibration=True))
    finally:
        stop.set()
        monitor.join(timeout=15)
    train_seconds = time.perf_counter() - started
    model = portable_models[selected]
    policies = [StopPolicy(model.revision, threshold) for threshold in protocol["thresholds"]]
    scopes = [n.rsplit("-", 1)[0] for n in protocol["calibration"]]
    family = tuple((scope, p.revision) for scope in scopes for p in policies)
    calibration, reports, choices, timing = [], [], {}, []
    for name in protocol["calibration"]:
        part = load_partition(work, name, manifest)
        rows, duration = predictions(model, part)
        calibration.extend(rows)
        timing.append(dict(partition=name, seconds=duration, prefixes=len(part["x"])))
        scope = rows[0]["dataset"]
        trajectories = tuple(Trajectory(r["query_id"], tuple(p[0] for p in r["predictions"]), tuple(r["complete"])) for r in rows)
        qualified = []
        for policy in policies:
            risk = assess_policy(policy, trajectories, scope=scope, family=family, alpha=protocol["alpha"])
            indexes = [policy.choose(t.scores) for t in trajectories]
            measured = metrics(rows, indexes)
            reports.append(dict(scope=scope, threshold=policy.threshold, risk=asdict(risk), metrics=measured))
            if risk.permits(policy, scope=scope, maximum_error=protocol["maximum_error"]):
                qualified.append((measured["mean_retained"], policy.threshold))
        choices[scope] = min(qualified)[1] if qualified else None
    save(out / "calibration.json", dict(family=family, reports=reports, choices=choices,
        independent_final_qualification=False, reason="Reused development/calibration collections; descriptive risk audit only."))
    tests, summary = [], []
    for name in protocol["test"]:
        part = load_partition(work, name, manifest)
        rows, duration = predictions(model, part)
        tests.extend(rows)
        timing.append(dict(partition=name, seconds=duration, prefixes=len(part["x"])))
        scope = rows[0]["dataset"]
        for method in protocol["controls"]:
            if method.startswith("fixed-"):
                limit = int(method.split("-")[1])
                indexes = [next((i for i, k in enumerate(r["lengths"]) if k >= min(limit, r["lengths"][-1])), r["count"] - 1) for r in rows]
            elif method == "full-pool":
                indexes = [r["count"] - 1 for r in rows]
            else:
                threshold = .9 if method == "raw-selected-.90" else choices.get(scope)
                policy = StopPolicy(model.revision, threshold) if threshold is not None else None
                indexes = [policy.choose(tuple(p[0] for p in r["predictions"])) if policy else r["count"] - 1 for r in rows]
            summary.append(dict(dataset=scope, method=method, **metrics(rows, indexes)))
    payload = dict(calibration=calibration, test=tests)
    raw = json.dumps(payload, separators=(",", ":"), sort_keys=True, allow_nan=False).encode()
    (out / "trajectories.json.gz").write_bytes(gzip.compress(raw, mtime=0))
    report = dict(schema=1, status="completed_regression_experiment", python=platform.python_version(),
        numpy=np.__version__, torch=torch.__version__, cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(0),
        protocol_sha256=sha(out / "protocol.json"), selected=selected, data_sources=[p["source"] for p in training + tuning],
        train_queries=sum(len(p["records"]) for p in training), tune_queries=sum(len(p["records"]) for p in tuning),
        train_seconds=train_seconds, gpu_samples=samples, max_allocated_bytes=torch.cuda.max_memory_allocated(),
        records_sha256=sha(out / "trajectories.json.gz"), calibration_sha256=sha(out / "calibration.json"),
        selection_sha256=sha(out / "selection.json"), timing=timing, summary=summary,
        total_seconds=time.perf_counter() - started,
        qualification="No production promotion. Reused public development sets, incomplete judgments, no answer-generation targets, and independent sampling unestablished.")
    save(out / "manifest.json", report)
    print(json.dumps({"selected": selected, "choices": choices, "train_queries": report["train_queries"],
                      "test_queries": len(tests), "seconds": report["total_seconds"]}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("register", "train"))
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--work", type=Path, default=ROOT.parent)
    args = parser.parse_args()
    if args.command == "register":
        register(args.out)
    else:
        train(args.work, args.out)

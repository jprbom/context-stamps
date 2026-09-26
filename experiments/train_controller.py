"""Reuse verified local BEIR caches, train a bounded ranker, retain all outcomes.

Copyright (c) 2026 Prashant Jagtap. MIT License.
No downloads, remote execution, API calls or raw text publication.
"""

import argparse
import hashlib
import json
import math
import os
import platform
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))
from run_hybrid_retrieval import BM25, labels, metric, read, save, sha  # noqa: E402

OUT = ROOT / "evidence/controller-v1"
NAMES = ("scifact", "nfcorpus", "arguana", "scidocs")


def normalized(text):
    return " ".join(text.casefold().split())


def bucket(text):
    return int(hashlib.sha256(normalized(text).encode()).hexdigest()[:8], 16) % 100


def bootstrap(a, b):
    delta = np.asarray(a) - np.asarray(b)
    rng = np.random.default_rng(20260926)
    samples = delta[rng.integers(0, len(delta), (5000, len(delta)))].mean(axis=1)
    return dict(mean=float(delta.mean()), lower=float(np.percentile(samples, 2.5)),
                upper=float(np.percentile(samples, 97.5)), n=len(delta))


def prepare(work, destination):
    protocol = read(OUT / "protocol.json")
    prior = {x["dataset"]: x for x in read(ROOT / "evidence/spherical-public-v1/manifest.json")["datasets"]}
    hybrid = {x["dataset"]: x for x in read(ROOT / "evidence/hybrid-retrieval-v1/manifest.json")["datasets"]}
    cache_dirs = [work / "scifact-cache", work / "replication-cache", work / "public-validation/cache"]
    # Hash verification also prevents silently using a different encoder/cache ordering.
    paths = {}
    for directory in cache_dirs:
        for path in directory.glob("*-*.npy"):
            paths[sha(path)] = path
    raw = {}
    for name in NAMES:
        source = work / ("public-validation/scidocs" if name == "scidocs" else "replication-data/" + name)
        docs = [json.loads(line) for line in (source / "corpus.jsonl").read_text(encoding="utf-8").splitlines()]
        queries = {row["_id"]: row["text"] for row in map(json.loads,
                   (source / "queries.jsonl").read_text(encoding="utf-8").splitlines())}
        files = {key: sha(source / key) for key in hybrid[name]["data_sha256"]}
        if files != hybrid[name]["data_sha256"]:
            raise ValueError("public corpus/queries/test labels changed: " + name)
        judgments = {split: labels(source / ("qrels/" + split + ".tsv"))
                     for split in ("train", "dev", "test") if (source / ("qrels/" + split + ".tsv")).exists()}
        for split in judgments:
            files["qrels/" + split + ".tsv"] = sha(source / ("qrels/" + split + ".tsv"))
        metadata = prior.get(name, hybrid[name])
        qids = metadata.get("query_ids", sorted(judgments["test"]))
        dpath = paths[metadata["embedding_sha256"]["documents"]]
        qpath = paths[metadata["embedding_sha256"]["queries"]]
        D, Q = np.load(dpath, allow_pickle=False), np.load(qpath, allow_pickle=False)
        if D.shape != (len(docs), 384) or Q.shape != (len(qids), 384):
            raise ValueError("cache shape mismatch")
        raw[name] = dict(source=source, docs=docs, queries=queries, judgments=judgments,
                         D=D, embeddings=dict(zip(qids, Q)), files=files,
                         inherited_embedding_sha256=metadata["embedding_sha256"])
    # Across domains as well as within official splits. Normalize text before grouping.
    test_texts = {normalized(r["queries"][qid]) for r in raw.values() for qid in r["judgments"]["test"]}
    held_texts = set(test_texts)
    assignments, excluded = {}, []
    for name, r in raw.items():
        parts = {"train": [], "tune": [], "calibration": [], "test": sorted(r["judgments"]["test"])}
        for split in ("train", "dev"):
            for qid in sorted(r["judgments"].get(split, {})):
                text = r["queries"][qid]
                if normalized(text) in test_texts:
                    excluded.append(dict(dataset=name, query_id=qid, reason="normalized_test_overlap"))
                    continue
                b = bucket(text)
                target = ("train" if b < 70 else "tune" if b < 85 else "calibration") if name == "scifact" else (
                    "train" if split == "train" else "tune" if b < 50 else "calibration")
                parts[target].append(qid)
                if target != "train":
                    held_texts.add(normalized(text))
        assignments[name] = parts
    for name, parts in assignments.items():
        keep = []
        for qid in parts["train"]:
            if normalized(raw[name]["queries"][qid]) in held_texts:
                excluded.append(dict(dataset=name, query_id=qid, reason="normalized_development_overlap"))
            else:
                keep.append(qid)
        parts["train"] = keep
    # Fail on a development cross-partition collision instead of claiming disjointness.
    split_texts = {s: {normalized(raw[n]["queries"][q]) for n, p in assignments.items() for q in p[s]}
                   for s in ("train", "tune", "calibration", "test")}
    for a, b in (("train", "tune"), ("train", "calibration"), ("tune", "calibration"), ("tune", "test"), ("calibration", "test")):
        if split_texts[a] & split_texts[b]:
            raise ValueError("normalized query text crosses partitions: " + a + "/" + b)
    save(OUT / "split-manifest.json", dict(assignments=assignments, exclusions=excluded,
         data_sha256={n: r["files"] for n, r in raw.items()},
         limitations="Shared document corpora; exact normalized query separation only, not semantic deduplication. All tests historically inspected."))
    destination.mkdir(parents=True, exist_ok=True)
    encoder = tokenizer = None
    manifest = []
    for name, r in raw.items():
        print("Preparing", name, {s: len(v) for s, v in assignments[name].items()}, flush=True)
        all_ids = sorted({q for ids in assignments[name].values() for q in ids})
        missing = [q for q in all_ids if q not in r["embeddings"]]
        if missing:
            import torch
            from transformers import AutoModel, AutoTokenizer
            if encoder is None:
                tokenizer = AutoTokenizer.from_pretrained(protocol["encoder"], revision=protocol["revision"],
                                                           local_files_only=True, trust_remote_code=False)
                encoder = AutoModel.from_pretrained(protocol["encoder"], revision=protocol["revision"],
                              local_files_only=True, trust_remote_code=False, use_safetensors=True).cuda().eval()
            start = time.perf_counter()
            for offset in range(0, len(missing), 64):
                ids = missing[offset:offset + 64]
                inputs = tokenizer([r["queries"][q] for q in ids], padding=True, truncation=True,
                                   max_length=256, return_tensors="pt").to("cuda")
                with torch.inference_mode():
                    hidden = encoder(**inputs).last_hidden_state.float()
                    mask = inputs["attention_mask"].unsqueeze(-1)
                    pooled = (hidden * mask).sum(1) / mask.sum(1).clamp_min(1)
                    vectors = torch.nn.functional.normalize(pooled, dim=1).cpu().numpy()
                r["embeddings"].update(zip(ids, vectors))
            print("Encoded", len(missing), "new training/development queries in", round(time.perf_counter() - start, 2), "seconds", flush=True)
        ids = [d["_id"] for d in r["docs"]]
        positions = {key: i for i, key in enumerate(ids)}
        docs = [(d.get("title", "") + " " + d["text"]).strip() for d in r["docs"]]
        index = BM25(docs)
        np.save(destination / (name + "-D.npy"), r["D"], allow_pickle=False)
        all_judgments = {q: rel for split in r["judgments"].values() for q, rel in split.items()}
        for split, qids in assignments[name].items():
            if not qids:
                continue
            Q = np.asarray([r["embeddings"][q] for q in qids], dtype=np.float32)
            candidates = np.zeros((len(qids), 128), dtype=np.int32)
            F = np.zeros((len(qids), 128, 6), dtype=np.float32)
            Y = np.zeros((len(qids), 128), dtype=np.float32)
            mask = np.zeros((len(qids), 128), dtype=bool)
            baseline, timings = [], []
            for i, qid in enumerate(qids):
                start = time.perf_counter()
                ds = (r["D"] @ Q[i]).astype(np.float64)
                ls = index.score(r["queries"][qid])
                eligible = np.ones(len(ids), dtype=bool)
                if qid in positions:
                    eligible[positions[qid]] = False
                dz = (ds - ds[eligible].mean()) / max(ds[eligible].std(), 1e-9)
                lz = (ls - ls[eligible].mean()) / max(ls[eligible].std(), 1e-9)
                hs = .75 * dz + .25 * lz
                orders = []
                for scores in (ds, hs, ls):
                    scores = scores.copy()
                    scores[~eligible] = -np.inf
                    orders.append(np.argsort(-scores, kind="stable"))
                chosen = sorted(set(orders[0][:48]) | set(orders[1][:48]) | set(orders[2][:32]))
                count = len(chosen)
                candidates[i, :count] = chosen
                mask[i, :count] = True
                ranks = [np.empty(len(ids), dtype=np.int32) for _ in range(2)]
                ranks[0][orders[0]] = np.arange(len(ids))
                ranks[1][orders[2]] = np.arange(len(ids))
                F[i, :count] = np.stack((ds[chosen], dz[chosen], hs[chosen], lz[chosen],
                               1 / (1 + ranks[0][chosen]), 1 / (1 + ranks[1][chosen])), axis=1)
                rel = all_judgments[qid]
                Y[i, :count] = [max(0, rel.get(ids[c], 0)) for c in chosen]
                timings.append((time.perf_counter() - start) * 1000)
                baseline.append(dict(query_id=qid, dataset=name, split=split,
                    metrics={m: metric(o, rel, ids) for m, o in zip(("dense", "hybrid", "bm25"), orders)},
                    ranked_ids={m: [ids[c] for c in o[:10]] for m, o in zip(("dense", "hybrid", "bm25"), orders)},
                    candidate_recall=float(np.count_nonzero(Y[i]) / max(1, sum(v > 0 for v in rel.values()))),
                    ideal=sum(v / math.log2(j + 2) for j, v in enumerate(sorted((v for v in rel.values() if v > 0), reverse=True)[:10])),
                    total_relevant=sum(v > 0 for v in rel.values())))
            stem = name + "-" + split
            np.savez(destination / (stem + ".npz"), Q=Q, candidates=candidates, F=F, Y=Y, mask=mask)
            save(destination / (stem + ".json"), dict(records=baseline, document_ids=ids, query_ids=qids))
            manifest.append(dict(dataset=name, split=split, count=len(qids),
                no_candidate_positive=int(np.sum(Y.sum(1) == 0)),
                candidate_recall=float(np.mean([x["candidate_recall"] for x in baseline])),
                cached_retrieval_ms_p50=float(np.median(timings)), cached_retrieval_ms_p95=float(np.percentile(timings, 95)),
                array_sha256=sha(destination / (stem + ".npz"))))
    save(OUT / "data-preparation.json", dict(partitions=manifest,
        encoder=protocol["encoder"], revision=protocol["revision"],
        inherited_embeddings={n: r["inherited_embedding_sha256"] for n, r in raw.items()},
        protocol_sha256=sha(OUT / "protocol.json"), split_sha256=sha(OUT / "split-manifest.json")))
    return manifest


def load_partitions(destination, split):
    rows = []
    for name in NAMES:
        path = destination / (name + "-" + split + ".npz")
        if not path.exists():
            continue
        arrays = dict(np.load(path, allow_pickle=False))
        arrays["D"] = np.load(destination / (name + "-D.npy"), allow_pickle=False)
        arrays["meta"] = read(path.with_suffix(".json"))
        arrays["name"] = name
        rows.append(arrays)
    return rows


def batch(data, indexes, device):
    import torch
    return tuple(torch.as_tensor(value, device=device) for value in
        (data["Q"][indexes], data["D"][data["candidates"][indexes]], data["F"][indexes], data["mask"][indexes]))


def scored_records(data, scores, method):
    result = []
    for i, original in enumerate(data["meta"]["records"]):
        order = np.argsort(-scores[i], kind="stable")[:10]
        gains = data["Y"][i, order]
        ndcg = float(sum(v / math.log2(j + 2) for j, v in enumerate(gains)) / max(original["ideal"], 1e-12))
        result.append(dict(dataset=data["name"], query_id=original["query_id"], method=method,
            ndcg10=ndcg, recall10=float(np.count_nonzero(gains) / max(1, original["total_relevant"])),
            mrr10=next((1 / (j + 1) for j, v in enumerate(gains) if v > 0), 0),
            ranked_ids=[data["meta"]["document_ids"][data["candidates"][i, j]] for j in order],
            ranked_gains=[float(v) for v in gains], ideal=original["ideal"],
            total_relevant=original["total_relevant"]))
    return result


def evaluate(model, partitions, method, device="cuda", steps=None):
    import torch
    model.eval()
    records = []
    with torch.inference_mode():
        for data in partitions:
            scores = []
            for offset in range(0, len(data["Q"]), 32):
                indexes = np.arange(offset, min(offset + 32, len(data["Q"])))
                scores.append(model(*batch(data, indexes, device), steps=steps).float().cpu().numpy())
            records.extend(scored_records(data, np.concatenate(scores), method))
    grouped = {name: float(np.mean([r["ndcg10"] for r in records if r["dataset"] == name]))
               for name in sorted({r["dataset"] for r in records})}
    return records, float(np.mean(list(grouped.values())))


def train(destination, output):
    import torch
    from torch.nn import functional as F

    from context_stamps.neural_controller import RecurrentEvidenceRanker, export_ranker, load_ranker

    protocol = read(OUT / "protocol.json")
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for this recorded RTX experiment")
    output.mkdir(parents=True, exist_ok=True)
    training, tuning = load_partitions(destination, "train"), load_partitions(destination, "tune")
    trials = []
    started = time.perf_counter()
    for attention in (False, True):
        for seed in protocol["seeds"]:
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            rng = np.random.default_rng(seed)
            model = RecurrentEvidenceRanker(attention=attention).cuda()
            optimizer = torch.optim.AdamW(model.parameters(), lr=protocol["learning_rate"], weight_decay=.01)
            trial = ("recurrent" if attention else "mlp") + "-" + str(seed)
            checkpoint = output / (trial + ".safetensors")
            _, best = evaluate(model, tuning, trial)
            export_ranker(model, checkpoint)
            history = [dict(epoch=0, tune_ndcg10=best)]
            best_epoch = 0
            torch.cuda.reset_peak_memory_stats()
            start = time.perf_counter()
            for epoch in range(1, protocol["epochs"] + 1):
                model.train()
                batches = []
                for d, data in enumerate(training):
                    eligible = np.where(data["Y"].sum(1) > 0)[0]
                    rng.shuffle(eligible)
                    batches.extend((d, eligible[i:i + protocol["batch_size"]])
                                   for i in range(0, len(eligible), protocol["batch_size"]))
                rng.shuffle(batches)
                losses = []
                for d, indexes in batches:
                    data = training[d]
                    inputs = batch(data, indexes, "cuda")
                    targets = torch.as_tensor(data["Y"][indexes], device="cuda")
                    targets = targets / targets.sum(1, keepdim=True).clamp_min(1)
                    optimizer.zero_grad(set_to_none=True)
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        scores = model(*inputs).float()
                        logprobs = F.log_softmax(scores, dim=-1)
                        teacher = F.softmax(inputs[2][..., 2].masked_fill(~inputs[3], -1e4), dim=-1)
                        loss = -(targets * logprobs).sum(1).mean() + .1 * F.kl_div(logprobs, teacher, reduction="batchmean")
                    if not torch.isfinite(loss):
                        raise FloatingPointError("non-finite loss")
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
                    optimizer.step()
                    losses.append(loss.item())
                _, value = evaluate(model, tuning, trial)
                history.append(dict(epoch=epoch, loss=float(np.mean(losses)), tune_ndcg10=value))
                if value > best:
                    best, best_epoch = value, epoch
                    export_ranker(model, checkpoint)
                print(trial, "epoch", epoch, "loss", round(float(np.mean(losses)), 4), "tune", round(value, 5), flush=True)
            trials.append(dict(trial=trial, seed=seed, attention=attention, selected_epoch=best_epoch,
                tune_ndcg10=best, seconds=time.perf_counter() - start,
                parameters=sum(p.numel() for p in model.parameters()),
                peak_allocated_bytes=torch.cuda.max_memory_allocated(), checkpoint_sha256=sha(checkpoint), history=history))
            save(OUT / "training.json", trials)
    selected = max(trials, key=lambda row: row["tune_ndcg10"])["trial"]
    model = load_ranker(output / (selected + ".safetensors")).cuda()
    # Commit selection and independent calibration decisions before accessing test arrays.
    calibration = load_partitions(destination, "calibration")
    cal_records, _ = evaluate(model, calibration, "selected")
    decisions = {}
    for data in calibration:
        name = data["name"]
        values = [r["ndcg10"] for r in cal_records if r["dataset"] == name]
        dense = [r["metrics"]["dense"]["ndcg10"] for r in data["meta"]["records"]]
        hybrid = [r["metrics"]["hybrid"]["ndcg10"] for r in data["meta"]["records"]]
        nd, nh, hd = bootstrap(values, dense), bootstrap(values, hybrid), bootstrap(hybrid, dense)
        route = "dense"
        if len(values) >= 50:
            if nd["lower"] > 0 and nh["lower"] > 0:
                route = "selected"
            elif hd["lower"] > 0:
                route = "hybrid"
        decisions[name] = dict(route=route, neural_minus_dense=nd, neural_minus_hybrid=nh, hybrid_minus_dense=hd)
    save(OUT / "selection.json", dict(selected=selected, per_scope=decisions, uncalibrated="dense",
         calibration_records=cal_records, protocol_sha256=sha(OUT / "protocol.json")))
    print("Frozen selection", selected, {k: v["route"] for k, v in decisions.items()}, flush=True)
    export_ranker(model, output / "selected-fp32.safetensors")
    export_ranker(model, output / "selected-int8.safetensors", int8=True)
    test = load_partitions(destination, "test")
    results = []
    for trial in trials:
        candidate = load_ranker(output / (trial["trial"] + ".safetensors")).cuda()
        records, _ = evaluate(candidate, test, trial["trial"])
        results.extend(records)
    for steps in (1, 4):
        records, _ = evaluate(model, test, "selected-steps" + str(steps), steps=steps)
        results.extend(records)
    quantized = load_ranker(output / "selected-int8.safetensors").cuda()
    records, _ = evaluate(quantized, test, "selected-int8-storage")
    results.extend(records)
    for data in test:
        for row in data["meta"]["records"]:
            for method, values in row["metrics"].items():
                results.append(dict(dataset=data["name"], query_id=row["query_id"], method=method,
                                    ranked_ids=row["ranked_ids"][method], **values))
        route = decisions.get(data["name"], {"route": "dense"})["route"]
        method = selected if route == "selected" else route
        results.extend({**row, "method": "gated-policy", "route": method} for row in list(results)
                       if row["dataset"] == data["name"] and row["method"] == method)
    save(OUT / "results.json", results)
    summary = {}
    for name in NAMES:
        methods = {}
        dense = [r["ndcg10"] for r in results if r["dataset"] == name and r["method"] == "dense"]
        for method in sorted({r["method"] for r in results}):
            rows = [r for r in results if r["dataset"] == name and r["method"] == method]
            methods[method] = {metric: float(np.mean([r[metric] for r in rows]))
                               for metric in ("ndcg10", "recall10", "mrr10")}
            methods[method]["delta_vs_dense"] = bootstrap([r["ndcg10"] for r in rows], dense)
        summary[name] = methods
    save(OUT / "summary.json", summary)
    sample = batch(test[0], np.arange(1), "cuda")
    timing = {}
    with torch.inference_mode():
        for precision in (torch.float32, torch.bfloat16):
            times = []
            for repeat in range(120):
                torch.cuda.synchronize()
                start = time.perf_counter()
                with torch.autocast("cuda", dtype=precision, enabled=precision != torch.float32):
                    scores = model(*sample)
                torch.cuda.synchronize()
                if repeat >= 20:
                    times.append((time.perf_counter() - start) * 1000)
            timing[str(precision)] = dict(p50_ms=float(np.median(times)), p95_ms=float(np.percentile(times, 95)))
        maximum_error = float((model(*sample) - quantized(*sample)).abs().max().item())
    save(OUT / "runtime.json", dict(gpu=torch.cuda.get_device_name(), torch=torch.__version__,
        cuda=torch.version.cuda, python=platform.python_version(), numpy=np.__version__,
        total_train_evaluate_seconds=time.perf_counter() - started, threads=torch.get_num_threads(),
        selected=selected, timing=timing, batch=1, candidates=128,
        checkpoint_bytes={key: (output / ("selected-" + key + ".safetensors")).stat().st_size for key in ("fp32", "int8")},
        checkpoint_sha256={key: sha(output / ("selected-" + key + ".safetensors")) for key in ("fp32", "int8")},
        max_logit_error_first_test_query=maximum_error,
        limits="Warm forward only, excluding encoding/retrieval/transfers. Int8 storage is reconstructed FP32; no resident memory or speed reduction asserted."))
    source_paths = ["experiments/train_controller.py", "experiments/run_hybrid_retrieval.py", "context_stamps/neural_controller.py"]
    save(OUT / "manifest.json", dict(protocol_sha256=sha(OUT / "protocol.json"),
        source_sha256={p: sha(ROOT / p) for p in source_paths}, selected=selected,
        query_counts={d["name"]: len(d["Q"]) for d in test},
        methods=sorted(Counter(r["method"] for r in results))))
    save(OUT / "checksums.json", {p.name: sha(p) for p in OUT.glob("*.json") if p.name != "checksums.json"})
    print(json.dumps({n: {m: round(v["ndcg10"], 5) for m, v in values.items()} for n, values in summary.items()}, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work", type=Path, required=True, help="Existing local cache/data directory")
    parser.add_argument("--phase", choices=("prepare", "train", "all"), default="all")
    args = parser.parse_args()
    # Hugging Face must use the existing cache. Model/config custom code is disabled.
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    destination = args.work / "controller-v1-data"
    if args.phase in ("prepare", "all"):
        prepare(args.work, destination)
    if args.phase in ("train", "all"):
        train(destination, args.work / "controller-v1-models")

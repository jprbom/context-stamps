"""Train and evaluate a small matching-aware adapter for 32-byte PQ.

The raw CodeSearchNet rows, embeddings, and adapter weights stay outside Git.
Copyright (c) 2026 Prashant Jagtap. MIT License.
"""

import argparse
import json
import statistics
import time
from collections import Counter, defaultdict
from pathlib import Path

import faiss
import numpy as np
import torch
from codesearchnet_quantization import MODEL, REVISION, digest, file_sha, metric
from sentence_transformers import SentenceTransformer

SEED = 20261002
RANK = 16
EPOCHS = 6
LR = 0.002
TEMPERATURE = 0.07
DRIFT_PENALTY = 0.1


def load_vectors(cache, split):
    data = np.load(cache / f"{split}-vectors.npz")
    return np.ascontiguousarray(data["docs"]), np.ascontiguousarray(data["queries"])


def pq_codebook(docs):
    faiss.omp_set_num_threads(4)
    index = faiss.IndexPQ(384, 32, 8, faiss.METRIC_INNER_PRODUCT)
    index.pq.cp.seed = SEED
    index.pq.cp.min_points_per_centroid = 16
    index.train(docs)
    return index


def groups(cache):
    records = json.loads((cache / "train.json").read_text(encoding="utf-8"))
    grouped = defaultdict(list)
    for i, row in enumerate(records):
        grouped[row["repo"]].append(i)
    output = []
    for name, ids in grouped.items():
        unique = Counter(" ".join(records[i]["query"].casefold().split()) for i in ids)
        ids = [i for i in ids if unique[" ".join(records[i]["query"].casefold().split())] == 1]
        if len(ids) < 5:
            continue
        ids = sorted(ids, key=lambda i: digest(f"{SEED}:{records[i]['id']}"))[:64]
        output.append((name, np.asarray(ids, dtype=np.int64)))
    return sorted(output, key=lambda pair: digest(f"{SEED}:{pair[0]}"))


def adapt(q, u, v):
    adjusted = q + (q @ u) @ v.T
    return torch.nn.functional.normalize(adjusted, dim=-1)


def fit_adapter(queries, docs, selected, label, out):
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    q = torch.from_numpy(queries).to(device)
    d = torch.from_numpy(docs).to(device)
    u = torch.nn.Parameter(torch.randn(384, RANK, device=device) * 0.001)
    v = torch.nn.Parameter(torch.randn(384, RANK, device=device) * 0.02)
    optimizer = torch.optim.AdamW([u, v], lr=LR, weight_decay=0.01)
    rng = np.random.default_rng(SEED)
    history = []
    start = time.perf_counter()
    for epoch in range(EPOCHS):
        order = rng.permutation(len(selected))
        losses = []
        for group_id in order:
            _, ids = selected[int(group_id)]
            ids_t = torch.from_numpy(ids).to(device)
            qg = q[ids_t]
            dg = d[ids_t]
            scores = adapt(qg, u, v) @ dg.T / TEMPERATURE
            labels = torch.arange(len(ids), device=device)
            loss = torch.nn.functional.cross_entropy(scores, labels)
            loss = loss + DRIFT_PENALTY * ((adapt(qg, u, v) - qg) ** 2).sum(dim=1).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        history.append(round(statistics.mean(losses), 6))
    np.savez_compressed(out / f"{label}-adapter.npz", u=u.detach().cpu().numpy(), v=v.detach().cpu().numpy())
    return {"groups": len(selected), "pairs": int(sum(len(ids) for _, ids in selected)), "epochs": EPOCHS,
            "loss_by_epoch": history, "fit_seconds": round(time.perf_counter() - start, 3),
            "device": device, "weights_sha256": file_sha(out / f"{label}-adapter.npz")}


def query_adapter(q, weights):
    if weights is None:
        return q
    u, v = weights
    adjusted = q + (q @ u) @ v.T
    return np.ascontiguousarray(adjusted / np.maximum(np.linalg.norm(adjusted, axis=1, keepdims=True), 1e-12))


def evaluate(cache, split, pq_template, adapters):
    cohorts = json.loads((cache / f"{split}.json").read_text(encoding="utf-8"))
    docs, queries = load_vectors(cache, split)
    routes = {"dense": ("dense", None), "pq32": ("pq", None),
              "float_adapter_dense": ("dense", adapters["float"]),
              "float_adapter_pq32": ("pq", adapters["float"]),
              "pq_adapter_dense": ("dense", adapters["pq"]),
              "pq_adapter_pq32": ("pq", adapters["pq"])}
    ranks = {name: [] for name in routes}
    timing = {name: [] for name in routes}
    per_repo = []
    doc_offset = query_offset = 0
    for cohort in cohorts:
        nd, nq = len(cohort["items"]), len(cohort["targets"])
        d = np.ascontiguousarray(docs[doc_offset:doc_offset + nd])
        q = np.ascontiguousarray(queries[query_offset:query_offset + nq])
        dense = faiss.IndexFlatIP(384)
        dense.add(d)
        compact = faiss.clone_index(pq_template)
        compact.reset()
        compact.add(d)
        indexes = {"dense": dense, "pq": compact}
        record = {"repo": cohort["repo"], "candidates": nd, "ranks": {}}
        for name, (kind, weights) in routes.items():
            index = indexes[kind]
            index.search(query_adapter(q, weights), nd)
            samples = []
            for _ in range(11):
                start = time.perf_counter()
                _, order = index.search(query_adapter(q, weights), nd)
                samples.append((time.perf_counter() - start) * 1000 / nq)
            found = [int(np.flatnonzero(order[i] == target)[0]) + 1
                     for i, target in enumerate(cohort["targets"])]
            record["ranks"][name] = found
            ranks[name].extend(found)
            timing[name].append(statistics.median(samples))
        per_repo.append(record)
        doc_offset += nd
        query_offset += nq
    assert doc_offset == len(docs) and query_offset == len(queries)
    return {"summary": {name: metric(values) for name, values in ranks.items()},
            "warm_lookup_median_ms_per_query": {name: round(statistics.median(values), 6)
                                                for name, values in timing.items()}, "rows": per_repo}


def train(cache, out):
    out.mkdir(parents=True, exist_ok=True)
    docs, queries = load_vectors(cache, "train")
    pq_index = pq_codebook(docs)
    pq_index.add(docs)
    reconstructed = np.ascontiguousarray(pq_index.reconstruct_n(0, len(docs)))
    pq_template = faiss.clone_index(pq_index)
    pq_template.reset()
    selected = groups(cache)
    fit = {"float": fit_adapter(queries, docs, selected, "float", cache),
           "pq": fit_adapter(queries, reconstructed, selected, "pq", cache)}
    weights = {name: tuple(np.load(cache / f"{name}-adapter.npz")[key] for key in ("u", "v"))
               for name in ("float", "pq")}
    result = {"task": "derived CodeSearchNet Python repository-local exact-function localization",
              "training": {"rank": RANK, "epochs": EPOCHS, "learning_rate": LR,
                           "temperature": TEMPERATURE, "drift_penalty": DRIFT_PENALTY,
                           "seed": SEED, "optimizer": "AdamW weight_decay=0.01", "routes": fit,
                           "train_vectors_sha256": file_sha(cache / "train-vectors.npz"),
                           "pq_codebook_bytes": len(faiss.serialize_index(pq_template)),
                           "adapter_bytes_each": sum(weights["pq"][i].nbytes for i in (0, 1))},
              "validation": evaluate(cache, "validation", pq_template, weights)}
    result["validation_gate"] = {"pass": result["validation"]["summary"]["pq_adapter_pq32"]["top1"] >=
                                 result["validation"]["summary"]["dense"]["top1"] and
                                 result["validation"]["summary"]["pq_adapter_pq32"]["mrr"] >=
                                 result["validation"]["summary"]["dense"]["mrr"] - 0.01,
                                 "rule": "PQ-adapter top1 >= frozen dense and MRR >= frozen dense - 0.01"}
    (out / "validation.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"summary": result["validation"]["summary"],
                      "gate": result["validation_gate"]}, indent=2))


def embed_test(cache):
    data = json.loads((cache / "matching-test-v1.json").read_text(encoding="utf-8"))
    model = SentenceTransformer(MODEL, revision=REVISION, device="cuda")
    docs = [row["code"] for cohort in data for row in cohort["items"]]
    queries = [cohort["items"][i]["query"] for cohort in data for i in cohort["targets"]]
    d = model.encode(docs, batch_size=128, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False)
    q = model.encode(queries, batch_size=128, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False)
    np.savez_compressed(cache / "matching-test-v1-vectors.npz", docs=d.astype("float32"), queries=q.astype("float32"))
    print(len(docs), len(queries), file_sha(cache / "matching-test-v1-vectors.npz"))


def test(cache, out):
    result = json.loads((out / "validation.json").read_text(encoding="utf-8"))
    if not result["validation_gate"]["pass"]:
        raise ValueError("validation gate failed; new cohort must remain unopened")
    cohort_manifest = json.loads((out / "cohort.json").read_text(encoding="utf-8"))
    if file_sha(cache / "matching-test-v1.json") != cohort_manifest["prepared_sha256"]:
        raise ValueError("new cohort differs from frozen manifest")
    if not (cache / "matching-test-v1-vectors.npz").exists():
        embed_test(cache)
    docs, _ = load_vectors(cache, "train")
    pq_index = pq_codebook(docs)
    weights = {name: tuple(np.load(cache / f"{name}-adapter.npz")[key] for key in ("u", "v"))
               for name in ("float", "pq")}
    result["test"] = evaluate(cache, "matching-test-v1", pq_index, weights)
    result["test_vectors_sha256"] = file_sha(cache / "matching-test-v1-vectors.npz")
    (out / "results.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(result["test"]["summary"], indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("phase", choices=["train", "test"])
    p.add_argument("--cache", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    {"train": lambda: train(a.cache, a.out), "test": lambda: test(a.cache, a.out)}[a.phase]()

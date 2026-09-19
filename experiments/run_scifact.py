"""Real SciFact retrieval comparison; reads licensed local BEIR data, publishes no text."""

import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import faiss
import numpy as np
import psutil
from sentence_transformers import SentenceTransformer

from context_stamps.baselines import bm25, corpus_mean_similarity, mmr_order
from context_stamps.selection import fit_selector, rank_candidates

MODEL = "sentence-transformers/all-MiniLM-L6-v2"
REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def read_qrels(path):
    values = {}
    with path.open(encoding="utf-8") as stream:
        for row in csv.DictReader(stream, delimiter="\t"):
            if int(row["score"]) > 0:
                values.setdefault(row["query-id"], set()).add(row["corpus-id"])
    return values


def metrics(order, positive, ids):
    hits = [int(ids[int(i)] in positive) for i in order[:10]]
    dcg = sum(hit / math.log2(rank + 2) for rank, hit in enumerate(hits))
    ideal = sum(1 / math.log2(rank + 2) for rank in range(min(10, len(positive))))
    return {"ndcg10": dcg / ideal if ideal else 0, "recall10": sum(hits) / len(positive)}


def coverage_order(query, scores, vectors, documents, pool, weight):
    indexes = list(map(int, pool))
    order = rank_candidates(
        query,
        [documents[i] for i in indexes],
        [float(scores[i]) for i in indexes],
        coverage_weight=weight,
        diversity_weight=weight * 0.5,
        pair_similarity=lambda i, j: float(vectors[indexes[i]] @ vectors[indexes[j]]),
    )
    return [indexes[i] for i in order]


def run(root, out, cache):
    out.mkdir(parents=True, exist_ok=True)
    cache.mkdir(parents=True, exist_ok=True)
    corpus = [json.loads(line) for line in (root / "corpus.jsonl").read_text(encoding="utf-8").splitlines()]
    queries = {
        r["_id"]: r["text"]
        for r in map(json.loads, (root / "queries.jsonl").read_text(encoding="utf-8").splitlines())
    }
    train_all, test = read_qrels(root / "qrels/train.tsv"), read_qrels(root / "qrels/test.tsv")
    test_docs = set().union(*test.values())
    # Exclude every official-training query sharing a labeled relevant document with test.
    usable = {q: docs for q, docs in train_all.items() if not docs & test_docs and q not in test}
    # Connected components of shared relevant documents stay in a single split.
    components = []
    for q in sorted(usable):
        matching = [c for c in components if c[1] & usable[q]]
        group_q, group_docs = {q}, set(usable[q])
        for comp in matching:
            group_q |= comp[0]
            group_docs |= comp[1]
            components.remove(comp)
        components.append((group_q, group_docs))
    train_ids, val_ids = [], []
    for qs, docs in components:
        key = hashlib.sha256("|".join(sorted(docs)).encode()).digest()[0]
        (val_ids if key % 4 == 0 else train_ids).extend(sorted(qs))
    train_ids, val_ids, test_ids = sorted(train_ids), sorted(val_ids), sorted(test)
    if not train_ids or not val_ids:
        raise ValueError("insufficient disjoint training and validation groups")
    ids = [r["_id"] for r in corpus]
    docs = [(r["title"] + " " + r["text"]).strip() for r in corpus]
    query_ids = sorted(set(train_ids + val_ids + test_ids))
    query_position = {q: i for i, q in enumerate(query_ids)}
    dataset_hashes = {
        p: digest(root / p) for p in ["corpus.jsonl", "queries.jsonl", "qrels/train.tsv", "qrels/test.tsv"]
    }
    key = hashlib.sha256(
        json.dumps([dataset_hashes, MODEL, REVISION, query_ids], sort_keys=True).encode()
    ).hexdigest()
    begin = time.perf_counter()
    reused_cache = (cache / f"{key}-D.npy").exists() and (cache / f"{key}-Q.npy").exists()
    if reused_cache:
        d = np.load(cache / f"{key}-D.npy", allow_pickle=False)
        q = np.load(cache / f"{key}-Q.npy", allow_pickle=False)
    else:
        model = SentenceTransformer(
            MODEL,
            revision=REVISION,
            device="cpu",
            trust_remote_code=False,
            model_kwargs={"use_safetensors": True},
        )
        print(f"Encoding {len(docs)} documents and {len(query_ids)} queries on CPU", flush=True)
        d = model.encode(docs, batch_size=32, normalize_embeddings=True, show_progress_bar=True).astype(
            "float32"
        )
        q = model.encode(
            [queries[i] for i in query_ids], batch_size=32, normalize_embeddings=True, show_progress_bar=True
        ).astype("float32")
        np.save(cache / f"{key}-D.npy", d)
        np.save(cache / f"{key}-Q.npy", q)
    encode_seconds = time.perf_counter() - begin
    faiss.omp_set_num_threads(1)
    dense = faiss.IndexFlatIP(d.shape[1])
    dense.add(d)
    rng = np.random.default_rng(20260919)
    planes = rng.standard_normal((d.shape[1], 128)).astype("float32")
    dbits = np.packbits(d @ planes >= 0, axis=1)
    qbits = np.packbits(q @ planes >= 0, axis=1)
    binary = faiss.IndexBinaryFlat(128)
    binary.add(dbits)
    dense_scores, dense_orders = dense.search(q, 100)
    _, binary_orders = binary.search(qbits, 100)
    train_examples = []
    id_position = {value: i for i, value in enumerate(ids)}
    for qid in train_ids:
        qi = query_position[qid]
        candidates = set(map(int, dense_orders[qi, :30])) | {id_position[i] for i in usable[qid]}
        train_examples.extend(
            (queries[qid], docs[i], float(q[qi] @ d[i]), int(ids[i] in usable[qid]))
            for i in sorted(candidates)
        )
    trained, traces = {}, {}
    training_start = time.perf_counter()
    for seed in [7, 19, 43]:
        trained[seed], traces[seed] = fit_selector(train_examples, seed=seed, epochs=600)
        trained[seed].save(out / f"selector-seed-{seed}.json")
    training_seconds = time.perf_counter() - training_start
    print(f"Trained on {len(train_ids)} query groups / {len(train_examples)} pairs", flush=True)
    # Freeze coverage weight using validation, never official test qrels.
    validation = []
    for weight in [0.0, 0.1, 0.3]:
        values = []
        for qid in val_ids:
            qi = query_position[qid]
            scores = d @ q[qi]
            order = coverage_order(queries[qid], scores, d, docs, dense_orders[qi], weight)
            values.append(metrics(order, usable[qid], ids)["ndcg10"])
        validation.append({"weight": weight, "ndcg10": float(np.mean(values))})
    chosen = max(validation, key=lambda r: r["ndcg10"])["weight"]
    records = []
    prior = corpus_mean_similarity(d)
    for index, qid in enumerate(test_ids):
        qi = query_position[qid]
        scores = d @ q[qi]
        pool = dense_orders[qi]
        start = time.perf_counter()
        lex = bm25(queries[qid], docs)
        lexical_ms = (time.perf_counter() - start) * 1000
        orderings = {
            "bm25_reference": sorted(range(len(ids)), key=lambda i: -lex[i])[:10],
            "faiss_dense": pool[:10].tolist(),
            "faiss_binary128": binary_orders[qi, :10].tolist(),
            "binary100_dense_rerank": sorted(map(int, binary_orders[qi]), key=lambda i: -scores[i])[:10],
        }
        # Explicit raw diallel cancellation diagnostic: GCA terms cancel algebraically.
        gca = prior
        sca = scores - scores.mean() - gca + gca.mean()
        raw = 0.6 * scores + 0.2 * gca + 0.2 * sca
        orderings["raw_diallel_control"] = np.argsort(-raw, kind="stable")[:10].tolist()
        start = time.perf_counter()
        orderings["classical_mmr"] = [int(pool[i]) for i in mmr_order(scores[pool], d[pool], limit=10)]
        mmr_ms = (time.perf_counter() - start) * 1000
        start = time.perf_counter()
        orderings["coverage_diversity"] = coverage_order(queries[qid], scores, d, docs, pool, chosen)
        coverage_ms = (time.perf_counter() - start) * 1000
        for seed, model in trained.items():
            orderings[f"trained_linear_{seed}"] = sorted(
                map(int, pool), key=lambda i: -model.score(queries[qid], docs[i], float(scores[i]))
            )[:10]
        for name, order in orderings.items():
            # Identical byte budget and whole-document policy for a secondary packing measure.
            used, packed = 0, []
            for i in order:
                cost = len((json.dumps({"source": ids[i]}) + "\n" + docs[i]).encode()) + (2 if packed else 0)
                if used + cost <= 8192:
                    used += cost
                    packed.append(i)
            redundant = [float(d[i] @ d[j]) >= 0.9 for n, i in enumerate(order) for j in order[n + 1 :]]
            records.append(
                {
                    "query_id": qid,
                    "method": name,
                    **metrics(order, test[qid], ids),
                    "packed_recall": metrics(packed, test[qid], ids)["recall10"],
                    "packed_bytes": used,
                    "duplicate_pair_rate": float(np.mean(redundant)) if redundant else 0,
                    "ranked_ids": [ids[i] for i in order],
                    "packed_ids": [ids[i] for i in packed],
                    "selection_ms": {
                        "bm25_reference": lexical_ms,
                        "classical_mmr": mmr_ms,
                        "coverage_diversity": coverage_ms,
                    }.get(name),
                }
            )
        if index % 50 == 0:
            print(f"Evaluated {index + 1}/{len(test_ids)} queries", flush=True)
    summary = []
    baseline = np.array([r["ndcg10"] for r in records if r["method"] == "faiss_dense"])
    for name in sorted({r["method"] for r in records}):
        subset = [r for r in records if r["method"] == name]
        values = np.array([r["ndcg10"] for r in subset])
        boot_ids = rng.integers(0, len(subset), (2000, len(subset)))
        summary.append(
            {
                "method": name,
                "n": len(subset),
                **{
                    metric: float(np.mean([r[metric] for r in subset]))
                    for metric in [
                        "ndcg10",
                        "recall10",
                        "packed_recall",
                        "packed_bytes",
                        "duplicate_pair_rate",
                    ]
                },
                "ndcg_ci95": np.quantile(values[boot_ids].mean(axis=1), [0.025, 0.975]).tolist(),
                "paired_delta_vs_dense_ci95": np.quantile(
                    (values - baseline)[boot_ids].mean(axis=1), [0.025, 0.975]
                ).tolist(),
            }
        )
    # Timed same-thread batched index searches, separate from embeddings and Python selection.
    index_timing = {}
    for name, engine, vectors in [("faiss_dense", dense, q), ("faiss_binary128", binary, qbits)]:
        engine.search(vectors[:1], 100)
        start = time.perf_counter()
        for _ in range(5):
            engine.search(vectors, 100)
        index_timing[name] = (time.perf_counter() - start) * 1000 / (5 * len(vectors))
    manifest = {
        "dataset": "BEIR SciFact",
        "dataset_url": "https://huggingface.co/datasets/BeIR/scifact",
        "dataset_license": "CC-BY-SA-4.0 (dataset card); raw corpus not redistributed",
        "data_sha256": dataset_hashes,
        "model": MODEL,
        "revision": REVISION,
        "model_license": "Apache-2.0",
        "device": "cpu",
        "documents": len(docs),
        "training_queries": train_ids,
        "validation_queries": val_ids,
        "test_queries": test_ids,
        "split_policy": "official test; training queries sharing labeled relevant documents with test excluded; remaining relevant-document connected components split 75/25 by hash",
        "training_pairs": len(train_examples),
        "training_seconds": training_seconds,
        "training_seeds": [7, 19, 43],
        "epochs": 600,
        "validation": validation,
        "selected_weight": chosen,
        "encoder_seconds_or_cache_load": encode_seconds,
        "reused_embedding_cache": reused_cache,
        "rss_bytes_at_end": psutil.Process().memory_info().rss,
        "raw_dense_vector_bytes": d.nbytes,
        "raw_binary_code_bytes": dbits.nbytes,
        "index_search_ms_per_query": index_timing,
        "memory_scope": "RSS is one combined process; raw payload excludes text, model, metadata and rerank vectors",
        "packages": {
            p: importlib.metadata.version(p)
            for p in ["numpy", "faiss-cpu", "torch", "sentence-transformers", "transformers", "psutil"]
        },
        "python": platform.python_version(),
        "os": platform.system(),
        "cpu_count": os.cpu_count(),
        "source_sha256": {
            str(p).replace("\\", "/"): digest(p)
            for p in [
                Path(__file__).relative_to(Path.cwd()),
                Path("context_stamps/baselines.py"),
                Path("context_stamps/selection.py"),
            ]
        },
        "limits": "single public retrieval dataset; no answer generation, model-token savings, energy or edge-device benchmark",
    }
    save(out / "manifest.json", manifest)
    save(out / "training.json", traces)
    save(out / "summary.json", summary)
    (out / "per-query.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("evidence/scifact-v1"))
    parser.add_argument("--cache", type=Path, required=True)
    args = parser.parse_args()
    run(args.data, args.out, args.cache)

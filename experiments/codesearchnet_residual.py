"""Test a 32-byte route with separately stored 8-bit precision residuals.

Raw CodeSearchNet rows and embeddings remain outside Git.
Copyright (c) 2026 Prashant Jagtap. MIT License.
"""

import argparse
import json
import statistics
import time
from pathlib import Path

import faiss
import numpy as np
from codesearchnet_matching_adapter import embed_test, load_vectors, pq_codebook
from codesearchnet_quantization import file_sha

K = 20


def score(order, targets):
    ranks = []
    for i, target in enumerate(targets):
        hits = np.flatnonzero(order[i] == target)
        ranks.append(int(hits[0]) + 1 if len(hits) else K + 1)
    return ranks


def summary(ranks):
    return {"n": len(ranks), "top1": sum(r == 1 for r in ranks),
            "top10": sum(r <= 10 for r in ranks),
            "mrr_at_20": round(sum(1 / r for r in ranks if r <= K) / len(ranks), 6)}


def routed_search(query, pq_index, sq_index, n):
    _, ids = pq_index.search(query, min(K, n))
    vectors = sq_index.reconstruct_batch(np.ascontiguousarray(ids.ravel(), dtype="int64"))
    vectors = vectors.reshape(len(query), ids.shape[1], 384)
    scores = np.einsum("qkd,qd->qk", vectors, query)
    return np.take_along_axis(ids, np.argsort(-scores, axis=1), axis=1)


def evaluate(cache, split, pq_template, sq_template):
    cohorts = json.loads((cache / f"{split}.json").read_text(encoding="utf-8"))
    docs, queries = load_vectors(cache, split)
    ranks = {name: [] for name in ("dense", "sq8", "pq32_sq8_top20")}
    timings = {name: [] for name in ranks}
    records = []
    doc_offset = query_offset = 0
    for cohort in cohorts:
        nd, nq = len(cohort["items"]), len(cohort["targets"])
        d = np.ascontiguousarray(docs[doc_offset:doc_offset + nd])
        q = np.ascontiguousarray(queries[query_offset:query_offset + nq])
        dense = faiss.IndexFlatIP(384)
        dense.add(d)
        pq_index = faiss.clone_index(pq_template)
        pq_index.reset()
        pq_index.add(d)
        sq_index = faiss.clone_index(sq_template)
        sq_index.reset()
        sq_index.add(d)
        methods = {"dense": lambda: dense.search(q, min(K, nd))[1],
                   "sq8": lambda: sq_index.search(q, min(K, nd))[1],
                   "pq32_sq8_top20": lambda: routed_search(q, pq_index, sq_index, nd)}
        record = {"repo": cohort["repo"], "candidates": nd, "ranks": {}}
        for name, search in methods.items():
            search()
            samples = []
            for _ in range(11):
                start = time.perf_counter()
                order = search()
                samples.append((time.perf_counter() - start) * 1000 / nq)
            found = score(order, cohort["targets"])
            record["ranks"][name] = found
            ranks[name].extend(found)
            timings[name].append(statistics.median(samples))
        records.append(record)
        doc_offset += nd
        query_offset += nq
    assert doc_offset == len(docs) and query_offset == len(queries)
    return {"summary": {name: summary(r) for name, r in ranks.items()},
            "warm_lookup_median_ms_per_query": {name: round(statistics.median(v), 6)
                                                for name, v in timings.items()},
            "rows": records}


def train(cache, out):
    out.mkdir(parents=True, exist_ok=True)
    faiss.omp_set_num_threads(4)
    docs, _ = load_vectors(cache, "train")
    pq_index = pq_codebook(docs)
    sq_index = faiss.IndexScalarQuantizer(384, faiss.ScalarQuantizer.QT_8bit, faiss.METRIC_INNER_PRODUCT)
    sq_index.train(docs)
    result = {"task": "derived CodeSearchNet Python repository-local exact-function localization",
              "metric": "top1, top10 and MRR@20; no full-rank MRR claim",
              "code_sizes": {"dense": 1536, "sq8": sq_index.code_size, "pq32_sq8_top20": pq_index.code_size + sq_index.code_size},
              "shared_bytes": {"dense": 45, "sq8": len(faiss.serialize_index(sq_index)),
                               "pq32_sq8_top20": len(faiss.serialize_index(pq_index)) + len(faiss.serialize_index(sq_index))},
              "train_vectors_sha256": file_sha(cache / "train-vectors.npz"),
              "validation": evaluate(cache, "validation", pq_index, sq_index)}
    base = result["validation"]["summary"]["dense"]
    eligible = [name for name in ("sq8", "pq32_sq8_top20")
                if result["validation"]["summary"][name]["top1"] >= base["top1"]
                and result["validation"]["summary"][name]["mrr_at_20"] >= base["mrr_at_20"] - 0.005]
    eligible.sort(key=lambda name: (-result["validation"]["summary"][name]["mrr_at_20"],
                                    result["validation"]["warm_lookup_median_ms_per_query"][name]))
    result["validation_gate"] = {"pass": bool(eligible), "selected": eligible[0] if eligible else None,
                                 "rule": "top1 >= dense and MRR@20 >= dense - 0.005; choose highest MRR@20, then lower lookup time"}
    (out / "validation.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"summary": result["validation"]["summary"], "gate": result["validation_gate"],
                      "timing": result["validation"]["warm_lookup_median_ms_per_query"]}, indent=2))


def test(cache, out):
    result = json.loads((out / "validation.json").read_text(encoding="utf-8"))
    if not result["validation_gate"]["pass"]:
        raise ValueError("validation gate failed; new cohort must remain unopened")
    manifest = json.loads((out.parent / "codesearchnet-matching-v1/cohort.json").read_text(encoding="utf-8"))
    if file_sha(cache / "matching-test-v1.json") != manifest["prepared_sha256"]:
        raise ValueError("new cohort differs from frozen manifest")
    if not (cache / "matching-test-v1-vectors.npz").exists():
        embed_test(cache)
    docs, _ = load_vectors(cache, "train")
    pq_index = pq_codebook(docs)
    sq_index = faiss.IndexScalarQuantizer(384, faiss.ScalarQuantizer.QT_8bit, faiss.METRIC_INNER_PRODUCT)
    sq_index.train(docs)
    result["test"] = evaluate(cache, "matching-test-v1", pq_index, sq_index)
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

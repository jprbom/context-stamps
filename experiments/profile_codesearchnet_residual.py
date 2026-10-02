"""Alternating-order warm CPU lookup profile for dense versus SQ8."""

import argparse
import json
import statistics
import time
from pathlib import Path

import faiss
import numpy as np
from codesearchnet_matching_adapter import load_vectors
from codesearchnet_quantization import file_sha


def profile(cache, threads):
    faiss.omp_set_num_threads(threads)
    train_docs, _ = load_vectors(cache, "train")
    sq_template = faiss.IndexScalarQuantizer(384, faiss.ScalarQuantizer.QT_8bit, faiss.METRIC_INNER_PRODUCT)
    sq_template.train(train_docs)
    cohorts = json.loads((cache / "matching-test-v1.json").read_text(encoding="utf-8"))
    docs, queries = load_vectors(cache, "matching-test-v1")
    samples = {"dense": [], "sq8": []}
    by_repo = []
    doc_offset = query_offset = 0
    for cohort in cohorts:
        n, nq = len(cohort["items"]), len(cohort["targets"])
        d = np.ascontiguousarray(docs[doc_offset:doc_offset + n])
        q = np.ascontiguousarray(queries[query_offset:query_offset + nq])
        indexes = {"dense": faiss.IndexFlatIP(384), "sq8": faiss.clone_index(sq_template)}
        indexes["sq8"].reset()
        for index in indexes.values():
            index.add(d)
            index.search(q, min(20, n))
        own = {"dense": [], "sq8": []}
        for repeat in range(50):
            order = ("dense", "sq8") if repeat % 2 else ("sq8", "dense")
            for name in order:
                start = time.perf_counter_ns()
                indexes[name].search(q, min(20, n))
                elapsed = (time.perf_counter_ns() - start) / 1e6 / nq
                own[name].append(elapsed)
                samples[name].append(elapsed)
        by_repo.append({"repo": cohort["repo"], "candidates": n,
                        "median_ms_per_query": {name: round(statistics.median(v), 6) for name, v in own.items()}})
        doc_offset += n
        query_offset += nq
    assert doc_offset == len(docs) and query_offset == len(queries)
    return {"protocol": "50 repeats per repository; alternating order; three-query batches; warm FAISS search only",
            "threads": threads, "queries": len(queries), "candidates": len(docs),
            "test_vectors_sha256": file_sha(cache / "matching-test-v1-vectors.npz"),
            "per_query_batch_time_ms": {name: {"median": round(float(np.median(v)), 6),
                                                "p95": round(float(np.quantile(v, 0.95)), 6)}
                                        for name, v in samples.items()}, "by_repository": by_repo}


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--cache", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--threads", type=int, choices=[1, 4], default=4)
    a = p.parse_args()
    a.out.write_text(json.dumps(profile(a.cache, a.threads), indent=2) + "\n", encoding="utf-8")

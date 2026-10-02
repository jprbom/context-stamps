"""Exploratory precision-layer diagnostic on locally prepared code embeddings.

The input cache is not redistributed. This probe uses already inspected cohorts;
it is a cost and correctness diagnostic, not a fresh retrieval benchmark.
Copyright (c) 2026 Prashant Jagtap. MIT License.
"""

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np

from context_stamps.residual_index import ResidualIndex


def normalized(values):
    if values.dtype != np.float32 or values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError("finite float32 matrix required")
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    if (norms <= 0).any():
        raise ValueError("zero vector")
    return (values / norms).astype(np.float32)


def grouped_bounds(values, queries, groups, top_k):
    """Ideal float64 scan with one scale per row and group; no wall-time claim."""
    rows, dimension = values.shape
    if dimension % groups:
        raise ValueError("group count must divide the embedding dimension")
    width = dimension // groups
    reconstructed = np.empty((rows, dimension), dtype=np.float64)
    source = values.astype(np.float64)
    for group in range(groups):
        part = slice(group * width, (group + 1) * width)
        scale = np.maximum(np.max(np.abs(source[:, part]), axis=1) / 127,
                           np.finfo(np.float64).tiny)
        code = np.rint(source[:, part] / scale[:, None]).clip(-127, 127).astype(np.int8)
        reconstructed[:, part] = code * scale[:, None]
    residual = np.linalg.norm(source - reconstructed, axis=1)
    candidates = []
    for query in queries:
        scores = reconstructed @ query.astype(np.float64)
        # Generous fixed slack makes this a diagnostic, not a numerical certificate.
        radius = residual + 1e-7
        threshold = np.partition(scores - radius, rows - top_k)[rows - top_k]
        candidates.append(int(np.count_nonzero(scores + radius >= threshold)))
    return {"groups": groups, "mean_residual_norm": float(np.mean(residual)),
            "mean_possible_refinements": float(np.mean(candidates)),
            "max_possible_refinements": max(candidates),
            "extra_float32_scale_bytes": rows * (groups - 1) * 4}


def probe(path):
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    with np.load(path, allow_pickle=False) as cache:
        if set(cache.files) != {"docs", "queries"}:
            raise ValueError("docs and queries required")
        docs, queries = normalized(cache["docs"]), normalized(cache["queries"])
    if docs.shape[1] != queries.shape[1] or len(docs) < 10:
        raise ValueError("incompatible code embeddings")
    ids = np.arange(len(docs))
    index = ResidualIndex(docs, encoder_id="code-precision-probe")

    def dense(query):
        scores = np.einsum("ij,j->i", docs, query.astype(np.float64), dtype=np.float64)
        return np.lexsort((ids, -scores))[:10].tolist()

    def verified_sq8(query):
        return index.search(query, encoder_id="code-precision-probe", eligible=ids,
                            fetch=lambda rows: docs[rows], limit=10)

    direct = [dense(query) for query in queries]
    verified = [verified_sq8(query) for query in queries]
    if direct != [row["rows"] for row in verified]:
        raise AssertionError("certified index disagrees with exact dense ranking")
    times = {"dense": [], "verified_sq8": []}
    for _ in range(3):
        start = time.perf_counter()
        for query in queries:
            dense(query)
        times["dense"].append(time.perf_counter() - start)
        start = time.perf_counter()
        for query in queries:
            verified_sq8(query)
        times["verified_sq8"].append(time.perf_counter() - start)
    return {"cache_sha256": digest, "rows": len(docs), "queries": len(queries),
            "dimensions": docs.shape[1], "dense_vector_bytes": docs.nbytes,
            "verified_sq8_index_bytes": index.index_bytes,
            "verified_sq8_with_backing_bytes": docs.nbytes + index.index_bytes,
            "verified_sq8_mean_refined": float(np.mean([row["refined_rows"] for row in verified])),
            "all_exact_top10": True,
            "three_full_sweep_seconds": times,
            "grouped_scale_diagnostic": [grouped_bounds(docs, queries, groups, 10)
                                         for groups in (1, 4, 16)]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = {"schema": 1, "scope": "already inspected CodeSearchNet-derived vectors",
              "runtime": "local CPU, sequential queries, warm loaded arrays",
              "cohorts": {name: probe(args.cache / f"{name}-vectors.npz")
                          for name in ("validation", "test")}}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({name: {"all_exact_top10": row["all_exact_top10"],
                             "verified_sq8_mean_refined": row["verified_sq8_mean_refined"]}
                      for name, row in result["cohorts"].items()}))


if __name__ == "__main__":
    main()

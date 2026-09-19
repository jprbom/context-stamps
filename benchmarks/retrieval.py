"""Synthetic held-out retrieval benchmark. No model download or private corpus."""

import argparse
import hashlib
import json
import platform
import time
from pathlib import Path

import numpy as np

import stamps
from context_stamps import Family, hamming, learning, stamp_vector
from context_stamps.learning import fit_family


def run(seed=20260919):
    rng = np.random.default_rng(seed)
    # Separate training, index and query rows from the same synthetic distribution.
    vectors = rng.normal(size=(776, 32))
    vectors[:, :4] += 2.5
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    train, index, queries = vectors[:512], vectors[512:712], vectors[712:]
    truth = np.argsort(-(queries @ index.T), axis=1, kind="stable")[:, :10]
    methods = [Family("synthetic-normalized-v1", 32, bits=16, seed=seed)]
    methods += [
        fit_family(train, encoder="synthetic-normalized-v1", bits=16, method=name, seed=seed)
        for name in ("centered", "itq")
    ]
    results = []
    for family in methods:
        start = time.perf_counter()
        stored = [stamp_vector(v, family) for v in index]
        query_stamps = [stamp_vector(v, family) for v in queries]
        stamp_ms = (time.perf_counter() - start) * 1000
        recalls, hybrid = [], []
        start = time.perf_counter()
        for q, code, expected in zip(queries, query_stamps, truth):
            order = sorted(range(len(index)), key=lambda i: (hamming(code, stored[i]), i))
            recalls.append(len(set(order[:10]) & set(expected)) / 10)
            shortlist = np.asarray(order[:50])
            reranked = shortlist[np.argsort(-(index[shortlist] @ q), kind="stable")[:10]]
            hybrid.append(len(set(reranked) & set(expected)) / 10)
        results.append(
            {
                "method": family.method,
                "family_id": family.identity,
                "recall_at_10": float(np.mean(recalls)),
                "hybrid_recall_at_10_shortlist_50": float(np.mean(hybrid)),
                "stamping_ms": round(stamp_ms, 3),
                "query_and_rerank_ms": round((time.perf_counter() - start) * 1000, 3),
            }
        )
    code_hash = hashlib.sha256()
    for path in (Path(__file__), Path(stamps.__file__), Path(learning.__file__)):
        code_hash.update(path.read_bytes().replace(b"\r\n", b"\n"))
    return {
        "scope": "synthetic vector retrieval; not semantic or application quality",
        "seed": seed,
        "training": len(train),
        "index": len(index),
        "queries": len(queries),
        "dim": 32,
        "bits": 16,
        "data_sha256": hashlib.sha256(vectors.tobytes()).hexdigest(),
        "code_sha256": code_hash.hexdigest(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "os": platform.system(),
        "raw_code_bytes_per_item": 2,
        "raw_float32_vector_bytes_per_item": 128,
        "memory_note": "raw payload only; excludes text, metadata, family, objects and rerank vectors",
        "results": results,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="results.json")
    parser.add_argument("--seed", type=int, default=20260919)
    args = parser.parse_args()
    result = run(args.seed)
    Path(args.out).write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))

"""Public retrieval and component-scale evaluation with no encoder retraining."""

import argparse
import hashlib
import json
import platform
import random
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import faiss
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from context_stamps.residual_index import ResidualIndex
from experiments.run_spherical_public import ndcg, qrels, read, save, sha

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "evidence/residual-v1"


def elapsed(start):
    return (time.perf_counter_ns() - start) / 1e6


def indexes(documents):
    start = time.perf_counter_ns()
    residual = ResidualIndex(documents, encoder_id="minilm-pinned")
    builds = {"residual_int8_exact": elapsed(start)}
    start = time.perf_counter_ns()
    flat = faiss.IndexFlatIP(documents.shape[1])
    flat.add(documents)
    builds["faiss_flat_ip"] = elapsed(start)
    start = time.perf_counter_ns()
    sq = faiss.IndexScalarQuantizer(documents.shape[1], faiss.ScalarQuantizer.QT_8bit, faiss.METRIC_INNER_PRODUCT)
    sq.train(documents)
    sq.add(documents)
    builds["faiss_sq8"] = elapsed(start)
    sizes = {"dense_float64_reference": documents.nbytes, "faiss_flat_ip": len(faiss.serialize_index(flat)),
             "faiss_sq8": len(faiss.serialize_index(sq)), "residual_int8_exact": residual.index_bytes}
    return residual, flat, sq, builds, sizes


def run_methods(query, documents, eligible, residual, flat, sq):
    def dense():
        values = np.einsum("ij,j->i", documents[eligible], query.astype(np.float64), dtype=np.float64)
        order = np.lexsort((eligible, -values))[:10]
        return {"rows": eligible[order].tolist(), "refined_rows": len(eligible)}
    allowed = set(eligible.tolist())
    def faiss_search(index):
        # Benchmark eligibility excludes at most one self-document; no other ACL claim.
        _, rows = index.search(query[None], 11)
        return {"rows": [int(i) for i in rows[0] if i in allowed][:10], "refined_rows": len(eligible)}
    return {"dense_float64_reference": dense, "faiss_flat_ip": lambda: faiss_search(flat),
            "faiss_sq8": lambda: faiss_search(sq),
            "residual_int8_exact": lambda: residual.search(query, encoder_id="minilm-pinned", eligible=eligible,
                                                           fetch=lambda ids: documents[ids])}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scifact", type=Path, required=True)
    parser.add_argument("--replication-data", type=Path, required=True)
    parser.add_argument("--scifact-cache", type=Path, required=True)
    parser.add_argument("--replication-cache", type=Path, required=True)
    args = parser.parse_args()
    faiss.omp_set_num_threads(1)
    prior = read(ROOT / "evidence/spherical-public-v1/manifest.json")
    records, summaries, storage = [], [], []
    rng = random.Random(81471)
    for metadata in prior["datasets"]:
        name = metadata["dataset"]
        root = args.scifact if name == "scifact" else args.replication_data / name
        assert all(sha(root / p) == digest for p, digest in metadata["data_sha256"].items())
        ids = [json.loads(line)["_id"] for line in (root / "corpus.jsonl").read_text(encoding="utf-8").splitlines()]
        labels = qrels(root / "qrels/test.tsv")
        key = hashlib.sha256(json.dumps([metadata["data_sha256"], prior["protocol"]["encoder"],
                                         prior["protocol"]["revision"], metadata["query_ids"]], sort_keys=True).encode()).hexdigest()
        cache = args.scifact_cache if name == "scifact" else args.replication_cache
        dp, qp = cache / (key + "-D.npy"), cache / (key + "-Q.npy")
        assert sha(dp) == metadata["embedding_sha256"]["documents"]
        assert sha(qp) == metadata["embedding_sha256"]["queries"]
        documents, queries = np.load(dp, allow_pickle=False), np.load(qp, allow_pickle=False)
        residual, flat, sq, builds, sizes = indexes(documents)
        storage.append({"dataset": name, "rows": len(documents), "build_ms": builds, "index_bytes": sizes,
                        "residual_external_backing_bytes": documents.nbytes})
        positions = {qid: i for i, qid in enumerate(metadata["query_ids"])}
        for function in run_methods(queries[0], documents, np.arange(len(documents)), residual, flat, sq).values():
            function()
        for qid in sorted(labels):
            query = queries[positions[qid]]
            eligible = np.array([i for i, docid in enumerate(ids) if docid != qid], dtype=np.int64)
            methods = list(run_methods(query, documents, eligible, residual, flat, sq).items())
            rng.shuffle(methods)
            outcomes = {}
            for method, function in methods:
                start = time.perf_counter_ns()
                result = function()
                outcomes[method] = {**result, "search_ms": elapsed(start)}
            gold = outcomes["dense_float64_reference"]["rows"]
            for method, result in outcomes.items():
                records.append({"dataset": name, "query_id": qid, "method": method,
                                "ndcg10": ndcg(result["rows"], labels[qid], ids),
                                "ranked_ids": [ids[i] for i in result["rows"]],
                                "same_order_as_dense": result["rows"] == gold,
                                "refined_fraction": result["refined_rows"] / len(eligible),
                                "search_ms": result["search_ms"]})
        for method in ("dense_float64_reference", "faiss_flat_ip", "faiss_sq8", "residual_int8_exact"):
            rows = [r for r in records if (r["dataset"], r["method"]) == (name, method)]
            summaries.append({"dataset": name, "method": method, "queries": len(rows),
                              "ndcg10": statistics.mean(r["ndcg10"] for r in rows),
                              "same_order_fraction": statistics.mean(r["same_order_as_dense"] for r in rows),
                              "mean_refined_fraction": statistics.mean(r["refined_fraction"] for r in rows),
                              "median_ms": statistics.median(r["search_ms"] for r in rows),
                              "p95_ms": float(np.percentile([r["search_ms"] for r in rows], 95))})
        print(name, summaries[-4:], flush=True)
    # Independent component-scale workload, no corpus duplication or claimed task quality.
    generator = np.random.default_rng(64109)
    scale = []
    for count in (1000, 10000, 100000):
        documents = generator.normal(size=(count, 384)).astype(np.float32)
        documents /= np.linalg.norm(documents, axis=1, keepdims=True)
        queries = generator.normal(size=(30, 384)).astype(np.float32)
        queries /= np.linalg.norm(queries, axis=1, keepdims=True)
        index = ResidualIndex(documents, encoder_id="scale")
        eligible = np.arange(count)
        def search(q):
            start = time.perf_counter_ns()
            result = index.search(q, encoder_id="scale", eligible=eligible, fetch=lambda rows: documents[rows])
            return result, elapsed(start)
        expected = []
        for query in queries:
            score = np.einsum("ij,j->i", documents, query.astype(np.float64), dtype=np.float64)
            expected.append(np.lexsort((eligible, -score))[:10].tolist())
        for workers in (1, 4):
            start = time.perf_counter_ns()
            with ThreadPoolExecutor(max_workers=workers) as pool:
                results = list(pool.map(search, queries))
            wall = elapsed(start)
            assert [r[0]["rows"] for r in results] == expected
            scale.append({"rows": count, "concurrency": workers, "queries": 30,
                          "throughput_qps": 30000 / wall, "median_ms": statistics.median(r[1] for r in results),
                          "p95_ms": float(np.percentile([r[1] for r in results], 95)),
                          "mean_refined_fraction": statistics.mean(r[0]["refined_rows"] / count for r in results),
                          "index_bytes": index.index_bytes, "backing_bytes": documents.nbytes,
                          "all_orders_equal_dense": True})
    for name, value in (("results", records), ("summary", summaries), ("storage", storage), ("scale", scale)):
        save(OUT / (name + ".json"), value)
    save(OUT / "manifest.json", {"protocol": read(OUT / "protocol.json"), "numpy": np.__version__,
                                 "faiss": faiss.__version__, "platform": platform.platform(),
                                 "source_sha256": {p: sha(ROOT / p) for p in (
                                     "experiments/run_residual.py", "context_stamps/residual_index.py",
                                     "experiments/run_spherical_public.py")},
                                 "input_sha256": {"evidence/spherical-public-v1/manifest.json": sha(
                                     ROOT / "evidence/spherical-public-v1/manifest.json")}})
    save(OUT / "checksums.json", {p.name: sha(p) for p in OUT.glob("*.json") if p.name != "checksums.json"})


if __name__ == "__main__":
    main()

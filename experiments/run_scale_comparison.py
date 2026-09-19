"""Compare component throughput against optimized Faiss baselines."""

import random
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import faiss
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.run_residual_v2 import indexes
from experiments.run_spherical_public import read, save, sha

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "evidence/scale-comparison-v1"


def main():
    protocol = read(OUT / "protocol.json")
    generator = np.random.default_rng(protocol["seed"])
    order_rng = random.Random(protocol["seed"])
    results, records = [], []
    for count in protocol["rows"]:
        documents = generator.normal(size=(count, 384)).astype(np.float32)
        documents /= np.linalg.norm(documents, axis=1, keepdims=True)
        queries = generator.normal(size=(30, 384)).astype(np.float32)
        queries /= np.linalg.norm(queries, axis=1, keepdims=True)
        faiss.omp_set_num_threads(1)
        residual, flat, sq, _, sizes = indexes(documents)
        eligible = np.arange(count)
        expected = [np.lexsort((eligible, -np.einsum("ij,j->i", documents, q.astype(np.float64), dtype=np.float64)))[:10].tolist() for q in queries]
        configurations = [(method, workers) for method in protocol["methods"] for workers in protocol["concurrency"]]
        order_rng.shuffle(configurations)
        for method, workers in configurations:
            def search(item):
                index, query = item
                faiss.omp_set_num_threads(1)
                start = time.perf_counter_ns()
                if method == "residual_int8_exact":
                    result = residual.search(query, encoder_id="minilm-pinned", eligible=eligible, fetch=lambda ids: documents[ids])
                    rows = result["rows"]
                else:
                    _, ids = (flat if method == "faiss_flat_ip" else sq).search(query[None], 10)
                    rows = ids[0].tolist()
                ms = (time.perf_counter_ns() - start) / 1e6
                return {"query": index, "ms": ms, "same_order": rows == expected[index]}
            search((0, queries[0]))
            start = time.perf_counter_ns()
            with ThreadPoolExecutor(max_workers=workers) as pool:
                observed = list(pool.map(search, enumerate(queries)))
            wall = (time.perf_counter_ns() - start) / 1e6
            records.extend({**r, "rows": count, "method": method, "concurrency": workers} for r in observed)
            results.append({"rows": count, "method": method, "concurrency": workers, "queries": len(observed),
                            "throughput_qps": 30000 / wall, "median_ms": statistics.median(r["ms"] for r in observed),
                            "p95_ms": float(np.percentile([r["ms"] for r in observed], 95)),
                            "same_order_fraction": statistics.mean(r["same_order"] for r in observed),
                            "index_bytes": sizes[method], "backing_bytes": documents.nbytes if method == "residual_int8_exact" else 0})
        print(count, results[-6:], flush=True)
    save(OUT / "results.json", records)
    save(OUT / "summary.json", results)
    save(OUT / "manifest.json", {"protocol": protocol, "faiss": faiss.__version__, "numpy": np.__version__,
        "source_sha256": {p: sha(ROOT / p) for p in ("experiments/run_scale_comparison.py", "experiments/run_residual_v2.py", "context_stamps/residual_index.py")}})
    save(OUT / "checksums.json", {p.name: sha(p) for p in OUT.glob("*.json") if p.name != "checksums.json"})


if __name__ == "__main__":
    main()

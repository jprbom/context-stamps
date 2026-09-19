"""Synthetic packed-index scale measurement; not an end-to-end agent benchmark."""

import hashlib
import json
import platform
import statistics
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from context_stamps.activation import StampSchema
from context_stamps.packed_index import PackedStampIndex
from context_stamps.spherical import SphericalStamp
from stamps import Family

ROOT = Path(__file__).resolve().parents[1]


def main():
    rng = np.random.default_rng(75231)
    names = ("a", "b", "c", "d")
    query = SphericalStamp.encode({n: [1, 2] for n in names}, {n: Family(n, 2, 64) for n in names})
    schema = StampSchema.for_stamp(query)
    rows = []
    for size in (1000, 10000, 100000):
        data = rng.integers(0, 256, (size, 32), dtype=np.uint8)
        start = time.perf_counter_ns()
        index = PackedStampIndex(schema, [str(i) for i in range(size)], data)
        build_ms = (time.perf_counter_ns() - start) / 1e6
        durations = []
        expected = None
        for _ in range(20):
            start = time.perf_counter_ns()
            result = index.search(query, eligible=np.arange(size), limit=10)
            durations.append((time.perf_counter_ns() - start) / 1e6)
            if expected is None:
                expected = result
            assert result == expected
        rows.append({"rows": size, "code_bytes": index.code_bytes, "build_ms": build_ms,
                     "queries": 20, "query_median_ms": statistics.median(durations),
                     "query_p95_ms": float(np.quantile(durations, .95)), "query_ms": durations,
                     "top10": expected, "data_sha256": hashlib.sha256(data.tobytes()).hexdigest()})
    output = ROOT / "evidence/packed-scale-v1"
    output.mkdir(exist_ok=True)
    source = {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest()
              for p in ("context_stamps/packed_index.py", "benchmarks/packed_scale.py")}
    (output / "results.json").write_text(json.dumps({"platform": platform.platform(), "seed": 75231,
        "numpy": np.__version__, "source_sha256": source, "results": rows,
        "limits": "random binary codes; linear scan; code bytes exclude keys, metadata, documents and temporary arrays; no ingestion/encoder or network timing"},
        indent=2) + "\n", encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()

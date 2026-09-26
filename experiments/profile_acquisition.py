"""CPU-only planner overhead; exact outputs, no retrieval or generation timing."""

import argparse
import gzip
import json
import math
import random
import statistics
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from source_evidence import sha  # noqa: E402

from context_stamps.acquisition import (  # noqa: E402
    AcquisitionModel,
    StopPolicy,
    candidate_order,
    plan_prefix,
    prefix_features,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--work", type=Path, default=ROOT.parent)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    if args.out.exists():
        raise ValueError("fresh profile directory required")
    args.out.mkdir(parents=True)
    evidence = ROOT / "evidence/acquisition-v1"
    before = time.perf_counter()
    model = AcquisitionModel.load(evidence / "mlp32-seed29.json")
    policy = StopPolicy(model.revision, .9)
    initialize_seconds = time.perf_counter() - before
    methods = ("full-pool-order", "guarded-fallback", "eager-feature-control")
    manifest = json.loads((ROOT / "evidence/controller-v2/data-manifest.json").read_text())
    original = json.loads(gzip.decompress((evidence / "trajectories.json.gz").read_bytes()))["test"]
    lookup = {(r["dataset"], r["query_id"]): r for r in original}
    cases, parity = [], 0.
    for dataset in ("scifact", "nfcorpus", "arguana", "scidocs", "fiqa"):
        info = next(p for p in manifest["partitions"] if p["dataset"] == dataset and p["split"] == "test")
        path = args.work / f"controller-v2-data/{dataset}-test.npz"
        if sha(path.read_bytes()) != info["array_sha256"]:
            raise ValueError("prepared score cache changed")
        with np.load(path, allow_pickle=False) as arrays:
            for i, query in enumerate(manifest["assignments"][dataset]["test"][:20]):
                rows = tuple(tuple(float(v) for v in row) for row in arrays["F"][i, arrays["mask"][i]])
                order, lengths, features = prefix_features(rows)
                predictions = [model.predict(tuple(float(v) for v in row)) for row in np.asarray(features, np.float32)]
                reference = lookup[dataset, query]
                if list(lengths) != reference["lengths"]:
                    raise ValueError("feature prefix drift")
                parity = max(parity, float(np.abs(np.asarray(predictions) - reference["predictions"]).max()))
                cases.append(dict(dataset=dataset, query_id=query, rows=rows, order=order))
    if parity > 1e-12:
        raise ValueError("feature/prediction drift after fallback optimization")

    def run(case, method):
        if method == "full-pool-order":
            return candidate_order(case["rows"])
        if method == "eager-feature-control":
            return prefix_features(case["rows"])[0]
        result = plan_prefix(case["rows"], model, policy, scope=case["dataset"])
        if result.status != "unqualified_full_pool":
            raise ValueError("unqualified planner did not retain pool")
        return result.indices

    for case in cases:
        for method in methods:
            if run(case, method) != case["order"]:
                raise ValueError("warmup output mismatch")
    rows = []
    for seed in (7, 29, 61):
        jobs = [(i, method, repeat) for i in range(len(cases)) for method in methods for repeat in range(10)]
        random.Random(seed).shuffle(jobs)
        for i, method, repeat in jobs:
            began = time.perf_counter_ns()
            result = run(cases[i], method)
            elapsed = time.perf_counter_ns() - began
            if result != cases[i]["order"]:
                raise ValueError("timed output mismatch")
            rows.append(dict(case=i, method=method, seed=seed, repeat=repeat, nanoseconds=elapsed, correct=True))
    summaries = []
    for method in methods:
        values = sorted(r["nanoseconds"] / 1000000 for r in rows if r["method"] == method)
        summaries.append(dict(method=method, samples=len(values), p50_ms=statistics.median(values),
                              p95_ms=values[math.ceil(.95 * len(values)) - 1]))
    raw = json.dumps(rows, separators=(",", ":"), sort_keys=True).encode()
    (args.out / "timings.json.gz").write_bytes(gzip.compress(raw, mtime=0))
    sources = ("context_stamps/acquisition.py", "experiments/profile_acquisition.py")
    report = dict(schema=1, source_hashes={p: sha((ROOT / p).read_bytes()) for p in sources},
        model_sha256=sha((evidence / "mlp32-seed29.json").read_bytes()), initialize_seconds=initialize_seconds,
        cases=[{k: v for k, v in c.items() if k not in ("rows", "order")} for c in cases],
        prediction_parity_max_error=parity, warmup_calls=300, seeds=[7, 29, 61], repeats_per_case_method=10,
        timing_sha256=sha((args.out / "timings.json.gz").read_bytes()), summary=summaries,
        boundary="Cached retrieval-score rows; whole local planner call, batch1, CPU only. Eager-feature control computes unused prefix features, without old model hashing overhead. Not a previous-version executable or end-to-end model latency. Initialization, retrieval, encoding, I/O and generation excluded.")
    (args.out / "manifest.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()

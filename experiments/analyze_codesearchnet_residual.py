"""Paired repository bootstrap and complete vector-index byte accounting."""

import argparse
import json
from pathlib import Path

import numpy as np


def analyze(source):
    data = json.loads(source.read_text(encoding="utf-8"))
    output = {"seed": 20261002, "bootstrap_replicates": 10000,
              "bootstrap_unit": "repository", "splits": {}}
    for split in ("validation", "test"):
        records = data[split]["rows"]
        dense = np.asarray([r["ranks"]["dense"] for r in records], dtype=float)
        total = sum(r["candidates"] for r in records)
        rng = np.random.default_rng(20261002 + (split == "test"))
        draw = rng.integers(0, len(records), size=(10000, len(records)))
        report = {"repositories": len(records), "queries": dense.size, "candidates": total, "routes": {}}
        for route in data[split]["summary"]:
            rank = np.asarray([r["ranks"][route] for r in records], dtype=float)
            diff_top1 = ((rank == 1).astype(float) - (dense == 1).astype(float)).mean(axis=1)
            reciprocal = np.where(rank <= 20, 1 / rank, 0)
            reciprocal_dense = np.where(dense <= 20, 1 / dense, 0)
            diff_mrr = (reciprocal - reciprocal_dense).mean(axis=1)
            bytes_total = total * data["code_sizes"][route] + data["shared_bytes"][route]
            report["routes"][route] = {
                "index_bytes": bytes_total,
                "warm_lookup_median_ms_per_query": data[split]["warm_lookup_median_ms_per_query"][route],
                "top1_delta_vs_dense": round(float(diff_top1.mean()), 6),
                "top1_delta_95pct_cluster_bootstrap": [round(float(v), 6) for v in np.quantile(diff_top1[draw].mean(axis=1), [0.025, 0.975])],
                "mrr_at_20_delta_vs_dense": round(float(diff_mrr.mean()), 6),
                "mrr_at_20_delta_95pct_cluster_bootstrap": [round(float(v), 6) for v in np.quantile(diff_mrr[draw].mean(axis=1), [0.025, 0.975])],
                "ranks_improved": int((rank < dense).sum()),
                "ranks_worsened": int((rank > dense).sum()),
            }
        output["splits"][split] = report
    return output


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("source", type=Path)
    p.add_argument("output", type=Path)
    a = p.parse_args()
    a.output.write_text(json.dumps(analyze(a.source), indent=2) + "\n", encoding="utf-8")

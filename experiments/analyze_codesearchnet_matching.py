"""Paired repository bootstrap for the matching-aware validation routes."""

import argparse
import json
from pathlib import Path

import numpy as np


def analyze(source):
    data = json.loads(source.read_text(encoding="utf-8"))
    records = data["validation"]["rows"]
    baseline = np.asarray([r["ranks"]["dense"] for r in records], dtype=float)
    rng = np.random.default_rng(20261002)
    draw = rng.integers(0, len(records), size=(10000, len(records)))
    output = {"split": "inspected validation only", "bootstrap_unit": "repository",
              "replicates": 10000, "seed": 20261002, "routes": {}}
    for name in data["validation"]["summary"]:
        if name == "dense":
            continue
        rank = np.asarray([r["ranks"][name] for r in records], dtype=float)
        top1 = ((rank == 1).astype(float) - (baseline == 1).astype(float)).mean(axis=1)
        mrr = (1 / rank - 1 / baseline).mean(axis=1)
        output["routes"][name] = {
            "top1_delta": round(float(top1.mean()), 6),
            "top1_delta_95pct_cluster_bootstrap": [round(float(x), 6) for x in np.quantile(top1[draw].mean(axis=1), [0.025, 0.975])],
            "mrr_delta": round(float(mrr.mean()), 6),
            "mrr_delta_95pct_cluster_bootstrap": [round(float(x), 6) for x in np.quantile(mrr[draw].mean(axis=1), [0.025, 0.975])],
            "ranks_improved": int((rank < baseline).sum()),
            "ranks_worsened": int((rank > baseline).sum()),
        }
    return output


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("source", type=Path)
    p.add_argument("output", type=Path)
    a = p.parse_args()
    a.output.write_text(json.dumps(analyze(a.source), indent=2) + "\n", encoding="utf-8")

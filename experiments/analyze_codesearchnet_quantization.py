"""Cluster bootstrap and storage break-even for the frozen study results."""

import argparse
import json
from pathlib import Path

import numpy as np


def analyze(path):
    result = json.loads(path.read_text(encoding="utf-8"))
    analysis = {"seed": 20260930, "bootstrap_replicates": 10000, "unit": "repository", "splits": {}}
    for split in ("validation", "test"):
        rows = result[split]["rows"]
        names = result[split]["summary"]
        dense = np.asarray([r["ranks"]["dense"] for r in rows], dtype=float)
        rng = np.random.default_rng(20260930 + (split == "test"))
        draws = rng.integers(0, len(rows), size=(10000, len(rows)))
        report = {}
        for name in names:
            if name == "dense":
                continue
            compact = np.asarray([r["ranks"][name] for r in rows], dtype=float)
            top1_delta = (compact == 1).astype(float) - (dense == 1).astype(float)
            mrr_delta = 1 / compact - 1 / dense
            top1_boot = top1_delta.mean(axis=1)[draws].mean(axis=1)
            mrr_boot = mrr_delta.mean(axis=1)[draws].mean(axis=1)
            overhead = result["codebook_bytes"][name] - result["codebook_bytes"]["dense"]
            break_even = int(np.ceil(overhead / (384 * 4 - 32)))
            report[name] = {
                "top1_delta_fraction": round(float(top1_delta.mean()), 6),
                "top1_delta_95pct_cluster_bootstrap": [round(float(x), 6) for x in np.quantile(top1_boot, [0.025, 0.975])],
                "mrr_delta": round(float(mrr_delta.mean()), 6),
                "mrr_delta_95pct_cluster_bootstrap": [round(float(x), 6) for x in np.quantile(mrr_boot, [0.025, 0.975])],
                "rank_improved": int((compact < dense).sum()),
                "rank_worsened": int((compact > dense).sum()),
                "storage_break_even_candidates_excluding_metadata_and_encoder": break_even,
            }
        analysis["splits"][split] = report
    return analysis


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("results", type=Path)
    p.add_argument("output", type=Path)
    args = p.parse_args()
    args.output.write_text(json.dumps(analyze(args.results), indent=2) + "\n", encoding="utf-8")

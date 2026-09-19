"""Render the cross-dataset comparison directly from committed measurements."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def main():
    original = json.loads((ROOT / "evidence/scifact-v1/summary.json").read_text())
    followup = json.loads((ROOT / "evidence/replication-v1/summary.json").read_text())
    dense = [next(r["ndcg10"] for r in original if r["method"] == "faiss_dense")]
    selected = [next(r["ndcg10"] for r in original if r["method"] == "coverage_diversity")]
    for dataset in ["nfcorpus", "arguana"]:
        for method, target in [("dense", dense), ("coverage_diversity", selected)]:
            target.append(
                next(
                    r["ndcg10"]
                    for r in followup
                    if r["dataset"] == dataset and r["stratum"] == "all" and r["method"] == method
                )
            )
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11})
    fig, ax = plt.subplots(figsize=(10, 5.2), layout="constrained")
    fig.set_facecolor("#f8fafc")
    ax.set_facecolor("#f8fafc")
    x = np.arange(3)
    for offset, values, label, color in [
        (-0.19, dense, "Dense retrieval", "#334155"),
        (0.19, selected, "Coverage / diversity", "#0f766e"),
    ]:
        bars = ax.bar(x + offset, values, width=0.36, label=label, color=color)
        ax.bar_label(bars, labels=[f"{v:.4f}" for v in values], padding=5, fontsize=11)
    ax.set_xticks(x, ["SciFact\n300 queries", "NFCorpus\n323 queries", "ArguAna\n1,406 queries"])
    ax.set_ylim(0, 0.83)
    ax.set_ylabel("nDCG@10 · higher is better")
    ax.set_title("The SciFact gain did not transfer", loc="left", fontsize=20, weight="bold", pad=42)
    ax.text(
        0,
        1.06,
        "Frozen selector settings · same pinned MiniLM encoder · all test queries",
        transform=ax.transAxes,
        color="#475569",
    )
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.15)
    ax.set_axisbelow(True)
    ax.legend(loc="upper right", frameon=False)
    fig.savefig(ROOT / "docs/assets/replication-results.png", dpi=170)
    plt.close(fig)


if __name__ == "__main__":
    main()

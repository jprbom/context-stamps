"""Render the checked hybrid-retrieval-v1 summary as a publication-style chart."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "evidence" / "hybrid-retrieval-v1" / "summary.json"
OUTPUT = ROOT / "docs" / "assets" / "hybrid-retrieval-comparison.png"


def main():
    rows = json.loads(SUMMARY.read_text(encoding="utf-8"))
    names = [row["dataset"].replace("scifact", "SciFact").replace("nfcorpus", "NFCorpus")
             .replace("arguana", "ArguAna").replace("scidocs", "SciDocs") for row in rows]
    methods = [("dense_minilm", "Dense MiniLM", "#315f67"),
               ("bm25", "BM25", "#9b88a5"),
               ("hybrid_z_075", "Frozen hybrid", "#b6654a")]
    x = np.arange(len(rows))
    width = .23
    fig, axis = plt.subplots(figsize=(10.8, 5.8), facecolor="#f4f2ec")
    axis.set_facecolor("#f4f2ec")
    for index, (key, label, color) in enumerate(methods):
        values = [row["methods"][key]["ndcg10"] for row in rows]
        bars = axis.bar(x + (index - 1) * width, values, width, label=label, color=color)
        axis.bar_label(bars, labels=[f"{value:.3f}" for value in values], padding=3, fontsize=9)
    axis.axvline(2.5, color="#81918e", linewidth=1, linestyle="--")
    axis.text(2.54, .735, "prospective holdout", color="#5e7073", fontsize=9)
    axis.set_ylim(0, .78)
    axis.set_ylabel("nDCG@10")
    axis.set_xticks(x, names)
    axis.set_title("A frozen lexical–semantic blend improves three datasets and regresses on SciDocs",
                   loc="left", fontsize=15, fontweight="bold", color="#17343a", pad=18)
    axis.text(0, 1.01, "SciDocs prevents a universal-superiority claim; scope certification must fall back to dense retrieval.",
              transform=axis.transAxes, fontsize=10, color="#5e7073")
    axis.spines[["top", "right"]].set_visible(False)
    axis.spines[["left", "bottom"]].set_color("#9baba8")
    axis.grid(axis="y", color="#d7ddda", linewidth=.8)
    axis.set_axisbelow(True)
    axis.legend(frameon=False, ncol=3, loc="upper left")
    fig.text(.01, .015, "Source: context-stamps hybrid-retrieval-v1 · fixed α=0.75 · paired query evidence preserved",
             fontsize=8.5, color="#5e7073")
    fig.tight_layout(rect=(0, .045, 1, 1))
    fig.savefig(OUTPUT, dpi=180, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(OUTPUT)


if __name__ == "__main__":
    main()

"""Rebuild publication figures from recorded evidence; requires matplotlib."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "evidence/controller-v2"
ASSETS = ROOT / "docs/assets"
NAMES = ("scifact", "nfcorpus", "arguana", "scidocs", "fiqa")
LABELS = ("SciFact", "NFCorpus", "ArguAna", "SciDocs", "FiQA (new)")


def read(name):
    return json.loads((EVIDENCE / name).read_text(encoding="utf-8"))


def main():
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "axes.spines.top": False,
                         "axes.spines.right": False, "axes.labelcolor": "#253442", "text.color": "#253442",
                         "svg.fonttype": "none", "savefig.facecolor": "white"})
    summary, coverage, runtime = read("summary.json"), read("candidate-coverage.json"), read("online-runtime.json")
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 6.2), gridspec_kw={"width_ratios": [1.45, 1]})
    methods = [("hybrid", "Hybrid", "#728b8b"), ("student", "Selected student", "#2f617e"),
               ("teacher", "Cross-encoder", "#a17859"), ("fusion", "Fixed fusion", "#805d76")]
    y = np.arange(5)
    for j, (method, label, color) in enumerate(methods):
        delta = [summary[n][method]["delta_vs_dense"] for n in NAMES]
        means = np.asarray([d["mean"] for d in delta])
        error = np.asarray([[m - d["lower_95"] for m, d in zip(means, delta)],
                            [d["upper_95"] - m for m, d in zip(means, delta)]])
        axes[0].errorbar(means, y + (j - 1.5) * .16, xerr=error, fmt="o", markersize=4.5,
                         capsize=2, color=color, label=label, linewidth=1)
    axes[0].axvline(0, color="#8f999f", linewidth=.8)
    axes[0].set(yticks=y, yticklabels=LABELS, xlabel="Change in nDCG@10 relative to dense retrieval",
                title="Quality: paired 95% bootstrap intervals")
    axes[0].invert_yaxis()
    axes[0].legend(loc="lower left", bbox_to_anchor=(0, -.38), ncol=2, frameon=False)
    axes[0].grid(axis="x", alpha=.15)
    for j, (method, label, color) in enumerate(methods):
        times = [runtime["timings"][n][method]["p50_ms"] for n in NAMES]
        axes[1].barh(y + (j - 1.5) * .18, times, height=.15, color=color, label=label)
    axes[1].set(yticks=y, yticklabels=LABELS, xlabel="Median milliseconds (log scale)",
                title="Online retrieval + reranking")
    axes[1].set_xscale("log")
    axes[1].invert_yaxis()
    axes[1].grid(axis="x", alpha=.15)
    fig.suptitle("Context Stamps · controller-v2 evidence", x=.06, ha="left", fontsize=18, fontweight="semibold")
    fig.text(.06, .875, "3,677 test queries · four regression collections and one fresh local transfer test", fontsize=11)
    fig.text(.06, .055, "Latency: 20 queries × 3 repeats per collection; cached query vectors; CPU student / GPU teacher.\n"
             "Excludes encoding, reader generation, network and concurrent load. Intervals are descriptive, not a universal guarantee.\n"
             "Prashant Jagtap · RTX 5080 Laptop GPU · source: evidence/controller-v2", fontsize=9, color="#566570")
    fig.subplots_adjust(top=.78, bottom=.29, left=.09, right=.98, wspace=.42)
    for suffix in ("png", "svg"):
        fig.savefig(ASSETS / ("controller-v2-quality-latency." + suffix), dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12.4, 5.3))
    ax.axis("off")
    columns = ["Dataset", "Dense", "Hybrid", "Selected\nstudent", "Cross-\nencoder", "Fixed\nfusion", "Gated\npolicy"]
    rows = [[label] + [f"{summary[n][m]['ndcg10']:.4f}" for m in ("dense", "hybrid", "student", "teacher", "fusion", "gated")]
            for n, label in zip(NAMES, LABELS)]
    table = ax.table(cellText=rows, colLabels=columns, cellLoc="center", loc="center",
                     colWidths=[.19] + [.128] * 6, bbox=[.02, .21, .96, .62])
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("#d4dce0")
        cell.set_linewidth(.6)
        if row == 0:
            cell.set_facecolor("#e7eef1")
            cell.set_text_props(weight="semibold", color="#253442")
        else:
            cell.set_facecolor("#f5f7f8" if row % 2 else "white")
        if col == 0:
            cell.set_text_props(ha="left")
    fig.text(.05, .92, "Retrieval quality after the methodology changes", fontsize=19, fontweight="semibold")
    fig.text(.05, .85, "nDCG@10 · higher is better · student and policy selected without test scores", fontsize=12)
    fig.text(.05, .08, "FiQA test was locally uninspected before this round; other collections are regression checks.\n"
             "Full embeddings/text are external to the 32-byte stamp. No stamp-only or industry-wide superiority is established.\n"
             "Prashant Jagtap · github.com/jprbom/context-stamps · evidence/controller-v2", fontsize=10, color="#566570")
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    for suffix in ("png", "svg"):
        fig.savefig(ASSETS / ("controller-v2-results-table." + suffix), dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10.8, 4.8))
    old = [coverage[n]["old_candidates"]["oracle_ndcg10"] for n in NAMES]
    new = [coverage[n]["new_candidates"]["oracle_ndcg10"] for n in NAMES]
    ax.bar(y - .18, old, .34, color="#9aaeb5", label="Old union: 48 dense / 48 hybrid / 32 BM25")
    ax.bar(y + .18, new, .34, color="#2f617e", label="New union: 96 dense / 96 hybrid / 64 BM25")
    ax.set(xticks=y, xticklabels=LABELS, ylim=(0, 1.06), ylabel="Oracle nDCG@10 ceiling")
    ax.legend(loc="upper left", bbox_to_anchor=(0, 1.24), frameon=False, fontsize=10)
    ax.grid(axis="y", alpha=.15)
    fig.suptitle("Candidate expansion raises the available ranking ceiling", fontsize=17, x=.09, ha="left")
    fig.text(.09, .025, "Oracle uses relevance judgments to order the shortlist. It is an upper bound, not a model result.\n"
             "Prashant Jagtap · evidence/controller-v2/candidate-coverage.json", fontsize=9, color="#566570")
    fig.subplots_adjust(top=.72, bottom=.2, left=.09, right=.98)
    for suffix in ("png", "svg"):
        fig.savefig(ASSETS / ("controller-v2-candidate-ceiling." + suffix), dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
    for svg in ASSETS.glob("controller-v2-*.svg"):
        svg.write_text(
            "\n".join(line.rstrip() for line in svg.read_text(encoding="utf-8").splitlines()) + "\n",
            encoding="utf-8", newline="\n",
        )

"""Research figure from recorded acquisition results; no inferred token savings."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def main():
    report = json.loads((ROOT / "evidence/acquisition-v1/manifest.json").read_text())
    names = ["scifact", "nfcorpus", "arguana", "scidocs", "fiqa"]
    methods = [("full-pool", "Full pool / guarded policy", "#657888"),
               ("raw-selected-.90", "Learned stop, threshold 0.90", "#3b8b89"),
               ("fixed-32", "Fixed 32 candidates", "#c49b75")]
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "svg.fonttype": "none"})
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.8))
    fig.patch.set_facecolor("#fafbfc")
    for ax in axes:
        ax.set_facecolor("#fafbfc")
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines[["bottom", "left"]].set_color("#cbd2d8")
        ax.grid(axis="y", color="#e1e5e8", zorder=0)
    for j, (method, label, color) in enumerate(methods):
        rows = [next(r for r in report["summary"] if r["dataset"] == n and r["method"] == method) for n in names]
        x = np.arange(5) + (j - 1) * .24
        axes[0].bar(x, [r["mean_retained"] for r in rows], .22, label=label, color=color, zorder=3)
        axes[1].bar(x, [r["mean_global_judged_recall"] for r in rows], .22, color=color, zorder=3)
    for ax in axes:
        ax.set_xticks(np.arange(5), ["SciFact", "NFCorpus", "ArguAna", "SciDocs", "FiQA"])
        ax.tick_params(axis="both", length=0, pad=8)
    axes[0].set_ylabel("Mean retained candidates per query")
    axes[1].set_ylabel("Mean recall against all positive judgments")
    axes[1].set_ylim(0, 1.06)
    axes[0].set_ylim(0, 180)
    fig.suptitle("Early stopping did not transfer reliably", x=.065, ha="left", y=.98, fontsize=20, weight="bold")
    fig.text(.065, .90, "3,677 public regression queries  |  1,187-parameter predictor trained locally on RTX", color="#526270", fontsize=12)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(.5, .085), ncol=3, frameon=False)
    fig.text(.065, .03, "No calibration scope passed the 5% error gate. Guarded policy retains the full pool.\n"
             "Candidate reduction is not a token or latency measurement. Existing datasets were reused; no answer generation.",
             fontsize=10, color="#526270")
    fig.subplots_adjust(left=.07, right=.98, top=.81, bottom=.25, wspace=.26)
    base = ROOT / "docs/assets/acquisition-v1-tradeoff"
    fig.savefig(base.with_suffix(".svg"), metadata={"Creator": "Prashant Jagtap", "Date": None})
    svg = base.with_suffix(".svg")
    svg.write_text("\n".join(line.rstrip() for line in svg.read_text(encoding="utf-8").splitlines()) + "\n",
                   encoding="utf-8", newline="\n")
    fig.savefig(base.with_suffix(".png"), dpi=160, metadata={"Author": "Prashant Jagtap"})
    plt.close(fig)


if __name__ == "__main__":
    main()

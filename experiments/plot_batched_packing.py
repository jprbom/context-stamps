"""Plot measured CPU compilation and process CPU cost; no model-quality claim."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
data = json.loads((ROOT/"evidence/batched-packing-v1/summary.json").read_bytes())["groups"]
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False,
                     "axes.spines.right": False, "svg.fonttype": "none"})
fig, axes = plt.subplots(1, 3, figsize=(13, 5.3), layout="constrained")
arms = ("structure", "scoped", "relations")
for ax, metric, title, scale in zip(axes,
        ("median_seconds", "p95_seconds", "total_process_cpu_seconds"),
        ("Median CPU compilation (ms)", "p95 CPU compilation (ms)", "Summed process CPU time (s)"),
        (1000, 1000, 1)):
    for j, method in enumerate(("baseline", "batched")):
        values = [data[a][method][metric]*scale for a in arms]
        positions = [i+(j-.5)*.34 for i in range(3)]
        ax.bar(positions, values, width=.30, color=("#91A1AC", "#437F86")[j],
               label=("Sequential packing", "Batched exact packing")[j])
        for x, value in zip(positions, values):
            ax.annotate(f"{value:.0f}", (x, value), xytext=(0, 5), textcoords="offset points", ha="center", fontsize=10)
    ax.set_title(title, loc="left", pad=14)
    ax.set_xticks(range(3), ["Structural", "Page\nfilter", "Relation\npackets"])
    ax.set_ylim(0, ax.get_ylim()[1]*1.20)
    ax.grid(axis="y", alpha=.16)
    ax.set_axisbelow(True)
axes[0].legend(frameon=False, fontsize=9, loc="upper left")
fig.suptitle("Context Stamps | Exact context compilation", x=.015, ha="left", fontsize=18)
fig.supxlabel("216 unchanged prompts • 432 pairs / 864 compilations • two repeats • 8 tokenizer workers\n"
              "Includes lookup, packing and source binding; excludes initialization and model inference.\n"
              "No new answer-quality or edge-device claim • Prashant Jagtap", fontsize=10)
for suffix in ("png", "svg"):
    fig.savefig(ROOT/f"docs/assets/batched-packing-v1.{suffix}", dpi=180, facecolor="white")

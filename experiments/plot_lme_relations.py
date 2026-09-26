"""Measured page/relation comparison; no illustrative model scores."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
path = ROOT/"evidence/lme-relations-v2"
data = json.loads((path/"summary.json").read_bytes())
review = json.loads((path/"failure-review.json").read_bytes())["models"]
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False,
                     "axes.spines.right": False, "svg.fonttype": "none"})
fig, axes = plt.subplots(1, 3, figsize=(13, 5.2), layout="constrained")
models = ("qwen2.5:1.5b", "qwen2.5-coder:7b")
arms = ("structure", "scoped", "relations")
labels = ("Structural baseline", "Page filter", "Page + relation packets")
colors = ("#91A1AC", "#819781", "#437F86")
for ax, metric, title in zip(axes, ("correct", "tokens", "request"),
                            ("Native credit / 72", "Total model tokens (thousands)", "Median whole request (s)")):
    for j, arm in enumerate(arms):
        values = [data[m][arm]["correct"] if metric == "correct" else
                  (data[m][arm]["input_tokens"]+data[m][arm]["output_tokens"])/1000 if metric == "tokens" else
                  review[m]["arms"][arm]["median_request_seconds"] for m in models]
        locations = [i+(j-1)*.25 for i in range(2)]
        ax.bar(locations, values, width=.22, color=colors[j], label=labels[j])
        for x, value in zip(locations, values):
            text = f"{value:.0f}" if metric in ("correct", "tokens") else f"{value:.2f}"
            ax.annotate(text, (x, value), xytext=(0, 5), textcoords="offset points", ha="center", fontsize=10)
    ax.set_title(title, loc="left", pad=14)
    ax.set_xticks([0, 1], ["Qwen2.5\n1.5B", "Qwen2.5-Coder\n7B"])
    ax.set_ylim(0, 72 if metric == "correct" else ax.get_ylim()[1]*1.17)
    ax.grid(axis="y", alpha=.16)
    ax.set_axisbelow(True)
axes[0].legend(frameon=False, fontsize=9, loc="upper left")
fig.suptitle("Context Stamps | Page ownership and recorded relations", x=.015, ha="left", fontsize=18)
fig.supxlabel("72 inspected development questions per reader • 432 local requests • no activation qualification\n"
              "Request time includes retrieval, compilation and generation; excludes index building and model loading.\n"
              "Prashant Jagtap • research/enterprise-context", fontsize=10)
for suffix in ("png", "svg"):
    fig.savefig(ROOT/f"docs/assets/lme-relations-v2.{suffix}", dpi=180, facecolor="white")

"""Plot the measured local policy result and its equally informed fixed control."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
data = json.loads((ROOT/"evidence/ruler-local-learning-v1/summary.json").read_bytes())["holdout"]
arms = ("reader", "learned", "fixed")
labels = ("Runtime\n+ reader", "Learned\nrouting", "Fixed verified\nrule")
colors = ("#96a4ad", "#3c687a", "#859a83")
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "svg.fonttype": "none"})
fig, axes = plt.subplots(1, 3, figsize=(12.6, 5.4))
series = ([data[a]["native_complete"] for a in arms], [data[a]["model_calls"] for a in arms],
          [(data[a]["input_tokens"]+data[a]["output_tokens"])/1000 for a in arms])
titles = ("Complete native credit / 208", "Reader calls / 208 inputs", "Total model tokens (thousands)")
for ax, values, title in zip(axes, series, titles):
    bars = ax.bar(range(3), values, color=colors, width=.62)
    ax.set_xticks(range(3), labels)
    ax.set_title(title, loc="left", fontsize=11, pad=16)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.spines["bottom"].set_color("#c4cacf")
    ax.set_axisbelow(True)
    ax.grid(axis="y", color="#e6eaec", linewidth=.7)
    ax.tick_params(axis="both", length=0)
    ax.set_ylim(0, max(values)*1.17)
    for bar, value in zip(bars, values):
        label = f"{value:.1f}" if ax is axes[2] else f"{value:.0f}"
        ax.text(bar.get_x()+bar.get_width()/2, value+max(values)*.025, label, ha="center")
fig.suptitle("Return a checked result when a model call adds no value", x=.06, y=.975,
             ha="left", fontsize=17, fontweight="medium")
fig.text(.06, .875, "Local Qwen2.5 1.5B Q4_K_M · 312 training inputs → frozen CPU policy → 208 held-out inputs", color="#4a555c")
fig.text(.06, .16, "Strict output check: 143/208 reader · 188/208 learned · 187/208 fixed. Native credit is more permissive.", fontsize=9)
fig.text(.06, .105, "Learned and fixed choose identical routes. Their QA-output variation is not a learning advantage.", fontsize=9)
fig.text(.06, .05, "Candidate inactive: device-memory and source-independence qualification incomplete. No weight update or edge claim.", fontsize=9, color="#4a555c")
fig.subplots_adjust(left=.06, right=.99, top=.74, bottom=.32, wspace=.34)
for suffix in ("png", "svg"):
    fig.savefig(ROOT/f"docs/assets/ruler-local-learning-v1.{suffix}", dpi=180, facecolor="white")

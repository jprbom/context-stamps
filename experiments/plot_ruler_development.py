"""Research figure from the retained local development result."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
summary = json.loads((ROOT/"evidence/ruler-development-v1/summary.json").read_bytes())
arms = ("full", "bm25", "tool_context", "runtime")
labels = ("Full\ncontext", "BM25", "Direct tools\n+ reader", "Runtime\n+ reader")
colors = ("#8c969e", "#a4b6c1", "#78958b", "#3c687a")
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "svg.fonttype": "none"})
fig, axes = plt.subplots(1, 3, figsize=(12.6, 4.9))
series = ([100*summary["overall"][a]["native_mean"] for a in arms],
          [summary["overall"][a]["complete_credit"] for a in arms],
          [summary["overall"][a]["input_tokens"]/1000 for a in arms])
titles = ("Native substring score (%)", "Complete native credit / 52", "Total input tokens (thousands)")
for ax, values, title in zip(axes, series, titles):
    bars = ax.bar(range(4), values, color=colors, width=.65)
    ax.set_xticks(range(4), labels)
    ax.set_title(title, loc="left", pad=18, fontsize=11)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.spines["bottom"].set_color("#c4cacf")
    ax.set_axisbelow(True)
    ax.grid(axis="y", color="#e6eaec", linewidth=.7)
    ax.tick_params(axis="both", length=0)
    ax.set_ylim(0, max(values)*1.17)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x()+bar.get_width()/2, value+max(values)*.025, f"{value:.1f}" if ax is not axes[1] else f"{value:.0f}",
                ha="center", fontsize=10)
fig.suptitle("What helps a small local reader?", x=.065, y=.98, ha="left", fontsize=18, fontweight="medium")
fig.text(.065, .88, "RULER-v1 development trial · Qwen2.5 1.5B Q4_K_M · 13 tasks × 2 lengths × 2 examples", color="#4a555c")
fig.text(.065, .09, "Zero-model-call control: 42/52 complete credit; abstains on 10 inputs. Two truncated reader outputs retained.", fontsize=9)
fig.text(.065, .045, "Small, benchmark-aware trial. Native credit is not factual verification; no edge, retention or general superiority claim.", fontsize=9, color="#4a555c")
fig.subplots_adjust(left=.065, right=.99, top=.75, bottom=.25, wspace=.32)
for suffix in ("png", "svg"):
    fig.savefig(ROOT/f"docs/assets/ruler-development-v1.{suffix}", dpi=180, facecolor="white")

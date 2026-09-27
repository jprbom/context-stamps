"""Render the recorded multi-hop comparison without generated illustrations."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ARMS = ("full", "bm25", "pointwise", "diffusion")
LABELS = ("Full", "BM25", "Pointwise", "Diffusion")
NAMES = ("Qwen2.5 1.5B", "Qwen2.5-Coder 7B*")
data = [json.loads((ROOT/f"evidence/multihop-v1/{m}/summary.json").read_bytes())["arms"]
        for m in ("small", "reference")]
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "svg.fonttype": "none",
                     "axes.spines.top": False, "axes.spines.right": False})
fig, axes = plt.subplots(2, 3, figsize=(14, 8.2))
fig.subplots_adjust(top=.79, bottom=.18, left=.055, right=.975, wspace=.30, hspace=.46)
fig.patch.set_facecolor("#FAFBFC")
for ax in axes.flat:
    ax.set_facecolor("#FAFBFC")
    ax.grid(axis="y", alpha=.16)
    ax.set_axisbelow(True)
    ax.set_xticks(range(4), LABELS, rotation=14)

ax = axes[0, 0]
values = [data[0][a]["complete_support"] for a in ARMS]
bars = ax.bar(range(4), values, width=.56, color=["#A2ACB4", "#A2ACB4", "#A2ACB4", "#357A79"])
ax.bar_label(bars, padding=3)
ax.set_ylim(0, 72)
ax.set_title("A   Complete support retained / 64", loc="left", fontsize=11)

for ax, metric, title, fmt in (
    (axes[0, 1], "answer_f1", "B   Answer F1 on answerable cases", "%.3f"),
    (axes[0, 2], "unanswerable_false_positives", "C   Unsupported answers / 64", "%d"),
    (axes[1, 0], "exact_answers", "D   Exact answers / 64", "%d"),
    (axes[1, 1], "total_model_tokens", "E   Total model tokens (thousands)", "%.1f"),
    (axes[1, 2], "generation_wall_seconds", "F   Summed request time (seconds)", "%.1f"),
):
    for i, (model, name) in enumerate(zip(data, NAMES)):
        scale = 1000 if metric == "total_model_tokens" else 1
        vals = [model[a][metric]/scale for a in ARMS]
        positions = np.arange(4)+(i-.5)*.37
        bars = ax.bar(positions, vals, width=.33, color=("#357A79", "#9BAAB8")[i], label=name)
        ax.bar_label(bars, fmt=fmt, padding=3+12*i if metric == "total_model_tokens" else 3, fontsize=8)
    ax.set_ylim(0, max(d[a][metric]/scale for d in data for a in ARMS)*1.26)
    ax.set_title(title, loc="left", fontsize=11)

handles, labels = axes[0, 1].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper left", bbox_to_anchor=(.051, .865), ncols=2, frameon=False)
fig.text(.055, .952, "Local relation learning: retrieval improves, answering remains unresolved",
         fontsize=17, weight="bold", color="#26343F")
fig.text(.055, .911, "MuSiQue development subset | 64 source-separated question pairs | 1,024 measured local requests",
         fontsize=11, color="#4B5B66")
fig.text(.055, .107, "Compact arms select six paragraphs. Thirteen learned parameters; base readers unchanged. Candidate remains inactive.", fontsize=10)
fig.text(.055, .077, "*7B added after small-reader analysis on the same cases. This is not a second fresh holdout or a full benchmark submission.", fontsize=9)
fig.text(.055, .047, "Times exclude cold warmup, compilation, data preparation and training. Evidence: multihop-v1 | Prashant Jagtap", fontsize=9, color="#53616B")
for suffix in ("png", "svg"):
    fig.savefig(ROOT/f"docs/assets/multihop-v1.{suffix}", dpi=170, facecolor=fig.get_facecolor())
plt.close(fig)

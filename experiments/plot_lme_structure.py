"""Research figure from measured structural-memory development results."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
data = json.loads((ROOT/"evidence/lme-structure-v1/summary.json").read_bytes())
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "axes.spines.top": False,
                     "axes.spines.right": False, "svg.fonttype": "none"})
fig, axes = plt.subplots(1, 3, figsize=(12.8, 5.2), layout="constrained")
colors = ("#8C9DAA", "#467E80")
models = ("qwen2.5:1.5b", "qwen2.5-coder:7b")
for ax, metric, title in zip(axes, ("correct", "tokens", "median_generation_seconds"),
                            ("Native credit / 72", "Total model tokens (thousands)", "Median generation time (s)")):
    for j, arm in enumerate(("raw", "structure")):
        vals = [(data[m][arm]["input_tokens"]+data[m][arm]["output_tokens"])/1000
                if metric == "tokens" else data[m][arm][metric] for m in models]
        x = [i+(j-.5)*.32 for i in range(2)]
        ax.bar(x, vals, width=.29, color=colors[j], label="Raw chunks" if j == 0 else "Structural views")
        for loc, value in zip(x, vals):
            ax.annotate(f"{value:.0f}" if metric == "correct" else f"{value:.2f}", (loc, value),
                        xytext=(0, 5), textcoords="offset points", ha="center", fontsize=10)
    ax.set_title(title, loc="left", fontsize=12, pad=13)
    ax.set_xticks([0, 1], ["Qwen2.5\n1.5B", "Qwen2.5-Coder\n7B"])
    ax.set_ylim(0, 72 if metric == "correct" else ax.get_ylim()[1]*1.2)
    ax.grid(axis="y", alpha=.16)
    ax.set_axisbelow(True)
axes[0].legend(loc="upper left", frameon=False, fontsize=9)
fig.suptitle("Context Stamps | Recorded structure and local readers", x=.015, ha="left", fontsize=18)
fig.supxlabel("LongMemEval-V2: 72 development questions, 288 local calls. Shared histories; no activation qualification.\n"
              "Generation timing excludes preparation and loading. Native credit is not a factuality guarantee.\n"
              "Prashant Jagtap • research/enterprise-context", ha="center", fontsize=10)
for suffix in ("png", "svg"):
    fig.savefig(ROOT/f"docs/assets/lme-structure-v1.{suffix}", dpi=180, facecolor="white")

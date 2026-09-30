"""Generate a restrained research figure from the checked localization evidence."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
data = json.loads((ROOT / "evidence/repoqa-localization-v1/results.json").read_text(encoding="utf-8"))
labels = ["Train · 40", "Validation · 30", "Held-out · 30"]
splits = ["train", "validation", "final"]
methods = [("dense", "Dense MiniLM", "#335f72"),
           ("tfidf", "TF–IDF", "#829398"),
           ("stamp", "256-bit stamp", "#be7758")]
fig, (ax, cost) = plt.subplots(1, 2, figsize=(10.2, 3.6), gridspec_kw={"width_ratios": [1.6, 1]})
fig.patch.set_facecolor("#fbfaf7")
for axis in (ax, cost):
    axis.set_facecolor("#fbfaf7")
    axis.spines[["top", "right"]].set_visible(False)
    axis.spines[["left", "bottom"]].set_color("#b2b9ba")
    axis.tick_params(colors="#445052", labelsize=9, length=0)
x = np.arange(3)
for i, (key, label, color) in enumerate(methods):
    values = [data["summary"][split]["methods"][key]["top1"] /
              data["summary"][split]["queries"] * 100 for split in splits]
    bars = ax.bar(x + (i - 1) * 0.23, values, 0.19, label=label, color=color)
    ax.bar_label(bars, fmt="%.0f%%", padding=2, fontsize=8, color="#344244")
ax.set_xticks(x, labels)
ax.set_ylim(0, 61)
ax.set_ylabel("Exact function at rank 1 (%)", fontsize=9)
ax.legend(frameon=False, fontsize=8, loc="upper right")
ax.grid(axis="y", color="#dfe3e1", linewidth=0.65)
ax.set_axisbelow(True)
cost.bar([0, 1], [1536, 32], color=["#335f72", "#be7758"], width=0.46)
cost.set_xticks([0, 1], ["Float32 dense", "Stamp"])
cost.set_ylabel("Routing vector bytes per function", fontsize=9)
cost.set_ylim(0, 1800)
cost.bar_label(cost.containers[0], labels=["1,536 B", "32 B"], padding=4, fontsize=9)
cost.grid(axis="y", color="#dfe3e1", linewidth=0.65)
cost.set_axisbelow(True)
fig.suptitle("Code localization: compact routing loses recall", ha="left", x=0.055,
             fontsize=13, fontweight="semibold", color="#213d46")
fig.text(0.055, 0.015, "RepoQA-derived Python localization · 10 pinned repositories · same candidate functions · not native RepoQA or patch pass",
         fontsize=8, color="#5b6668")
fig.tight_layout(rect=(0, 0.04, 1, 0.93), w_pad=3.2)
for suffix in ("svg", "png"):
    target = ROOT / f"docs/assets/repoqa-context-localization-v1.{suffix}"
    fig.savefig(target, dpi=180, facecolor=fig.get_facecolor())
    if suffix == "svg":
        target.write_bytes(("\n".join(line.rstrip() for line in target.read_text(encoding="utf-8").splitlines())
                            + "\n").encode("utf-8"))
plt.close(fig)

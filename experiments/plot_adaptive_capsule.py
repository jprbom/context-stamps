"""Render the v0.4 design figure; values are profile allocation, not benchmark results."""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "docs" / "assets" / "adaptive-capsule.png"

labels = ["Semantic 96", "Task 32", "Entity 32", "Relation 32",
          "Temporal 16", "Authority 16", "Policy 16", "Modality 16"]
sizes = [96, 32, 32, 32, 16, 16, 16, 16]
colors = ["#2563EB", "#0EA5E9", "#14B8A6", "#22C55E",
          "#84CC16", "#F59E0B", "#F97316", "#A855F7"]

plt.rcParams.update({"font.family": "DejaVu Sans"})
fig = plt.figure(figsize=(16, 9), facecolor="#F8FAFC")
ax = fig.add_axes([0, 0, 1, 1])
ax.set_axis_off()
fig.text(.06, .91, "Spherical Context Capsule", fontsize=27, fontweight="bold", color="#0F172A")
fig.text(.06, .865, "Inspectable facets, a fixed 256-bit routing profile, and risk-controlled escalation",
         fontsize=14, color="#475569")

pie = fig.add_axes([.05, .16, .42, .64], facecolor="#F8FAFC")
pie.pie(sizes, colors=colors, startangle=90, counterclock=False,
        wedgeprops={"width": .34, "edgecolor": "white", "linewidth": 2})
pie.text(0, .08, "256 bits", ha="center", va="center", fontsize=24, fontweight="bold", color="#0F172A")
pie.text(0, -.14, "32 raw bytes", ha="center", va="center", fontsize=13, color="#475569")
pie.legend(labels, loc="upper center", bbox_to_anchor=(.5, .02), ncol=2,
           frameon=False, fontsize=10)
pie.set_aspect("equal")

def box(x, y, w, h, title, body, color):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.012,rounding_size=0.018",
                                facecolor="white", edgecolor=color, linewidth=2))
    ax.text(x + .02, y + h - .035, title, fontsize=14, fontweight="bold", color="#0F172A", va="top")
    ax.text(x + .02, y + h - .082, body, fontsize=11, color="#475569", va="top", linespacing=1.45)

box(.52, .66, .40, .14, "1  Compile observed facets",
    "Every emitted value records its source and rule.\nMissing query facets stay absent.", "#0EA5E9")
box(.52, .45, .40, .14, "2  Bind scope and calibration",
    "Domain, schema, facet mask and weights must match.\nThe policy carries a validation count and error bound.", "#14B8A6")
box(.52, .24, .40, .14, "3  Choose the safest eligible route",
    "Exact ID  →  calibrated compact exit  →  precise retrieval\nThe router abstains when an explicit ID is unavailable.", "#F59E0B")
ax.annotate("", xy=(.72, .65), xytext=(.72, .60), arrowprops={"arrowstyle": "-|>", "color": "#64748B", "lw": 2})
ax.annotate("", xy=(.72, .44), xytext=(.72, .39), arrowprops={"arrowstyle": "-|>", "color": "#64748B", "lw": 2})
fig.text(.52, .13, "Boundary", fontsize=11, fontweight="bold", color="#DC2626")
fig.text(.59, .13, "The capsule routes to evidence; it does not contain or authenticate the evidence.",
         fontsize=11, color="#475569")
fig.text(.06, .025, "Context Stamps v0.4 research design · Prashant Jagtap · allocation is an ablation target, not a measured optimum",
         fontsize=9.5, color="#64748B")
OUTPUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUTPUT, dpi=150, facecolor=fig.get_facecolor(), bbox_inches="tight")
print(OUTPUT)

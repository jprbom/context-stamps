"""Render the sanitized coding-context development result as a research figure.

Copyright (c) 2026 Prashant Jagtap. MIT License.
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def pair(record):
    rows = {row["arm"]: row for row in record["rows"]}
    if set(rows) != {"baseline", "stamp"}:
        raise ValueError("paired baseline and stamp rows required")
    return rows


def plot(source, png, svg):
    data = json.loads(source.read_text(encoding="utf-8"))
    panels = [("16-step shell agent", pair(data["agent_attempt_2"])),
              ("One-shot code generation", pair(data["code_generation_replay"]))]
    colors = {"baseline": "#56677A", "stamp": "#2A7F88"}
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.8), dpi=160)
    for ax, (title, rows) in zip(axes, panels):
        labels = ("Direct source", "Stamp + relations") if title.startswith("16-step") else ("Direct", "Stamp closure")
        values = [rows["baseline"]["input_tokens"], rows["stamp"]["input_tokens"]]
        ax.barh(labels[::-1], values[::-1], color=[colors["stamp"], colors["baseline"]], height=.48)
        ax.set_xlim(0, max(values) * 1.08)
        ax.set_title(title, loc="left", fontsize=11, fontweight="bold", pad=16)
        ax.set_xlabel("Model input tokens", fontsize=9, color="#31404E")
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x / 1000:g}k" if x >= 1000 else f"{x:g}"))
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.spines["bottom"].set_color("#CFD7DD")
        ax.tick_params(axis="both", length=0, labelsize=9, colors="#31404E")
        ax.grid(axis="x", color="#E8EDF0", linewidth=.7)
        ax.set_axisbelow(True)
        for y, arm in enumerate(("stamp", "baseline")):
            row = rows[arm]
            ax.text(row["input_tokens"] * .97, y,
                    f"{row['input_tokens']:,}  ·  pass {int(row['passed'])}/1",
                    ha="right", va="center", fontsize=8.4, fontweight="bold", color="white")
    fig.suptitle("Coding-context development check", x=.06, y=.99, ha="left",
                 fontsize=13, fontweight="bold", color="#1F303A")
    fig.text(.06, .01, "One selected Terminal-Bench task · Qwen2.5-Coder 7B · neither arm passed native verification",
             fontsize=8.5, color="#52616D")
    fig.subplots_adjust(left=.18, right=.98, top=.78, bottom=.25, wspace=.55)
    png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png, dpi=180, facecolor="white")
    fig.savefig(svg, facecolor="white")
    plt.close(fig)
    # Matplotlib writes SVG path commands with trailing spaces. Normalize the
    # research artifact so the repository's whitespace check remains useful.
    svg.write_text("\n".join(line.rstrip() for line in svg.read_text(encoding="utf-8").splitlines()) + "\n",
                   encoding="utf-8", newline="\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--png", type=Path, required=True)
    parser.add_argument("--svg", type=Path, required=True)
    args = parser.parse_args()
    plot(args.source, args.png, args.svg)

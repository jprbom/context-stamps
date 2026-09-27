"""Research figure from retained TechQA quality and cost measurements.

Copyright (c) 2026 Prashant Jagtap. MIT License.
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot(analysis, prefix):
    data = json.loads(Path(analysis).read_bytes())
    if data["policy_activation"] or data["selection"]["mode"] != "fixed_citation_check":
        raise ValueError("figure describes the recorded inactive fixed-policy result")
    names = ("direct_raw", "direct_exact_span", "cited_raw", "cited_checked", "abstain_all")
    labels = ("Direct answer", "Direct + exact span", "Quoted answer", "Quoted + source check", "Always abstain")
    colors = ("#52758E", "#52758E", "#B27957", "#B27957", "#9B9E9E")
    metrics = data["metrics"]
    pos, neg = metrics["direct_raw"]["positive_count"], metrics["direct_raw"]["negative_count"]
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "svg.fonttype": "none",
                         "svg.hashsalt": "context-stamps-techqa-v1", "axes.spines.top": False,
                         "axes.spines.right": False, "axes.titleweight": "normal"})
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.4), gridspec_kw={"height_ratios": [1.4, 1]})
    fig.patch.set_facecolor("#FAFAF7")
    fig.suptitle("Local technical QA: answer quality, abstention and cost", x=.08, ha="left", y=.965, fontsize=17)
    fig.text(.08, .916, f"Qwen3.5:4b · {pos+neg} public development questions · learned policies rejected on calibration",
             color="#4F595E", fontsize=10)
    for ax in axes.flat:
        ax.set_facecolor("#FAFAF7")
        ax.tick_params(axis="both", length=0, labelcolor="#26353D")
        ax.spines["left"].set_visible(False)
        ax.spines["bottom"].set_color("#CCD1D1")
        ax.set_axisbelow(True)
    left, right = axes[0]
    quality = [metrics[name]["positive_f1"]*100 for name in names]
    false_positives = [metrics[name]["false_positive_count"]*100/neg for name in names]
    for ax, values in ((left, quality), (right, false_positives)):
        ax.barh(range(5), values, color=colors, height=.55)
        ax.set_yticks(range(5), labels if ax is left else [""]*5)
        ax.invert_yaxis()
        ax.grid(axis="x", color="#DEE2E1", lw=.7)
    left.set_title(f"Answerable cases: character-span F1  (n={pos})", loc="left", pad=12, fontsize=11)
    left.set_xlabel("F1 (%) · higher is better")
    left.set_xlim(0, max(10, max(quality)*1.4))
    right.set_title(f"Unanswerable cases: false positives  (n={neg})", loc="left", pad=12, fontsize=11)
    right.set_xlabel("False-positive rate (%) · lower is better")
    right.set_xlim(0, 115)
    for i, name in enumerate(names):
        left.text(quality[i]+left.get_xlim()[1]*.015, i, f"{quality[i]:.2f}", va="center", fontsize=9)
        right.text(false_positives[i]+1.5, i, f"{metrics[name]['false_positive_count']}/{neg}", va="center", fontsize=9)
    costs = [data["costs"][mode] for mode in ("direct", "cited")]
    tokens = [(r["input_tokens"]+r["output_tokens"])/1e6 for r in costs]
    axes[1, 0].barh([0, 1], tokens, color=[colors[0], colors[2]], height=.48)
    axes[1, 0].set_yticks([0, 1], ["Direct interface", "Quoted interface"])
    axes[1, 0].invert_yaxis()
    axes[1, 0].set_title("Actual model tokens · all 310 requests per interface", loc="left", pad=12, fontsize=11)
    axes[1, 0].set_xlabel("Input + output tokens (millions)")
    axes[1, 0].set_xlim(0, max(tokens)*1.3)
    for i, value in enumerate(tokens):
        axes[1, 0].text(value+max(tokens)*.025, i, f"{value:.3f}M", va="center", fontsize=9)
    medians, tails = [r["request_wall_median"] for r in costs], [r["request_wall_p95"] for r in costs]
    ax = axes[1, 1]
    for i, (median, tail) in enumerate(zip(medians, tails)):
        ax.plot([median, tail], [i, i], color=colors[i*2], lw=2)
        ax.scatter([median], [i], color=colors[i*2], s=50)
        ax.scatter([tail], [i], color=colors[i*2], marker="|", s=140)
        ax.text(tail+max(tails)*.025, i, f"{median:.1f} / {tail:.1f}s", va="center", fontsize=9)
    ax.set_yticks([0, 1], ["Direct", "Quoted"])
    ax.set_ylim(1.55, -.55)
    ax.set_xlim(0, max(tails)*1.5)
    ax.set_title("Measured loopback request time", loc="left", pad=12, fontsize=11)
    ax.set_xlabel("Seconds · dot = median, tick = p95")
    for ax in axes[1]:
        ax.grid(axis="x", color="#DEE2E1", lw=.7)
    fig.subplots_adjust(left=.235, right=.965, top=.84, bottom=.17, hspace=.7, wspace=.28)
    fig.text(.08, .082, "Filters run after generation and retain its token cost. Exact quotation does not establish semantic correctness.", fontsize=9, color="#4F595E")
    fig.text(.08, .052, "RTX laptop; CPU preparation/tests also ran on the host. Shared compilation, warmups and limits are reported separately.", fontsize=9, color="#4F595E")
    fig.text(.08, .02, "Prashant Jagtap · Context Stamps research · no candidate activated", fontsize=9, color="#4F595E")
    prefix = Path(prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(prefix.with_suffix(".png"), dpi=180, facecolor=fig.get_facecolor())
    fig.savefig(prefix.with_suffix(".svg"), metadata={"Date": None}, facecolor=fig.get_facecolor())
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis", type=Path, required=True)
    parser.add_argument("--prefix", type=Path, required=True)
    args = parser.parse_args()
    plot(args.analysis, args.prefix)

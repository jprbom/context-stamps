"""Research figure from the retained native text-memory development results."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
data = json.loads((ROOT/"evidence/lme-memory-v1/summary.json").read_bytes())["holdout"]
arms = ("none", "state", "linked", "learned")
labels = ("No memory", "State retrieval", "Linked memory", "Learned choice")
colors = ("#9aa4ad", "#4f728b", "#57958f", "#b19668")
fig, axes = plt.subplots(1, 2, figsize=(12.8, 6.5), dpi=160)
fig.patch.set_facecolor("#fafbf9")
for ax in axes:
    ax.set_facecolor("#fafbf9")
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#bcc6cc")
    ax.tick_params(colors="#344450", labelsize=9)
    ax.set_axisbelow(True)
    ax.grid(axis="y", color="#e3e8e8", linewidth=.7)
accuracy = [100*data[a]["correct"]/data[a]["n"] for a in arms]
bars = axes[0].bar(labels, accuracy, color=colors, width=.62)
axes[0].set_ylabel("Native deterministic credit (%)", fontsize=10)
axes[0].set_ylim(0, max(accuracy)*1.4+5)
axes[0].set_title("Answer quality on 222 held-out questions", loc="left", fontsize=12, pad=20)
for bar, a in zip(bars, arms):
    axes[0].text(bar.get_x()+bar.get_width()/2, bar.get_height()+1,
                 f"{data[a]['correct']}/{data[a]['n']}", ha="center", color="#243746", fontsize=10)
tokens = [(data[a]["input_tokens"]+data[a]["output_tokens"])/1000 for a in arms]
bars = axes[1].bar(labels, tokens, color=colors, width=.62)
axes[1].set_ylabel("Total reader input + output tokens (thousands)", fontsize=10)
axes[1].set_ylim(0, max(tokens)*1.3)
axes[1].set_title("Reader cost for the same question set", loc="left", fontsize=12, pad=20)
for bar, value in zip(bars, tokens):
    axes[1].text(bar.get_x()+bar.get_width()/2, bar.get_height()+max(tokens)*.025,
                 f"{value:,.1f}k", ha="center", color="#243746", fontsize=10)
fig.suptitle("Context Stamps | Local workflow memory", x=.07, ha="left", y=.97, fontsize=19, color="#243746")
fig.text(.07, .88, "LongMemEval-V2 Small, text-only subset · Qwen2.5 1.5B Q4_K_M · 72 training questions", fontsize=10, color="#53636d")
fig.text(.07, .075, "Learned choice reuses its selected measured arm. Shared source histories; no independent retention qualification.\n"
         "128 judge-dependent and 29 image questions excluded. Native credit is not a factuality certificate.\n"
         "Prashant Jagtap · Public code and evidence: github.com/jprbom/context-stamps (research/enterprise-context)", fontsize=9, color="#53636d", linespacing=1.6)
fig.subplots_adjust(left=.07, right=.97, bottom=.25, top=.77, wspace=.25)
for extension in ("png", "svg"):
    fig.savefig(ROOT/f"docs/assets/lme-memory-v1.{extension}", facecolor=fig.get_facecolor())

"""Research figures from measured runtime evidence; no generated illustration."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs/assets"
EVIDENCE = ROOT / "evidence/runtime-v1"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False,
                     "axes.spines.right": False, "svg.fonttype": "none", "figure.facecolor": "white"})


def read(name):
    return json.loads((EVIDENCE / name).read_text(encoding="utf-8"))


def save(fig, name):
    for suffix in ("png", "svg"):
        path = OUT / f"runtime-v1-{name}.{suffix}"
        fig.savefig(path, dpi=180)
        if suffix == "svg":
            path.write_text("\n".join(line.rstrip() for line in path.read_text(encoding="utf-8").splitlines()) + "\n", encoding="utf-8", newline="\n")
    plt.close(fig)


def main():
    latency = read("latency-summary.json")
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 5.4))
    methods = ("reference_fusion", "accelerated_cold", "accelerated_warm")
    for ax, name in zip(axes, ("scifact", "fiqa")):
        data = latency[name]
        medians = [data["latency"][m]["p50_ms"] for m in methods]
        p95 = [data["latency"][m]["p95_ms"] for m in methods]
        y = np.arange(3)
        ax.barh(y, medians, color=["#9aaeb5", "#649393", "#2f617e"], height=.55)
        ax.scatter(p95, y, marker="|", s=140, color="#263540", label="p95")
        for i, (median, tail) in enumerate(zip(medians, p95)):
            ax.text(tail + 3, i, f"{median:.1f} / {tail:.1f}", va="center", fontsize=10)
        ax.set(yticks=y, yticklabels=["Original fusion", "Optimized · cold", "Optimized · warm"],
               xlabel="Milliseconds · labels: median / p95", xlim=(0, max(p95) * 1.44),
               title=f"{ {'scifact': 'SciFact', 'fiqa': 'FiQA'}[name]} · nDCG@10 {data['runtime_ndcg']:.4f}")
        ax.invert_yaxis()
        ax.grid(axis="x", alpha=.15)
    fig.suptitle("Lower reranking latency with the measured rankings preserved", x=.055, ha="left", fontsize=17)
    fig.text(.055, .855, "Same candidates, model weights and 512-token pair limit · length-aware batches and exact token reuse", fontsize=10)
    fig.text(.055, .045, "Cold/warm refers to the passage-token cache. Query embeddings and model are warm; reader, network and index builds excluded.\n"
             "20 queries × 3 measured repeats per collection · RTX 5080 Laptop GPU · Prashant Jagtap · evidence/runtime-v1", fontsize=9, color="#566570")
    fig.subplots_adjust(left=.17, right=.98, top=.73, bottom=.22, wspace=.65)
    save(fig, "latency")

    fig, ax = plt.subplots(figsize=(12.4, 5.3))
    ax.axis("off")
    rows = []
    for name, label in (("scifact", "SciFact"), ("nfcorpus", "NFCorpus"), ("arguana", "ArguAna"), ("scidocs", "SciDocs"), ("fiqa", "FiQA")):
        data = latency[name]
        rows.append([label, f"{data['reference_fusion_ndcg']:.4f} = {data['accelerated_fusion_ndcg']:.4f}",
            f"{data['queries']-data['top10_mismatches']}/{data['queries']}"]
            + [f"{data['latency'][m]['p50_ms']:.2f}" for m in methods])
    table = ax.table(cellText=rows, colLabels=["Dataset", "nDCG@10\nOriginal = optimized", "Identical\ntop-10", "Original\np50 ms", "Cold cache\np50 ms", "Warm cache\np50 ms"],
        cellLoc="center", colWidths=[.15, .24, .15, .15, .15, .16], bbox=[.015, .21, .97, .62])
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("#d4dce0")
        cell.set_linewidth(.6)
        cell.set_facecolor("#e7eef1" if row == 0 else "#f5f7f8" if row % 2 else "white")
        if row == 0:
            cell.set_text_props(weight="bold")
    fig.text(.045, .92, "Reranking execution: preserved rankings and measured latency", fontsize=18)
    fig.text(.045, .06, "All 3,677 queries replayed; all collections previously inspected. Full-fusion diagnostics shown; deployed scope choices unchanged.\n"
             "Timing: 20 queries × 3 repeats, warm model/query vectors; cold/warm refers to passage tokens. Reader and network excluded.\n"
             "Prashant Jagtap · RTX 5080 Laptop GPU · evidence/runtime-v1", fontsize=9, color="#566570")
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    save(fig, "table")

    workflow = read("workflow-summary.json")
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 5.4))
    modes = ("full_scope", "prepared_context", "verified_reuse")
    for ax, (name, data) in zip(axes, workflow.items()):
        counts = [data[m]["prompt_tokens"] for m in modes]
        ax.bar(range(3), counts, color=["#9aaeb5", "#649393", "#2f617e"], width=.6)
        for i, mode in enumerate(modes):
            ax.text(i, counts[i] + max(counts) * .025, f"{counts[i]:,}\n{data[mode]['verified']}/48 verified", ha="center", fontsize=10)
        ax.set(xticks=range(3), xticklabels=["Full scope", "Prepared\ncontext", "Verified\nreuse"], ylim=(0, max(counts) * 1.26),
               title=name, ylabel="Server-reported input tokens · 48 requests")
        ax.grid(axis="y", alpha=.15)
    fig.suptitle("Two local readers on the same research-configuration fixture", x=.07, ha="left", fontsize=17)
    fig.text(.07, .03, "12 fictional tasks × initial/repeat/edit/repeat · 50% repeated requests by design · one timing run; model cold load excluded.\n"
             "Exact parser/calculator control: 48/48 verified, zero model calls. Reader weights are unchanged; their failures remain.\n"
             "This is not an open-ended coding or production benchmark. Prashant Jagtap · evidence/runtime-v1", fontsize=9, color="#566570")
    fig.subplots_adjust(left=.09, right=.98, top=.77, bottom=.24, wspace=.25)
    save(fig, "readers")

    ablation = read("spherical-ablation.json")
    keys = ["semantic_float", "semantic_256", "facets_float", "facets_256", "without_relation", "without_entity", "exact_metadata"]
    labels = ["Semantic · float", "Semantic · 32 B", "Full facets · float", "Full facets · 32 B", "Without relation", "Without entity", "Exact metadata"]
    fig, ax = plt.subplots(figsize=(10.5, 5.2))
    counts = [ablation["results"][k] for k in keys]
    ax.barh(range(7), counts, color=["#9aaeb5", "#9aaeb5", "#649393", "#2f617e", "#9aaeb5", "#9aaeb5", "#b99568"])
    for i, count in enumerate(counts):
        ax.text(count + 1, i, f"{count}/80", va="center")
    ax.set(yticks=range(7), yticklabels=labels, xlim=(0, 95), xlabel="Correct top-1 selections")
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=.15)
    fig.suptitle("What declared facets contribute in an ambiguous synthetic fixture", x=.06, ha="left", fontsize=16)
    fig.text(.06, .04, "Same eight candidates and top-1 budget. Every distractor differs in one declared facet; semantic text is deliberately identical.\n"
             "Exact metadata ties the full stamp. External source/graph storage is additional. No real-world superiority established.\n"
             "Prashant Jagtap · evidence/runtime-v1/spherical-ablation.json", fontsize=9, color="#566570")
    fig.subplots_adjust(left=.23, right=.95, top=.83, bottom=.26)
    save(fig, "facets")


if __name__ == "__main__":
    main()

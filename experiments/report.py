"""Render tables and scientific plots directly from recorded experiment outputs."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def run():
    root = Path(__file__).resolve().parents[1]
    public = json.loads((root / "evidence/scifact-v1/summary.json").read_text())
    synthetic = json.loads((root / "evidence/synthetic-v1/summary.json").read_text())
    manifest = json.loads((root / "evidence/scifact-v1/manifest.json").read_text())
    text = """# Recorded experiments

These are executed measurements, including negative results. They are not claims of general agent correctness, prompt-injection resistance or superiority over complete memory platforms. [Reproduction](#reproduce) and [model cards](models.md) explain the scope.

## Public data: BEIR SciFact

5,183 documents and all 300 official test queries. Pinned MiniLM, CPU execution. Trained selectors use official training labels with relevant-document group separation from test. Coverage weight is selected on a separate validation split. Neural rerankers and selection methods use the top 100 dense candidates; binary reranking uses the top 100 binary candidates. See the [run manifest](../evidence/scifact-v1/manifest.json).

| Method | nDCG@10 | Recall@10 | Recall after 8,192-byte packing | Paired nDCG difference from dense: 95% interval |
|---|---:|---:|---:|---|
"""
    for row in public:
        low, high = row["paired_delta_vs_dense_ci95"]
        text += f"| {row['method']} | {row['ndcg10']:.4f} | {row['recall10']:.4f} | {row['packed_recall']:.4f} | [{low:+.4f}, {high:+.4f}] |\n"
    text += """
![Measured SciFact retrieval comparison](assets/scifact-results.png)

Intervals use 2,000 paired query-bootstrap resamples. Queries sharing documents may be correlated; these intervals are not a substitute for independent-dataset replication. No multiplicity-adjusted hypothesis test is claimed. A zero interval for the raw diallel control reflects the expected ranking equivalence. Original and shared-library replication summaries are both retained. See [per-query rankings and metrics](../evidence/scifact-v1/per-query.jsonl).

Coverage/diversity improved this dataset's retrieval metric; the small trained rerankers did not beat the dense baseline. This justifies exposing the deterministic candidate selector, while keeping trained models experimental. BM25 remains a strong comparator. The byte-packing metric is whole-document retention, not actual model-token usage or generated-answer accuracy.

### Resource measurements

"""
    text += f"Raw dense float32 vectors: **{manifest['raw_dense_vector_bytes']:,} bytes**; raw 128-bit codes: **{manifest['raw_binary_code_bytes']:,} bytes**. This excludes text, metadata and model weights. Binary-plus-dense reranking still retains dense vectors.\n\n"
    for method, latency in manifest["index_search_ms_per_query"].items():
        text += f"- {method}: {latency:.4f} ms/query, batched top-100 index search, one Faiss thread.\n"
    text += """
These index timings exclude embedding and packing. BM25 is a transparent Python reference, not an optimized production index; its speed is not used to claim superiority. The manifest records encoding/cache time and combined-process RSS. RSS is not the incremental memory of any single method. No energy, ARM/mobile or real SLM-generation measurement is included.

## Synthetic changing-evidence fixtures

Original fictional code, support, edge-device, policy and API scenarios. There are 150 training, 45 validation, 90 same-domain test and 60 held-out-domain cases. Cases include repeats, stale values, changed instructions, missing facts, tight budgets and malicious-looking text. Metrics count annotated required facts; no generator is called.

| Split | Method | Complete fact coverage | Stale selected | Abstention |
|---|---|---:|---:|---:|
"""
    for row in synthetic:
        text += f"| {row['split']} | {row['method']} | {row['success']:.3f} | {row['stale_selected']:.3f} | {row['abstained']:.3f} |\n"
    text += """
The required-source control is deliberately given trusted essential source IDs. Its benefit demonstrates enforcement of caller requirements, not automatic discovery of necessary evidence. Tight-budget and missing-evidence cases are intentionally unsatisfiable. When a required source cannot be supplied, the runtime returns an empty packet and an explicit insufficient-evidence status.

The fixtures expose weak learned generalization. Three relevance-model seeds and a restoration-label model are included even when they lose. Freshness filtering blocks declared stale items, but retrieval can still select adversarial text: the `injection_selected` field makes that limitation visible. No prompt-injection protection is claimed. Coverage/diversity settings are selected on validation, not test. This is exploratory research with template-related data, not a blinded product acceptance trial.

## Evidence files

- [Synthetic manifest](../evidence/synthetic-v1/manifest.json), [fixtures](../evidence/synthetic-v1/fixtures.jsonl), [per-case decisions](../evidence/synthetic-v1/per-case.jsonl), [omission/restoration interventions](../evidence/synthetic-v1/interventions.jsonl), [training losses](../evidence/synthetic-v1/training.json).
- [SciFact manifest](../evidence/scifact-v1/manifest.json), [per-query rankings](../evidence/scifact-v1/per-query.jsonl), [training losses](../evidence/scifact-v1/training.json), [initial summary](../evidence/scifact-v1/initial-run-summary.json).
- Selector JSON exports and licenses accompany each experiment. No raw SciFact corpus, private chat, private Cortex data or downloaded encoder weights are included.

## Reproduce

Run from a repository checkout:

```bash
python -m pip install -e ".[learn]"
python examples/changing_evidence.py
python experiments/run_evidence.py --out evidence/synthetic-reproduction

python -m pip install -e ".[experiment]"
python experiments/run_scifact.py --data /path/to/BEIR/scifact --cache /path/to/local-cache --out evidence/scifact-reproduction
python experiments/report.py
```

Obtain SciFact separately from the [BEIR source](https://huggingface.co/datasets/BeIR/scifact) and observe its license. The directory must contain corpus.jsonl, queries.jsonl and qrels/train.tsv and test.tsv. Model loading uses the immutable revision in the script. The report command renders the checked-in v1 evidence; use those paths for a fresh replacement report. Timings vary by host. Compare metric values and data/model hashes before comparing speed.

## Remaining validation

Real coding-agent task completion, live SLM answer faithfulness, large stores, independent datasets, calibrated uncertainty and edge-device energy measurements remain unestablished. Changes should be evaluated against these records without selecting only favorable methods or seeds.
"""
    (root / "docs/experiments.md").write_text(text, encoding="utf-8")
    display = [
        r
        for r in public
        if r["method"]
        in {
            "bm25_reference",
            "faiss_dense",
            "faiss_binary128",
            "binary100_dense_rerank",
            "classical_mmr",
            "coverage_diversity",
            "trained_linear_7",
        }
    ]
    display.sort(key=lambda r: r["ndcg10"])
    fig, ax = plt.subplots(figsize=(10, 5.5), layout="constrained")
    labels = [r["method"].replace("_", " ") for r in display]
    ax.barh(
        labels,
        [r["ndcg10"] for r in display],
        color=["#087f8c" if r["method"] == "coverage_diversity" else "#65758b" for r in display],
    )
    for index, row in enumerate(display):
        ax.text(row["ndcg10"] + 0.01, index, f"{row['ndcg10']:.3f}", va="center")
    ax.set_xlim(0, 0.82)
    ax.set_xlabel("nDCG@10 · higher is better")
    ax.set_title(
        "SciFact: 300 test queries, same MiniLM encoder\nCoverage/diversity helps here; trained linear reranking does not",
        loc="left",
        fontsize=13,
    )
    ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(root / "docs/assets/scifact-results.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    run()

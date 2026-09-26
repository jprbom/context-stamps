"""Render evidence-linked result tables and plain-text publication drafts."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "evidence/runtime-v1"


def read(name):
    return json.loads((EVIDENCE / name).read_text(encoding="utf-8"))


def write(path, value):
    path.write_text(value, encoding="utf-8", newline="\n")


def main():
    latency, workflow = read("latency-summary.json"), read("workflow-summary.json")
    fiqa = latency["fiqa"]
    before = fiqa["latency"]["reference_fusion"]["p50_ms"]
    cold = fiqa["latency"]["accelerated_cold"]["p50_ms"]
    warm = fiqa["latency"]["accelerated_warm"]["p50_ms"]
    parity = sum(d["queries"] - d["top10_mismatches"] for d in latency.values())
    table = "| Dataset | Fusion nDCG@10 | Optimized nDCG@10 | Identical top-10 | Existing policy retained |\n|---|---:|---:|---:|---:|\n"
    timing = "| Dataset | Original fusion p50 / p95 ms | Optimized cold p50 / p95 ms | Optimized warm p50 / p95 ms |\n|---|---:|---:|---:|\n"
    for name in ("scifact", "nfcorpus", "arguana", "scidocs", "fiqa"):
        d = latency[name]
        table += f"| {name} | {d['reference_fusion_ndcg']:.4f} | {d['accelerated_fusion_ndcg']:.4f} | {d['queries']-d['top10_mismatches']}/{d['queries']} | {d['runtime_ndcg']:.4f} |\n"
        timing += "| " + name + " | " + " | ".join(f"{d['latency'][m]['p50_ms']:.2f} / {d['latency'][m]['p95_ms']:.2f}" for m in ("reference_fusion", "accelerated_cold", "accelerated_warm")) + " |\n"
    reader_table = "| Local reader | Mode | Verified | Model calls | Reported input tokens | p50 / p95 ms |\n|---|---|---:|---:|---:|---:|\n"
    reader_lines = []
    for model, modes in workflow.items():
        for mode in ("full_scope", "prepared_context", "verified_reuse"):
            d = modes[mode]
            reader_table += f"| {model} | {mode} | {d['verified']}/{d['requests']} | {d['calls']} | {d['prompt_tokens']:,} | {d['p50_ms']:.1f} / {d['p95_ms']:.1f} |\n"
        base, prepared, reused = modes["full_scope"], modes["prepared_context"], modes["verified_reuse"]
        reader_lines.append(f"{model}: full scope used {base['prompt_tokens']:,} reported input tokens with {base['verified']}/48 verified answers; "
            f"prepared context used {prepared['prompt_tokens']:,} with {prepared['verified']}/48; verified reuse used {reused['prompt_tokens']:,} "
            f"and {reused['calls']} model calls with {reused['verified']}/48.")
    results = f"""# Runtime-v1: lower execution cost and bounded workflow composition

By Prashant Jagtap · 26 September 2026

The corrected optimized cross-encoder preserves **{parity:,}/3,677 top-10 rankings** against the frozen full-fusion reference across five previously inspected public collections. The deployed scope decisions remain unchanged: fusion for SciFact/FiQA and dense for NFCorpus/ArguAna/SciDocs. FiQA remains **0.4125 nDCG@10**, versus the earlier dense 0.3687 and hybrid 0.3888 controls.

In the paired local FiQA timing run, full fusion falls from **{before:.2f} ms median to {cold:.2f} ms with a cold passage-token cache and {warm:.2f} ms warm**. Warm execution is {before/warm:.2f}× faster ({100*(1-warm/before):.1f}% less stage latency). This is retrieval/reranking execution on the same workload, not a new model-quality or universal speed result. Dense retrieval remains cheaper.

## What changed

The scorer retains FP32 weights and BF16 autocast, the full candidate union and the 512-token pair limit. Stable length sorting reduces padded work. A bounded exact-content cache avoids retokenizing passages. Long queries delegate full pair truncation to the pinned tokenizer. No relevant candidate is dropped for speed, and no shorter evidence limit is substituted.

The first candidate incorrectly handled the odd token in longest-first truncation when both sequences were long. It changed 231 ArguAna top-10 rankings and was rejected there. [Initial evidence and source](../evidence/runtime-v1/initial-candidate), [correction](../evidence/runtime-v1/correction.json). The corrected implementation was rerun across every query and timing scope. Small logit differences can still occur from GPU batching; ranking parity is an observed result on this regression suite, not bitwise equivalence on every future input.

Full FP16/BF16 weight conversion was faster in the seven-query development pilot but changed scores, so it was not promoted. Previous whole-layer student int8 was slower on the measured CPU. Precision is placed where the measured accuracy and operator behavior support it; lower bit width alone does not establish a speed benefit. See [Sentence Transformers inference guidance](https://www.sbert.net/docs/cross_encoder/usage/efficiency.html), [PyTorch numerical behavior](https://docs.pytorch.org/docs/main/notes/numerical_accuracy.html) and [tokenizer truncation semantics](https://huggingface.co/docs/transformers/main/pad_truncation).

## Quality and latency

{table}
{timing}
![Paired latency comparison](assets/runtime-v1-latency.png)

![Uploadable numeric comparison](assets/runtime-v1-table.png)

Each timing cell uses 20 evenly spaced queries with three measured repeats after warmup. Methods rotate in order and GPU work is synchronized. Cold/warm refers only to passage token IDs. Query encoding, model/index cold load, downstream generation, network and concurrent load are excluded. Full-fusion timings on scopes that retain dense are diagnostic, not their deployed cost. The historical four collections and FiQA are all regression data now; this optimization round has no new holdout.

## Expert routing was trained but did not qualify

A ridge model learns hybrid-minus-fusion relevance benefit from 14 observable candidate statistics, using 3,134 SciFact/NFCorpus training queries and equal domain weights. A fixed regularization coefficient of 10 is used. Threshold selection uses 290 tuning queries and requires nonnegative mean benefit in each tuning domain. A separate calibration gate requires at least 30 cheap exits, nonnegative mean benefit and a corrected lower gain bound of at least −0.001 against existing fusion.

The selected threshold, 0.01, produced only five proposed cheap exits on FiQA calibration, with mean change −0.000374. SciFact had none. Both failed; **no cheap-route scope is enabled**. The latency improvement comes from execution changes, not an unqualified retrieval shortcut. The int16-rounded router coefficient experiment preserves boundary decisions through a rounding-error bound and FP64 fallback. It retains original coefficients and is not an integer-kernel speed claim.

## Unified runtime and local readers

`ContextRuntime` composes `prepare_context`, `resolve`, `invalidate`, `record_outcome` and `run_verified`. It checks source/dependency revisions, roles, whole-packet byte/token budgets, bounded missing-evidence recovery, no-progress conditions and host-owned result verification. It supports explicit registered experts and exact result reuse. It neither updates its weights during inference nor executes plans found in context.

The two-reader fixture uses 12 fictional research configurations, each with initial/repeat/source-edit/repeat events. All modes see the same task and correct bindings. Full-scope input includes ten unrelated project notes. Prepared context keeps the target dependency closure; verified reuse additionally caches only previously verified exact computations. Readers are the pinned local Qwen2.5 1.5B Q4_K_M model and the existing locally fine-tuned Cortex 1.7B BF16 model. No new reader-model fine-tuning occurred. Local custom weights are not redistributed.

{reader_table}
![Local reader tokens and verified outcomes](assets/runtime-v1-readers.png)

Input/output token counts come from the local model server. They are not assumed tokenizer estimates or dollars. The 50% repeated-request ratio is designed into this narrow fixture. Timing covers packet preparation and local HTTP/cache operations, with one run per mode and ordinary server prefix caching. These results do not establish general coding productivity, neural recursive intelligence or service-scale latency.

The Cortex reader improves from 36/48 verified outputs with full scope to 42/48 with prepared context, but six outputs still fail. They are retained and `run_verified` does not cache or release them as verified results. An [exact parser/calculator control](../evidence/runtime-v1/exact-tool-control.json) handles all 48 requests with zero model calls and rejects three malformed/ambiguous packets. It parses the declared source grammar and uses the existing exact-decimal tool; it does not read evaluation answers. For this structured task, direct calculation is the strongest control. This does not repair or fine-tune the reader weights, and the grammar is not a general natural-language solver.

## Spherical contribution: an explicit control

The 80-task synthetic ablation uses eight equal-semantic candidates per task; distractors differ in one declared facet. Same candidate set and top-1 budget throughout. Semantic float and semantic 256-bit selection each succeed 9/80. Full float facets and full 256-bit facets each succeed 80/80; removing relation yields 35/80 and removing entity 43/80. **Exact metadata also succeeds 80/80.**

![Same-budget synthetic facet ablation](assets/runtime-v1-facets.png)

This fixture isolates the value of supplied facet information and exercises the 32-byte representation. It does not show that the stamp beats exact metadata, dense semantic retrieval on natural documents or a graph system. Source text, schema and relationship maps remain external. A natural-workflow ablation and independent replication are still required.

## Reproduce and use

[Complete API example](unified-runtime.md) · [remaining milestone gates](unified-context-roadmap.md).

```powershell
$Work = (Resolve-Path '..\\..\\work').Path
$Python = Join-Path $Work 'controller-venv\\Scripts\\python.exe'
$env:OMP_NUM_THREADS = '4'
$env:MKL_NUM_THREADS = '4'
$env:OPENBLAS_NUM_THREADS = '4'
& $Python experiments/train_expert_router.py --work $Work
& $Python experiments/audit_runtime_latency.py --work $Work
# Run separately from GPU timing. Requires the two pinned local Ollama models.
& $Python experiments/benchmark_unified_runtime.py
& $Python experiments/ablate_spherical_runtime.py
& $Python experiments/verify_runtime.py
& $Python examples/unified_context.py
```

Reuse the verified caches and environment from the [controller-v2 runbook](controller-methodology-v2.md). Run reproduction in a separate checkout: experiment scripts write derived evidence. The archived initial failure is preserved in the published release and is not regenerated by the corrected scorer. The verifier checks recorded artifact/source hashes and reported gains; numerical reruns on different hardware require a new manifest rather than expected identical hashes. No GPU job overlaps the local-reader benchmark in the recorded run.

Evidence: [protocol](../evidence/runtime-v1/protocol.json), [latency](../evidence/runtime-v1/latency-summary.json), [router training](../evidence/runtime-v1/router-training.json), [reader observations](../evidence/runtime-v1/workflow-observations.json), [facet ablation](../evidence/runtime-v1/spherical-ablation.json). Implementation and documentation © 2026 Prashant Jagtap; MIT source, separately attributed evidence.
"""
    write(ROOT / "docs/runtime-v1-results.md", results)
    article = f"""# Context Stamps: making context handling faster without giving up the measured retrieval quality

By Prashant Jagtap

An AI workflow spends time on more than generating its final answer. It retrieves evidence, checks which version is current, gathers dependencies, formats context and often repeats work it has already completed.

My work on Context Stamps starts with those operations. I want a research or coding workflow to carry a compact reference to the context it needs, resolve the relevant evidence, and reuse work only when the underlying inputs still match.

Repository: https://github.com/jprbom/context-stamps

## The context problem

Semantic retrieval can find a document about the right subject and still miss an important distinction: the wrong experiment, an old dataset revision, a result from another task, or a dependency that changed after the result was produced.

The Spherical Context QR reference represents several views separately: semantics, task, entity, relationships, time, authority, policy and modality. Their allocated bits fit into 256 bits, or 32 bytes.

The detailed source, graph and access rules stay outside that reference. A resolver checks the exact identities and versions before returning evidence. The stamp helps organize context access; it cannot reconstruct arbitrary documents from 32 bytes.

## What the RTX experiments showed

I previously trained nine small-controller variants on 3,134 public SciFact/NFCorpus queries, with separate tuning and calibration. Those runs exposed a useful limit: the small learned model did not consistently beat hybrid retrieval across domains.

A fixed hybrid/cross-encoder blend performed better on the locally fresh FiQA evaluation at that time: 0.4125 nDCG@10 versus 0.3687 dense and 0.3888 hybrid. nDCG measures how well relevant documents are ordered near the top. It is not answer accuracy.

That gain had a cost. Cross-encoder reranking was much slower than dense retrieval. The next question was whether I could remove execution overhead while keeping the achieved rankings.

## Reducing latency without dropping evidence

I kept the model weights, candidate set and 512-token pair limit. I changed how the work reaches the GPU.

Pairs of similar length now run together, reducing computation on padding. Repeated passage text can reuse its exact token IDs. FP32 weights remain in place while matrix operations use the existing BF16 execution path.

The corrected run preserved {parity:,} of 3,677 full top-10 rankings across the five regression collections. The original deployment decisions also remain intact: fusion for SciFact and FiQA, dense elsewhere.

On FiQA, the new paired measurement was:

Original fusion — {before:.2f} ms median
Optimized, cold passage-token cache — {cold:.2f} ms
Optimized, warm passage-token cache — {warm:.2f} ms

That is a {before/warm:.2f}× warm-stage speedup, with FiQA nDCG@10 still at 0.4125. These measurements include retrieval and reranking, but exclude query encoding, model cold load, answer generation and networking. They describe this local RTX setup, not an end-to-end service guarantee.

## A failure the wider test caught

The first optimization mishandled token truncation when both the query and document were long. It changed 231 ArguAna rankings. I retained that run, corrected the preparation path to use the pinned tokenizer for long pairs, and repeated the full test.

I also tried converting all weights to FP16 or BF16. The development pilot ran faster, but its scores changed. That configuration was not promoted. The earlier small-model int8 path had not improved CPU speed either. Precision needs to fit both the numerical problem and the operators available on the hardware.

## A single interface for the workflow

The repository now has a local ContextRuntime API. It prepares context, checks dependencies and permissions, enforces a byte or tokenizer-backed budget, resolves missing evidence within a bounded loop, and records verified outcomes.

An application supplies its own verifier. For an experiment, that might require the right dataset, configuration and checkpoint revisions. For code, it might check the source and test bindings. Missing evidence can trigger another resolution step; a repeated state or exhausted budget stops the loop.

Exact computation reuse binds the request, model, prompt, tool, policy, verifier and source versions. A source edit invalidates the earlier receipt. An unverified output is not cached as a successful result.

## Testing with two local small models

I tested the runtime with a local Qwen2.5 1.5B quantized model and my existing Cortex 1.7B BF16 model. The fixture contained 12 fictional research configurations, each queried before and after a source edit, with one repeat at each version.

The tasks asked for batch size, gradient accumulation and their product. The application checked the three integer outputs exactly. I compared full-scope input, a dependency-complete packet and verified reuse.

{chr(10).join(reader_lines)}

The Cortex reader still produced six outputs that failed verification after context preparation. Those failures remain in the evidence. I also tested an exact parser/calculator control: all 48 requests passed without a model call. For this declared structured format, direct calculation is the most efficient choice. The context runtime can use that tool through the same verified-computation interface.

Half the requests were repeats by design. This is evidence for that workload and its source-change handling; it is not an open-ended coding or general agent-intelligence benchmark. The repository includes individual observations, token counts, latency and failures.

## Testing the spherical part separately

I also added a controlled 80-task fixture in which candidates had identical semantic descriptions but differed in declared context facets.

Semantic-only selection succeeded in 9 of 80 cases. Full float facets and the full 32-byte stamp each succeeded in all 80. Removing the relation view reduced the result to 35; removing entity reduced it to 43. An exact-metadata control also achieved 80.

The result shows what the supplied metadata contributes in this constructed setting. It does not establish that a stamp is superior to exact metadata or that these gains transfer to natural research workflows. That comparison is now explicit in the evidence.

## How to use it

The core needs no GPU or downloaded model:

git clone https://github.com/jprbom/context-stamps.git
cd context-stamps
python -m pip install -e .
python examples/unified_context.py

The example prepares a version-checked evidence packet, reuses a verified computation and revokes the receipt after a dependency changes. Python APIs, CLI tools and portable agent instructions remain available. Neural ranking and GPU optimization are optional.

My immediate applications are research experiment tracking, evidence handoffs between agents, code-analysis invalidation and repeated verified computations. The host still owns authentication, source ingestion, adapter timeouts and the definition of a correct outcome.

## What remains

A trained cheap-expert router did not pass calibration, so it remains disabled. Short latency budgets do not silently force a weaker route. Predictive context-state learning, stronger verified workflow supervision and independent application trials remain the next research steps.

The current bounded loop does not change its own weights or establish recursive intelligence. I want those claims to follow measured task gains at matched compute, rather than treating extra architectural components as proof.

Code, figures, retained failures and reproduction instructions:
https://github.com/jprbom/context-stamps

Runtime results:
https://github.com/jprbom/context-stamps/blob/main/docs/runtime-v1-results.md

Usable API:
https://github.com/jprbom/context-stamps/blob/main/docs/unified-runtime.md

Original implementation and article © 2026 Prashant Jagtap. MIT-licensed source; public-data-derived evidence has separate attribution and terms.
"""
    write(ROOT / "docs/research-update-runtime-v1.md", article)
    launch = ROOT.parent / "runtime-v1-launch"
    launch.mkdir(exist_ok=True)
    plain = "\n".join(line.lstrip("# ") if line.startswith("#") else line for line in article.splitlines()) + "\n"
    for platform in ("linkedin", "x"):
        write(launch / f"{platform}-article.txt", plain)
    linkedin = (f"I’ve updated Context Stamps with a local context runtime and a measured latency improvement.\n\n"
        f"The optimized reranker preserved {parity:,}/3,677 top-10 rankings in the regression suite. FiQA median retrieval/reranking time fell from {before:.1f} ms to {warm:.1f} ms with a warm passage-token cache; nDCG@10 remains 0.4125.\n\n"
        "The article covers the truncation bug caught by wider testing, two local small-model workflow tests, exact reuse and the remaining gaps. The cheaper-expert learner failed calibration and stays disabled.\n\n"
        "Source and evidence: https://github.com/jprbom/context-stamps\n\nRead the article: [insert published article link]\n\n#MachineLearning #InformationRetrieval #OpenSource")
    xpost = (f"Context Stamps: FiQA reranking {before:.0f}→{warm:.0f} ms median (warm token cache), nDCG@10 unchanged at 0.4125. "
             "Added bounded context workflows and local small-model tests. Evidence and failures included.\nhttps://github.com/jprbom/context-stamps")
    write(launch / "linkedin-post.txt", linkedin)
    write(launch / "x-post.txt", xpost)
    counts = dict(article_characters=len(plain), linkedin_post_characters=len(linkedin), x_post_characters=len(xpost),
        note="Drafts only. Upload the PNG tables/figures; do not paste Markdown tables into the LinkedIn editor. No posts were sent.")
    assert len(plain) < 15000 and len(linkedin) < 3000 and len(xpost) <= 280, counts
    write(launch / "publishing-notes.json", json.dumps(counts, indent=2) + "\n")
    print(json.dumps(counts))


if __name__ == "__main__":
    main()

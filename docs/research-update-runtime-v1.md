# Context Stamps: making context handling faster without giving up the measured retrieval quality

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

The corrected run preserved 3,677 of 3,677 full top-10 rankings across the five regression collections. The original deployment decisions also remain intact: fusion for SciFact and FiQA, dense elsewhere.

On FiQA, the new paired measurement was:

Original fusion — 191.34 ms median
Optimized, cold passage-token cache — 115.95 ms
Optimized, warm passage-token cache — 89.17 ms

That is a 2.15× warm-stage speedup, with FiQA nDCG@10 still at 0.4125. These measurements include retrieval and reranking, but exclude query encoding, model cold load, answer generation and networking. They describe this local RTX setup, not an end-to-end service guarantee.

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

cortex-harness:1.7b-v11: full scope used 85,246 reported input tokens with 36/48 verified answers; prepared context used 14,974 with 42/48; verified reuse used 8,425 and 27 model calls with 42/48.
qwen2.5:1.5b: full scope used 86,062 reported input tokens with 48/48 verified answers; prepared context used 15,790 with 48/48; verified reuse used 7,895 and 24 model calls with 48/48.

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

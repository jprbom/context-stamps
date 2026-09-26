# Context Stamps: what changed after testing the failure cases

By Prashant Jagtap

When an AI workflow retrieves a relevant document, it has answered only one question: does this material appear useful for the request?

It still needs to establish whether the source is current, which other sources it depends on, whether the requesting agent may use it, and whether an earlier computation remains valid. A research result can become stale because a dataset, evaluation script or checkpoint changed, even when the wording of the request stays the same.

That is the context problem I am working on with Context Stamps.

Repository: https://github.com/jprbom/context-stamps

## The idea and its boundaries

Spherical Context QR is a 256-bit, or 32-byte, reference across several context views: semantics, task, entity, relationships, time, authority, policy and modality.

Each view is represented separately before its allocated bits are packed into the stamp. The document, detailed relationship map and access rules remain outside it. A resolver uses the reference alongside exact identities and version checks to recover the evidence a workflow needs.

The distinction matters. A compact similarity reference cannot reconstruct arbitrary context. Its useful role is to help organize access to context while retaining a precise recovery path. When compact routing has not qualified for a scope, the current implementation falls back to precise retrieval.

I also wanted to test a more ambitious question: could a small learned controller improve which evidence is selected, without requiring a large new model?

## What the previous experiments exposed

The earlier learned selectors did not generalize reliably. A recurrent model also deteriorated sharply when I gave it more inference passes than it had seen during training. More computation was making the result worse.

The training setup had several weaknesses. One dataset supplied most of the examples. The objective did not directly emphasize errors near the top of the ranking. Candidate lists could exclude useful documents before the model had a chance to score them. The available embeddings had already compressed away token-level details.

These failures changed the next experiment. I kept the old results and added stronger controls rather than replacing the evidence with a cleaner-looking run.

## What I trained locally

I reused 3,134 public training queries from SciFact and NFCorpus, with 290 separate tuning queries. The four previously inspected test collections remained regression checks.

I added FiQA for a fresh local transfer evaluation: 500 development queries for calibration and 648 test queries. No FiQA training queries entered fitting or tuning. This separation concerns my local experiment; it does not establish that every example was absent from upstream model pretraining.

The experiment ran on my RTX 5080 Laptop GPU. I trained three recipes with three seeds each, for twelve epochs. The selected small model has 98,817 parameters. Recurrent variants have 100,929. Those counts exclude the frozen encoder, teacher and retrieval index.

A frozen cross-encoder scored about 1.18 million query–document pairs locally. Recorded candidate preparation and teacher scoring took about 20 minutes, with initial download and FiQA encoding additional. The nine training loops and tuning checks took about 5.4 minutes, excluding final evaluation.

The changes were practical:

- Balance the two training domains within each optimizer step.
- Weight ranking errors by their effect on top-10 retrieval quality.
- Preserve separate query and document roles in the input features.
- Expand the candidate pool and measure its attainable ranking ceiling.
- Compare teacher distillation with a simpler model that does not use it.
- Replace unbounded recurrent accumulation with a damped update that converges for fixed inputs.

The complete protocol, checkpoints, training histories and per-query evidence are in the repository.

## What improved—and what did not

The strongest new result comes from a fixed blend of hybrid retrieval and cross-encoder scores on FiQA. Its deployment gate passed on development data before the test set was evaluated.

FiQA — nDCG@10, higher is better

Dense: 0.3687
Hybrid: 0.3888
Selected learned student: 0.3853
Fixed hybrid/cross-encoder blend: 0.4125

The blend improves on dense by 0.0438 nDCG and on hybrid by 0.0238. The paired 95% interval for the gain over hybrid is approximately +0.0138 to +0.0340.

The other collections show why scope matters:

SciFact — dense 0.6451; hybrid 0.7225; student 0.7316; blend 0.7269.

NFCorpus — dense 0.3167; hybrid 0.3503; student 0.3441; blend 0.3609.

ArguAna — dense 0.5041; hybrid 0.5373; student 0.5171; blend 0.5204.

SciDocs — dense 0.2164; hybrid 0.2043; student 0.2003; blend 0.1981.

The student beats hybrid on SciFact and trails it on the other four collections. The simpler student won neural tuning; neither recurrence nor distillation established a stronger general solution.

The frozen policy accepts the blend for SciFact and FiQA and uses dense for the other scopes. NFCorpus's favorable test result does not override its earlier inconclusive calibration. SciFact's small blend advantage over hybrid is also inconclusive on test despite the calibration gate passing.

These results support a scoped retrieval improvement. They do not establish universal superiority or a retrieval gain caused by the 32-byte stamp itself.

## The cost of better retrieval

On the measured FiQA queries, median retrieval-plus-ranking latency was approximately:

Dense: 6.8 ms
Hybrid: 13.7 ms
Learned student: 19.6 ms
Hybrid/cross-encoder blend: 193.8 ms

The measurements exclude query encoding, reader generation, networking and concurrent load. They use 20 fixed queries with three timed repeats per method.

The more accurate path is substantially slower here. That makes cost-aware expert routing a useful next step: invoke expensive ranking only when the expected benefit justifies it, then test the router against the best fixed alternative.

This experiment did not measure reader-token savings. The repository's separate exact-computation reuse pilot halved calls and input tokens on a workload with 50% repeated computations, but that was a narrow local fixture study. Its benefit should not be attributed to the new retrieval model.

## Two engineering fixes that held up

First, the new recurrent update is numerically stable under additional passes. On 160 stress queries, eight-pass states were within four ten-millionths of the 32-pass reference. Four-, eight- and 32-pass retrieval scores were almost unchanged. This addresses the earlier extra-depth failure; it does not demonstrate recursive intelligence or self-improving weights.

Second, I found a quantization problem. Embedding values and retrieval statistics occupied very different numerical ranges, but whole-layer int8 treated their concatenated input together.

I split the projection so that the embedding branch can use int8 while retrieval features remain in full precision. The transformation preserves the original full-precision computation before quantization; no new weights were trained for this fix.

Mean score error fell by roughly 28–122 times across the five regression collections. Some rankings still change. The small CPU forward pass was also slightly slower than FP32, so this is a fidelity improvement, not a speed claim.

## How someone can use the project

The core runs locally without a GPU or downloaded model. From a reviewed clone:

git clone https://github.com/jprbom/context-stamps.git
cd context-stamps
python -m pip install -e .
python examples/progressive_context.py
python examples/computation_reuse.py

The repository also provides Python APIs, CLI tools, portable agent instructions and MCP integration. Neural training and ranking are optional and have a separate local RTX runbook.

I see the immediate application in research and coding workflows where provenance and changes matter: keeping experiment conclusions attached to the right dataset and checkpoint; invalidating a code-analysis result when its source or dependency changes; and reusing a verified computation when its request, model, policy and source bindings still match.

Those are intended applications of the interfaces. Broad developer-productivity or agent-success improvements still require independent workflow studies.

## The next milestones

The longer-term target is one context runtime that can choose evidence, enforce a budget, select an appropriate expert, reuse valid work and reconsider a result within a bounded loop.

The next priorities are cost-aware expert routing, verified workflow supervision and a same-budget ablation of the spherical facets and relationship views. Predictive context-state learning and a token-interaction student are research directions to test against simpler controls.

Token efficiency, iterative reasoning, multimodal quality, model portability and edge deployment now have separate acceptance gates in the roadmap. Passing one retrieval benchmark cannot stand in for all of them.

Source code is MIT-licensed with my copyright notice. Public-dataset-derived checkpoints and evidence carry separate attribution and terms. The source, failed experiments, model cards, figures and reproduction instructions are available here:

https://github.com/jprbom/context-stamps

Results: https://github.com/jprbom/context-stamps/blob/main/docs/controller-v2-results.md

RTX runbook: https://github.com/jprbom/context-stamps/blob/main/docs/controller-methodology-v2.md

Runtime milestones: https://github.com/jprbom/context-stamps/blob/main/docs/unified-context-roadmap.md

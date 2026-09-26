# What local training changed in Context Stamps

By Prashant Jagtap · 26 September 2026

Historical controller-v1 article. The [revised follow-up article](research-update-controller-v2.md) covers the subsequent nine-run study, fresh FiQA evaluation, runtime trade-offs and unified-runtime milestones.

In a research or coding workflow, finding a relevant document is only part of the context problem. A result may depend on a particular dataset revision, evaluation script, model checkpoint and policy. When one changes, the system needs to know which conclusions can still be reused.

I am building Context Stamps around that problem. Its Spherical Context QR is a 256-bit reference across several context facets. The document and relationship map remain outside those 32 bytes. The reference helps route a request; exact checks determine what evidence or previous computation is still valid.

The latest work was to test whether a small learned controller could improve evidence ranking, and whether better computation identity could reduce repeated model work. I trained and tested both locally on my RTX 5080 laptop GPU.

## The training experiment

I reused the public SciFact and NFCorpus data already downloaded for the project. After separating normalized query text across partitions, the experiment had 3,134 training queries, 290 tuning queries and 288 independent calibration queries. ArguAna and SciDocs were evaluation-only collections.

The controller uses frozen MiniLM embeddings and retrieval statistics. I compared a small residual ranker with a recurrent attention variant. Both learn corrections around a semantic/lexical retrieval score. The attention model revisits the candidate set using shared weights; it does not rewrite itself or modify the reader model's attention.

Each architecture trained for 12 epochs with three seeds. That produced six runs, including the simpler baseline. The recurrent model has 83,137 trainable parameters; the simpler one has 66,497. The encoder and retrieval index are additional components.

The smaller model won on tuning data. However, independent calibration did not provide sufficient evidence to deploy either learned controller over the existing baselines. That is the central result of this training iteration.

## What the public-data results show

The following values are nDCG@10, a measure of whether relevant documents appear near the top. Higher is better. The learned model was selected using tuning data, not by choosing the best test result for each dataset.

SciFact
Dense: 0.6451 · Hybrid: 0.7225 · Selected learned: 0.7190

NFCorpus
Dense: 0.3167 · Hybrid: 0.3503 · Selected learned: 0.3506

ArguAna
Dense: 0.5041 · Hybrid: 0.5373 · Selected learned: 0.5294

SciDocs
Dense: 0.2164 · Hybrid: 0.2043 · Selected learned: 0.1934

The learned approach remained above this dense reference on three datasets, but much of that advantage was already present in hybrid retrieval. On SciDocs, it made the transfer problem worse. These test sets were inspected during earlier project work, so this is a regression study, not an untouched evaluation or an industry-wide ranking.

I also checked whether more recurrent passes helped. They did not. Taking the attention model from its trained two passes to four substantially reduced accuracy. The serving interface now enforces the checkpoint's trained depth. Extra computation needs its own training and evidence.

## Where there was a practical efficiency benefit

I added a computation identity that binds the request to the source content and revision, model, prompt, tool, policy and principal. A previous result can be reused only when those bindings match. A changed source or policy produces a cache miss.

In a separate local Qwen 1.5B experiment, I tested 12 fictional experiment-configuration tasks with source edits, policy changes and repeated requests. Each mode processed 216 observations across three timing repetitions. Half the computations were intentionally identical repeats.

Always calling the model used 216 calls and 12,780 input tokens. Exact reuse used 108 calls and 6,390 input tokens. Both modes returned 216 correct outputs, and total elapsed time fell from 20.70 seconds to 10.94 seconds, a reduction of 47.2%.

The p95 latency did not improve: 114.9 ms without reuse versus 117.1 ms with reuse. Misses still pay the model-call cost. The result supports a specific repeated-workload benefit; it does not establish a general agent-speed improvement. The fixtures also provide an explicit correctness oracle, which a real application must replace with its own verifier.

## Quantization and local deployment

Per-row int8 storage reduced the selected checkpoint from 267,060 bytes to 70,440 bytes, about 3.79 times smaller. There was a small ranking change, so the export is not lossless. It reconstructs FP32 weights when loaded and does not claim int8 execution speed.

At this model size, a warm batch-one forward pass was faster on CPU than on the GPU in the measured setup. GPU training is useful; GPU inference is not automatically the best choice for a small controller. Those timings exclude encoding, candidate retrieval and reader generation.

## How I would use this today

A research assistant could track which result depends on a dataset snapshot, training configuration and evaluation script. An agentic coding harness could invalidate a cached analysis when its source, tool or execution policy changes. A local SLM application could avoid repeating an already verified computation while retaining exact source bindings.

Those are integration targets, not completed production benchmarks. The repository includes offline examples, optional trained checkpoints, the complete local training commands, query-level results, failure records and reproducible checks. Its default retrieval remains conservative; loading an experimental model is an explicit choice.

The next research step is stronger candidate coverage and broader, correctly licensed supervision for evidence sufficiency and verified actions. I also need a new untouched evaluation collection before making a new generalization claim. A larger controller or extra recurrence would need to earn its cost against the simpler alternatives.

Repository: https://github.com/jprbom/context-stamps

Training instructions and complete evidence: https://github.com/jprbom/context-stamps/blob/main/docs/local-rtx-controller.md

Source: MIT, copyright Prashant Jagtap. Public-data-derived experimental weights and evidence retain separate dataset attribution and terms.

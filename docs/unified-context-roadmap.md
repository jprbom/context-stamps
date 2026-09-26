# Unified context runtime: implementation scope and acceptance gates

By Prashant Jagtap · 26 September 2026

The target is one usable context runtime that chooses evidence, checks relationships and permissions, controls a context budget, reuses valid work and requests additional computation when it is worth the cost. Existing model architectures are components to evaluate inside that runtime. Combining their names is not an acceptance criterion, and no finite benchmark can certify universal superiority.

## A coherent architecture

```text
Application request + trusted identity + task/budget policy
    -> observed facets and versioned spherical reference
    -> exact source resolution / candidate retrieval
    -> bounded expert choice: dense, hybrid, learned, cross-encoder, exact tool
    -> evidence and declared relationship checks
    -> budgeted context packet and inspectable route reason
    -> exact computation reuse or external reader/tool execution
    -> application outcome verification
    -> bounded reconsideration, stop or abstain
```

The same request/result contract should carry the chosen source IDs and versions, eligibility scope, evidence requirements, token/latency budgets, route explanation, reuse status, iteration count and verified outcome. The current repository supplies many of these primitives, but this complete end-to-end orchestrator is a **roadmap**, not an already validated one-call model.

**Implemented:** spherical references, explicit relation maps, source identity, dependency validation, selective packet invalidation, exact computation reuse, budgeted evidence selection, precise/compact routing controls and experimental learned rankers. **Partially measured:** retrieval quality, narrow local reader/workflow pilots, CPU/GPU inference and numerical recurrence. **Not established:** general recursive intelligence, neural mixture-of-experts superiority, predictive latent context modeling, frontier-model internals or universal downstream gains.

## The next implementation sequence

1. **One orchestration interface.** Compose the existing primitives behind `prepare_context`, `resolve`, `record_outcome` and `invalidate` operations. Keep external reader/tool adapters explicit. Define immutable scope/profile identities, missing-evidence outcomes and budget exhaustion before introducing automatic actions. Do not silently execute model-generated plans.
2. **Cost-aware expert routing.** Start with interpretable routing among existing retrievers, the cross-encoder and exact tools. Train a router from training-only query features and verified benefit/cost labels. Compare it with always-dense, always-hybrid, always-cross-encoder and the best fixed expert. A workflow expert router is distinct from training a neural mixture-of-experts language model.
3. **Predictive context state.** Learn to predict the next required evidence/state from verified workflow traces. Hold out projects, time periods and task templates. A latent prediction loss can be auxiliary supervision; downstream verified completion must decide whether the representation is useful. Keep the exact graph and provenance authoritative.
4. **Bounded iterative reasoning.** Let the runtime retrieve, inspect missing requirements, request an additional expert and verify the result under a strict compute budget. Include stop/abstain rules, repeated-state detection and rollback. The current contractive ranker stabilizes its numerical state; it does not already perform this reasoning loop or improve its own weights.
5. **Representation and compression.** Test a token-interaction student or a small encoder adapter with broader licensed training-only data. Preserve the 32-byte portable stamp and keep richer residuals, vectors and relationship maps external. Split-precision projection now reduces measured int8 distortion; quantization-aware training is a later comparison if the accuracy/cost trade-off warrants it.
6. **Adapters and deployment.** Keep the standard-library local path, Python API, CLI, portable agent instructions and MCP. Add reader adapters through one explicit contract. Measure on actual target hardware before claiming SLM, edge, image, audio, video or frontier-model improvement.

## Milestones that can be failed

The following are proposed release gates for the integrated runtime. Freeze numerical margins, datasets, budgets and statistical tests before each new run. “Pass” applies to a specified suite and environment, not every possible task.

| Gate | Evidence required | Current status |
|---|---|---|
| Reproducibility | Pinned inputs; separated fitting/tuning/calibration/test; all seeds/failures; clean-room replay and independent replication | Local source/artifact replay exists; independent replication remains open |
| Retrieval quality | Paired confidence intervals against dense, hybrid and a capable reranker on new held-out domains; predeclared noninferiority margin | Fresh FiQA fusion gain; student generalization and other scopes remain incomplete |
| Spherical contribution | Same encoder, sources, budget and backend with/without each facet, relation view and stamp route | Interface controls exist; a causal application benefit is not established |
| Token efficiency | Measured reader input/output tokens at matched verified task success; include resolver/adapter overhead | Narrow repeated-work pilots only; broad reader study required |
| Latency and throughput | End-to-end p50/p95, concurrency sweep, cold/warm states and memory at matched quality | Local stage timings exist; higher-quality FiQA fusion is slower; production gate open |
| Expert routing | Beat the best fixed expert on a predeclared quality/cost objective, including router overhead | Frozen scope choices tested; trained cost-aware routing not implemented |
| Iterative reasoning | Higher verified task success than equal-compute noniterative controls; bounded stops and no regression under adversarial feedback | Numerical recurrence repaired; reasoning benefit not demonstrated |
| Latent state learning | Held-out project/trajectory prediction plus improved verified workflow outcomes | Not implemented or trained in the current release |
| Safety and isolation | Defined injection, stale-context, revocation, tenant, malformed-input and dependency-poisoning suites; fuzzing and independent review | Local fail-closed controls tested; no complete adversarial service guarantee |
| Model portability | Matched experiments on at least two open models and a separately authorized frontier API; frozen prompts/settings | Some earlier local model pilots; no universal booster evidence |
| Multimodal usefulness | Task-specific image/speech/video metrics and human assessment where needed, with lawful data/model use | Earlier integration pilots only; no broad perceptual-quality improvement |
| Edge deployment | Actual target-device memory, energy, p95 latency and offline behavior against a fixed baseline | Laptop CPU/int8 measured; dedicated edge-device gate open |

Do not repeatedly tune on the current five test sets and call the next result independent. They are now development/regression resources. New data should be licensed and fingerprinted; private chats or enterprise documents should not become training inputs without an explicit data decision.

## What should be prioritized

The highest-value next experiment is **cost-aware expert routing**, because the current fresh-domain quality gain costs roughly 194 ms versus 6.8 ms dense retrieval in the measured FiQA stage benchmark. The second is **verified workflow supervision**, which is necessary to learn useful iterative behavior instead of just ranking documents. The third is **a same-budget spherical ablation**, which directly tests the central multidimensional context hypothesis.

Additional transformer blocks, larger networks and autonomous retraining should follow evidence from those experiments. A bounded offline train/evaluate/release cycle can improve the system while preserving reproducible failures and preventing a model from promoting itself using its own unverified outputs.

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

The local `ContextRuntime` now composes chosen source IDs and versions, eligibility scope, evidence requirements, token/latency budgets, route explanation, reuse status, iteration count and verified outcomes. [Runnable API and boundaries](unified-runtime.md). It accepts explicit host-owned reader/retriever/verifier adapters. A distributed service, autonomous tool agent and universal model booster remain research/deployment work.

**Implemented:** spherical references, explicit relation maps, source identity, dependency validation, selective packet invalidation, exact computation reuse, budgeted evidence selection, bounded verified context recovery, expert registration and experimental learned rankers/router. The trained cheap-route policy failed calibration and remains disabled. **Partially measured:** retrieval quality, narrow local reader/workflow pilots, GPU execution improvements, a synthetic facet ablation and numerical recurrence. **Not established:** general recursive intelligence, neural mixture-of-experts superiority, predictive latent context modeling, frontier-model internals or universal downstream gains.

## The next implementation sequence

1. **One orchestration interface — implemented locally.** `prepare_context`, `resolve`, `record_outcome`, `invalidate` and `run_verified` compose the existing primitives. External adapters remain explicit. Missing evidence, budget exhaustion and no-progress outcomes stop the loop. Model-generated plans never become automatic executable actions. Next: independent application integrations and process-level deadlines.
2. **Cost-aware expert routing — trained, not promoted.** A 14-feature ridge router was fitted on 3,134 public training queries, selected on 290 tuning queries and gated separately. Neither enabled fusion scope qualified for cheap exits. Its disabled result is retained; the runtime keeps qualified baseline choices. Next: richer training-only benefit/cost labels and a new holdout. A workflow expert router is distinct from a neural mixture-of-experts language model.
3. **Predictive context state.** Learn to predict the next required evidence/state from verified workflow traces. Hold out projects, time periods and task templates. A latent prediction loss can be auxiliary supervision; downstream verified completion must decide whether the representation is useful. Keep the exact graph and provenance authoritative.
4. **Bounded context recovery — implemented; reasoning benefit still open.** The runtime retrieves, checks an application verifier, resolves missing eligible evidence and verifies again under byte/token/deadline/iteration bounds. Repeated state and permission/version failures stop it. It does not update its own weights or prove general reasoning improvements. Next: verified real coding/research trajectories against equal-compute controls.
5. **Representation and compression.** Test a token-interaction student or a small encoder adapter with broader licensed training-only data. Preserve the 32-byte portable stamp and keep richer residuals, vectors and relationship maps external. Split-precision projection now reduces measured int8 distortion; quantization-aware training is a later comparison if the accuracy/cost trade-off warrants it.
6. **Adapters and deployment.** Keep the standard-library local path, Python API, CLI, portable agent instructions and MCP. Add reader adapters through one explicit contract. Measure on actual target hardware before claiming SLM, edge, image, audio, video or frontier-model improvement.

## Milestones that can be failed

The following are proposed release gates for the integrated runtime. Freeze numerical margins, datasets, budgets and statistical tests before each new run. “Pass” applies to a specified suite and environment, not every possible task.

| Gate | Evidence required | Current status |
|---|---|---|
| Reproducibility | Pinned inputs; separated fitting/tuning/calibration/test; all seeds/failures; clean-room replay and independent replication | Local source/artifact replay exists; independent replication remains open |
| Retrieval quality | Paired confidence intervals against dense, hybrid and a capable reranker on new held-out domains; predeclared noninferiority margin | Fresh FiQA fusion gain; student generalization and other scopes remain incomplete |
| Spherical contribution | Same encoder, sources, budget and backend with/without each facet, relation view and stamp route | Synthetic fixed-budget ablation completed; exact metadata ties full facets; real-workflow benefit remains open |
| Token efficiency | Measured reader input/output tokens at matched verified task success; include resolver/adapter overhead | Narrow repeated-work pilots only; broad reader study required |
| Latency and throughput | End-to-end p50/p95, concurrency sweep, cold/warm states and memory at matched quality | Local stage timings exist; higher-quality FiQA fusion is slower; production gate open |
| Expert routing | Beat the best fixed expert on a predeclared quality/cost objective, including router overhead | Router trained and rejected by calibration; defaults preserve the earlier scope choices |
| Iterative reasoning | Higher verified task success than equal-compute noniterative controls; bounded stops and no regression under adversarial feedback | Bounded missing-evidence recovery implemented and tested; broad reasoning benefit not demonstrated |
| Latent state learning | Held-out project/trajectory prediction plus improved verified workflow outcomes | Not implemented or trained in the current release |
| Safety and isolation | Defined injection, stale-context, revocation, tenant, malformed-input and dependency-poisoning suites; fuzzing and independent review | Local fail-closed controls tested; no complete adversarial service guarantee |
| Model portability | Matched experiments on at least two open models and a separately authorized frontier API; frozen prompts/settings | Some earlier local model pilots; no universal booster evidence |
| Multimodal usefulness | Task-specific image/speech/video metrics and human assessment where needed, with lawful data/model use | Earlier integration pilots only; no broad perceptual-quality improvement |
| Edge deployment | Actual target-device memory, energy, p95 latency and offline behavior against a fixed baseline | Laptop CPU/int8 measured; dedicated edge-device gate open |

Do not repeatedly tune on the current five test sets and call the next result independent. They are now development/regression resources. New data should be licensed and fingerprinted; private chats or enterprise documents should not become training inputs without an explicit data decision.

## What should be prioritized

The latest [runtime experiment](runtime-v1-results.md) addresses execution overhead independently of expert quality. Its first cheaper-expert learner did not qualify; future router work needs better supervision and new held-out workloads. The next priorities are verified real-workflow trajectories, independent application integration and a natural-data spherical ablation. The synthetic metadata fixture supplies a regression test, not a replacement for those studies.

Additional transformer blocks, larger networks and autonomous retraining should follow evidence from those experiments. A bounded offline train/evaluate/release cycle can improve the system while preserving reproducible failures and preventing a model from promoting itself using its own unverified outputs.

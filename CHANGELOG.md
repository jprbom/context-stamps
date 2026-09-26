# Candidate changes

All versions below are research releases or candidates. v0.5.0 is available from the public source repository; no PyPI release is announced here.

## Development update after 0.5.0 — 27 September 2026

- Profiled and refined local trajectory search without changing any of 882 tested returned fragment lists. A selective query plan lowers measured median lookup time from 279 to 99 ms and p95 from 357 to 312 ms. Retained the first optimization's p95 regression, both paired lookup logs, exact tie-boundary tests and a separately pinned runner. No whole-workflow speedup is claimed.

- Added source-bound local trajectory views and a CPU ridge policy fitted from paired outcomes. The LongMemEval-V2 text development run uses 72 training and 222 held-out questions: no memory 21/222, state retrieval 50/222, linked memory 50/222, learned choice 54/222. Retained 15 gains, 11 regressions, five truncations, increased model-token use and all excluded judge/image cases. Candidate remains inactive. Added the local runbook, native score/policy replay, research figure and nine new core tests; full local core suite passes 361 without skips.

- Added protected adaptation/retention cohorts to new local learning registries. Three quality/deadline checks per cohort and one pooled cost check share the round's risk budget; legacy replay is explicit and cannot downgrade a protected registry. Thirteen new cohort tests pass; the full local core suite passes 352 without skips.
- Fitted a CPU routing policy on 312 actual local RULER inputs and evaluated 208 fresh inputs, retaining 1,248 action records. Verified exact results reduce reader calls 208 to 32 and total model tokens 22.5%; native complete credit changes 193 to 197, with strict output checks 143 to 188. Learned choices match the fixed verified rule on every input. Retained unresolved QA, output variation, source overlaps and missing resource qualification; the candidate remains inactive. Added independent exact-result checks, offline policy/metric replay and the RTX runbook.

- Completed a 52-input, 208-call local RULER-v1 development comparison across all 13 tasks at nominal 4K/16K. Runtime complete native credit is 44/52 versus full context 34/52, with 76.1% fewer input tokens. Direct operations plus the reader achieve 45/52 with fewer tokens than the runtime; the zero-model-call control gets 42/52 and abstains on ten inputs. Retained both output truncations, all predictions and the permissive native metric. Added 14 boundary/evidence tests, reproduction scripts and a research figure. No weight activation, independent generalization, edge qualification or advantage over direct tools is claimed.

- Completed all 164 paired HumanEval+ tasks: base 49/164, adapter 49/164, with 19 gains and 19 regressions. Output-format failures fall 20 to 5, but generated tokens increase 7.3% and summed generation batch time increases 36.0%. Retained all 328 outputs, native grading, paired statistics and a review of every regression. Adapter remains inactive; no useful code-adaptation or Context Stamps gain is established.

- Moved external runtime callbacks outside the shared state lock and added context/receipt revalidation. Concurrent exact requests share a bounded pending computation; revocation wakes waiters and adapter exceptions release capacity. Eleven new concurrency tests pass, with 339 tests passing in the complete local environment. Historical source and benchmark evidence remain replayable. No production speedup or hard callback cancellation is claimed.
- Clarified local adaptation at three levels: verified memory, CPU context policies and optional small weight updates on capable hardware. Every supported device should retain a local evaluation and rollback path; automatic weight improvement remains unqualified.
- Completed the public-code training control: 368 independently checked MBPP examples, 184 optimizer steps, 65.42 seconds of local RTX training and a 4.17 MiB adapter. Retained all training-source and native HumanEval+ grader audits, including the reference numerical failure on task 32. Training completion does not activate the adapter.

## Development update after 0.5.0 — 26 September 2026

- Completed one offline Qwen 1.5B LoRA fixture experiment: 544,768 trainable parameters, 144 training rows, 27.73 seconds on RTX. Frozen base unchanged; adapter reload reproduces one recorded batch exactly. Test letter-selection improves 12/72 to 52/72 but introduces one new failure and trails the exact rule. Adapter stays inactive. Added training/replay code, probes, model card and local integration runbook; no general retention or autonomous deployment claim.

- Added bounded typed file tools, artifact-presence completion checks and an order-preserving response-schema serializer. Retained failed variants, 76 task calls and 24 schema canaries. Both models recover schema-copy correctness (0/3 to 3/3), but corrected typed interfaces still pass 0/2 coding tasks each. Twelve isolated file probes and eleven new boundary tests pass. No model training or task-quality benefit is attributed to this change.

- Added a pinned local Qwen2.5-Coder 7B control: 1/2 selected coding tasks pass, both exhaust the 12-step budget. All 24 calls and native test reports replay offline. This exposes evidence-use, repair-loop and termination gaps; it is a larger-model reference, not learned runtime improvement.

- Added a local Terminal-Bench 2.1 development pilot through isolated Harbor environments. The Qwen2.5 1.5B baseline passes 0/2 tasks; all six calls and the first failed interface attempt remain available for offline replay. Native reference/empty controls validate both graders. Offline containers have bounded CPU, memory, processes, output and time, with no host mounts. Trusted Python helpers now ignore agent-created modules. Sixteen boundary tests cover the harness; no runtime benefit or fine-tuning result is claimed.

- Made locally improving small domain models the primary application objective. Added a bounded Beta-smoothed cell policy and local SQLite candidate registry with frozen plans, fresh task IDs, error-budget accounting across rounds, paired quality/resource checks, activation and rollback. Added 24 boundary tests and a replayable simulation: 60 training fixtures and two 600-task evaluation fixtures. No new SLM weights, GPU training, model-quality gain or edge-device qualification is claimed.

- Added a pinned local LongBench v2 control runner with native prompts/scoring, whole-input token checks, retained failures, immutable registrations and offline evidence replay. Thirty RTX calls cover ten short-context tasks across six domains: full context 4/10, BM25 4/10, no context 3/10. No runtime gain or full-benchmark claim. Ten new evaluation-boundary tests pass alongside seven existing provider checks. Harbor 0.23.0 CLI preflight also passes; coding-agent task/sandbox qualification remains.

- Added a bounded optional acquisition planner, portable three-head predictor and whole-trajectory risk accounting. Six RTX fits use public training partitions; 788 calibration and 3,677 regression trajectories retain all false early stops. No scope passed the 5% error gate; unqualified scopes keep the full candidate pool and skip unused feature/model work. Added 14 targeted tests, a complete example, source-bound evidence, model cards and a research figure; all 304 local tests pass. No default routing change, answer-sufficiency qualification or end-to-end token/latency claim.

- Added `IncrementalExecutor`: bounded pure-computation graphs, affected active descendants, exact dependency/version reuse, request-hash receipts, current authority checks and explicit failure outcomes. Added 23 tests and a complete incremental-metrics example. All 290 local tests and 720 fictional workflow requests pass. Localized changes avoided callbacks; policy-wide changes forced recomputation and added overhead. No new trained-model, token-saving, distributed scheduling or hard callback-cancellation claim.

- Added `ContextStore` and `ContextWorkingSet`: signed persistent temporal state/ACLs, authorized metadata inventories, verified cold payloads, exact dependency paging, pins, bounded residency, explicit asynchronous prefetch and retained-source eviction. Added 34 tests and a complete persistence-to-managed-decision example. All 267 local tests and 540 fictional replay checks pass. Working-set reuse reduces payload reads; prefetch regresses against ordinary working-set reuse and remains opt-in. Retention/deletion policy, distributed coordination and learned speculation remain unfinished.

- Added `ManagedExecutor`: exclusive durable dispatch claims, cancellable adapter/verifier processes, current-state checks, bounded result protocol, reserved outcome capacity and terminal provider reconciliation. Schema 2 preserves unknown usage and leaves existing schema-1 bytes unchanged. The final local suite passes 233 tests without skips; offline timing includes process startup and does not show a latency benefit. Retained two earlier runs and corrected a Windows mixed-clock measurement failure. Historical engineering sources are archived and verified as data so later code changes do not overwrite prior evidence.

- Added `AuditStore`: transactional reference-only experience history, actor-bound keyed receipts, exact-event retry handling, strict plan/outcome ordering, current authorization checks and external checkpoint verification. Added an integrated context-to-decision-to-audit example, 19 audit tests and local append/reopen measurements. The full local suite passes 201 tests; external action execution/cancellation and production-scale reliability remain unqualified.

- Added research interfaces for canonical temporal/epistemic state, finite negative knowledge, explicit conflict resolution and serialized-cost context compilation. Added typed batch decisions with scoped calibration, packet integrity bindings and mutation/revocation rejection. The 182-test CPU run and 80 synthetic compiler measurements retain their source hashes and limits; no new model-quality gain is claimed.

- Added `ContextRuntime` for registered experts, exact dependency closure, tokenizer-backed budgets, bounded evidence recovery, outcome verification and source-bound computation reuse.
- Added length-aware BERT reranking with a bounded passage-token cache. Preserved the first long-query truncation failure, fixed it through the pinned tokenizer and reran all five public collections. Whole-weight FP16/BF16 conversion remains experimental.
- Trained a cost-aware ridge router using existing public training partitions; neither fusion scope qualified for cheaper exits. Added two local-reader workflow pilots, a fixed-budget synthetic facet ablation with an exact-metadata control, figures and offline evidence replay.

- Added controller-v2: balanced domain sampling, metric-aware pair loss, query/document role features, pinned cross-encoder distillation and a mathematically contractive attention recurrence. Trained and retained all nine seed/recipe runs.
- Expanded candidate unions and measured both recall and oracle top-10 ceilings. Added fresh local FiQA calibration/test partitions without using FiQA training data. Frozen fusion improves FiQA retrieval over dense and hybrid; the learned student remains experimental and is not a universal improvement.
- Added real dynamic-int8 CPU execution, online retrieval/reranking timings, batch-shape ranking checks, new regression tests, source/artifact fingerprints, updated research figures and local training instructions. Public source and derived artifacts retain separate attribution.

- Implemented an optional residual ranker with shared recurrent attention, a no-attention baseline, listwise training, hybrid-distribution regularization and bounded score corrections. Trained six local RTX runs on public training splits; preserved all outcomes and independent calibration decisions.
- Added experimental FP32/int8-storage safetensors checkpoints, per-query evidence, CPU/CUDA/precision measurements and a reproducible local runbook. The learned model did not qualify for promotion; no default retrieval replacement or base-model fine-tuning is claimed.
- Locked serving to checkpoint training depth after four-step inference caused severe transfer regressions. Direct forward overrides remain an explicit research ablation interface.
- Added exact computation identities, a bounded expiring result cache and exact decimal tools. A real local Qwen pilot halved calls/input tokens on a 50%-repeat workload and reduced total elapsed time 47.2%; p95 latency slightly worsened. This is a narrow synthetic workload, not a general agent-speed claim.
- Retained all earlier compact-retrieval and multimodal limitations. No source/corpus graph or arbitrary document is contained in the 256-bit stamp.

## 0.5.0

- Added typed, directional relation maps with bounded personalized diffusion as an eighth-view input to the 256-bit product-sphere profile. The map remains external and versioned; the stamp is not a serialized graph.
- Added independent per-view ITQ fitting for a product of facet spheres. The allocation must total exactly 256 bits, and trained families now encode through the same 32-byte stamp format.
- Added a precise MiniLM/BM25 score-fusion backend and a validation-bound scope certificate. A profile is enabled only when the paired bootstrap lower bound exceeds the caller's minimum gain; otherwise retrieval falls back to dense.
- Preserved the frozen SciDocs prospective failure: the 0.75/0.25 blend reduced nDCG@10 by 0.0121. The same blend improved SciFact, NFCorpus and ArguAna by 0.0774, 0.0337 and 0.0344 respectively. This is evidence for scope gating, not universal superiority.
- Added checked evidence for 9,087 query/method records, an animated reference diagram, a static research cover and revised publication material. No claim is made that the 256-bit capsule alone exceeds dense retrieval or changes model-internal attention.

## 0.4.0

- Added a bounded deterministic facet compiler with per-facet source and rule provenance. Missing facets are omitted; authority, policy and modality remain host assertions.
- Added a versioned structured 256-bit research profile across semantic, task, entity, relation, temporal, authority, policy and modality views.
- Bound compact exits to an application risk budget and exposed route reasons, the certified upper error bound and calibration count.
- Added eight authored facet controls, 100 exact 32-byte round trips and five route-path controls. These test behavior, not public retrieval improvement.
- Retained the unqualified public compact policy and precise fallback. No new model training or attention claim is made.

## 0.3.3

- Added explicit partial-facet queries and calibration scopes tied to active views, families and weights.
- Replaced global session invalidation with dependency-selective receipt invalidation.
- Added 4,000 mutation-oracle comparisons and a 6,600-handoff replay against exact and global-cache controls; identical evidence retained.
- Rewrote the README around current defaults, three-seed findings, usable examples and measured limits. Historical experiments remain available.
- Kept document stamps at 32 bytes. No new model training or internal attention change in this update.

## 0.3.2

- Added exact-first routing, optional validated compact exits and bounded context sessions.
- Recorded three-seed ITQ training: mean compact relevance improved, dense remained stronger, one ArguAna seed regressed slightly.
- Preserved dense ranking through fallback when compact calibration failed.
- Added narrowly scoped secret-scanner exceptions with five controls; retained all default rules.

## 0.3.1 and earlier

See the [historical retrieval repair](docs/retrieval-repair.md), [spherical results](docs/spherical-results.md) and [failure ledger](docs/failures-and-fixes.md). Earlier losses and unsuccessful runs have not been replaced by newer measurements.

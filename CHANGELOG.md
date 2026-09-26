# Candidate changes

All versions below are research releases or candidates. v0.5.0 is available from the public source repository; no PyPI release is announced here.

## Development update after 0.5.0 — 26 September 2026

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

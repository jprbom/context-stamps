# Candidate changes

All versions below are research releases or candidates. v0.5.0 is available from the public source repository; no PyPI release is announced here.

## Development update after 0.5.0 — 26 September 2026

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

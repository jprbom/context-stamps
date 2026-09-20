# Failure ledger and release gates

## v0.5.0: relation-aware product spheres and a failed prospective blend

The frozen 0.75 MiniLM / 0.25 BM25 blend improved three previously inspected datasets but regressed on the untouched SciDocs check: 0.2043 versus 0.2164 dense nDCG@10, with a paired 95% interval from −0.0194 to −0.0046. The result is preserved under `evidence/hybrid-retrieval-v1`. A fixed blend is therefore not a universal replacement for dense retrieval.

`certify_hybrid_scope` now binds the profile to one validation scope and enables it only when a paired bootstrap lower bound exceeds the minimum requested gain. Otherwise it abstains and selects dense scores. This protects a validated deployment policy; it does not convert the post-failure remediation into prospective evidence. The next test must freeze the scope rule before evaluating a new untouched test set.

Typed relation diffusion and per-facet ITQ now feed the 256-bit product-sphere capsule. These additions make the multidimensional mechanism concrete, but their causal retrieval benefit has not yet been isolated from the precise hybrid backend.

## v0.4.0: inspectable facets and risk-budgeted routing

The repository now has a deterministic observed-facet compiler, but it is not an independently evaluated semantic extractor. Its eight authored cases validate reproducible behavior only. The structured 96/32/32/32/16/16/16/16 profile is exactly 256 bits and passes 100 round trips; its retrieval value and bit allocation remain untested. Compact exits now require a nonzero validation count and a certified error upper bound within the caller's budget. The public compact policy still does not qualify, so precise fallback remains the measured default. [Design, controls and next gate](adaptive-capsules.md).

## v0.3.3: partial facets and selective invalidation

Missing query facets can be represented explicitly with `FacetQuery`; policies are scoped to the observed mask, families and weights. This is not automatic facet inference. Session mutations now invalidate affected dependency packets rather than all receipts. 4,000 oracle comparisons and the changing-context replay verify correctness and expose the cache tradeoff. [Current evidence](selective-context.md).

The nine scanner false positives are resolved through narrowly scoped exact matches with five controls, not a disabled rule. Standalone compact relevance and real agent-task generalization remain open research gaps.

## v0.3.2: trained compression, routing and reuse

The three-seed mean of the trained 32-byte ITQ baseline improves on all three public datasets but still trails dense retrieval; one seed slightly regresses on ArguAna. The compact confidence policy failed qualification and remains disabled. `ProgressiveRouter` uses exact IDs when available and a precise backend otherwise; all 2,029 public rankings match dense by fallback. This does not repair information loss inside the stamp.

The initial session cache added CPU overhead. Restricting version checks to cached dependencies, while invalidating globally on every mutation, reduced that overhead in the follow-up replay. Both runs and original source snapshots are retained. Reuse reduces repeated transfer payloads in a shared-resolver scenario, not model evidence tokens. [Details and remaining research gates](progressive-routing.md).

Research release. Passing a controlled fixture is not proof of general robustness.

| Failure or gap | Response | Evidence / remaining limitation |
|---|---|---|
| Windows manifest separators prevented Linux evidence replay | Normalize recorded relative path separators in the verifier | Original manifests and numerical results retained; cross-platform CI rerun |
| Public 256-bit retrieval loses dense semantic information | Added residual-bound refinement as a separate richer index | All 2,029 dense orders recovered; the 32-byte code alone remains lossy; Faiss is faster in memory |
| Equal bit allocation may waste capacity | Seven layouts, 21 training runs and validation-only selection | Equal 64-bit facets retained; no allocation improvement claimed |
| Model violates coding output contract | Retained v1/v2 failures; typed extraction plus bounded deterministic rendering | v3/v4 selected workflows pass; declarative configuration only |
| Warmup and transport differ from final path | Matched v4 model settings and measured 32-byte transport | Original latency outlier retained; eight unique workflows |
| Pairwise scorer appears better on procedural fixtures | Paired task bootstrap and exact-field control | Intervals do not establish a reliable gain over ridge; no broad generalization claim |
| Standalone API lost the multidimensional design | Added normalized multi-view spherical stamps and per-facet scores | Scale invariance, antipodes, strict encoder schemas; `tests/test_spherical.py` |
| Similarity conflated with identity | Exact digests remain independent; new exact-facet activation guard | Forced identical-stamp/different-entity test; no identity decisions from approximate scores |
| Stamps confused with context graphs | Independent representation and explicit relationship APIs; separate ablations | Graph-only known-root control is strong and requires no stamp |
| v0.2 coverage reranking regressed on NFCorpus and ArguAna | New safe ranking API preserves supplied baseline order by default | Historical results retained; no claim that neural generalization is repaired |
| Historical learned selector had zero complete coverage | Added explicit structured requirements and exact budgeted cover | 60/90 feasible historical test cases recovered; 30 correctly abstained; extra typed metadata is required |
| First spherical scorer exploited inverse wording overlap | Retained v1; counterbalanced wording independently of labels in v2 | v2 is an audited follow-up, not an independent confirmation of v1 |
| v2 approximate scorer sometimes activated the wrong entity | Exact application constraints precede approximate activation | 480 positive and 480 missing-answer regressions replayed by `verify_spherical.py` |
| Learned threshold false activation | Preserve unguarded rates; require exact constraints where available | Threshold alone still has false activations; no universal threshold supplied |
| Stamp-only handoff omitted dependencies | Traverse explicit dependency closure or abstain | Stamp-only completeness was zero in the designed two-part task |
| Stale or changed dependencies reused | Bind edges to endpoint content digests and revisions | Same-revision content change, revision-only change, transitive traversal tests |
| Hidden context leaked through a handoff | Check every dependency role; return generic empty failure | Host must authenticate role; `affected()` is a trusted administrative API |
| Cyclic dependencies hang traversal | Visited-set traversal | 100 random graphs checked against independent fixed-point closure and reverse closure |
| Conflicting or oversized evidence silently truncated | Return empty insufficient packet | Explicit conflict, exact byte boundary and missing-context tests |
| Compact payload schema ambiguity | Versioned payload plus shared schema digest; bounded strict decoder | Bootstrap schema costs are counted separately; no authenticity or encryption claim |
| Revalidating an existing edge failed at full capacity | Allow replacement while rejecting new edges beyond the limit | 4,096-edge boundary test; original source preserved with historical evidence |
| Multimodal timer omitted per-call preparation | Preserve v1 diagnostics; start v2 timing before all routing preparation | Same prompts, model settings and seeds; rerun rather than reinterpret the old timings |
| Token reduction presented as latency improvement | Separate actual model tokens, transport tokens and end-to-end timings | Local historical QA latency remained worse despite smaller prompts |
| Internal model attention improvement implied | Keep external routing separate from attention research | No attention weights, KV-cache policy or neural architecture modified |
| Synthetic success treated as model understanding | Same-information exact-field control included | Exact-field control matches guarded method on supplied-field fixtures |
| Multimodal / edge claims untested | Maintain explicit modality and hardware matrix | Integration pilots do not establish perceptual quality or edge energy savings |
| Prompt injection and secret detection | Document as unsolved host/model responsibilities | No claim that a stamp sanitizes malicious instructions |
| Large-scale / concurrent production operation | Bounded in-memory reference implementation | No distributed consistency, multi-tenant service or high-load benchmark |
| Frozen lexical/semantic blend regressed on SciDocs | Added scope-bound certificate and dense fallback | Failure retained; remediation still needs an untouched preregistered evaluation |
| Precise hybrid gain attributed to the capsule | Documented separate compact routing and precise recovery stages | No causal capsule gain claimed; same-budget ablation remains open |

## Before a public release

1. Reproduce deterministic evidence and pass the full test/security suite on the exact commit.
2. Keep every failed run and distinguish prospective evaluation from regression repair.
3. Review data/model licenses and exclude private corpora, raw weights and secrets.
4. Complete independent workflow studies; benchmark supplied facets against exact-field and vector baselines.
5. Evaluate automatic facet extraction and missing/wrong relationships separately.
6. Confirm release scope and repository visibility with the owner. This candidate stays private.

No finite test suite can cover all adversarial inputs or every agent workflow. Each
new issue should receive a fixture, a named failure category, a fix where feasible,
and a rerun that preserves the original failure record.

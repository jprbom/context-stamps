# Failure ledger and release gates

Private research candidate. Passing a controlled fixture is not proof of general robustness.

| Failure or gap | Response | Evidence / remaining limitation |
|---|---|---|
| Public 256-bit retrieval loses dense semantic information | Preserve all 2,029 query comparisons; recommend dense retrieval for general search | Two-view lexical fusion worsened results; compression loss remains unresolved |
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

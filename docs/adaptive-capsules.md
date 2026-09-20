# Adaptive context capsules

Private v0.4.0 research candidate. By Prashant Jagtap.

The implementation makes three previously conceptual parts concrete: observed-facet compilation, a structured 256-bit profile and risk-budgeted routing. It does not establish that this profile improves public retrieval.

## Observed facets

`FacetCompiler` is a bounded deterministic baseline. It emits a value, source and rule for each observed facet. It never fills missing query facets. Text rules currently identify action terms, code/path-like entities, declared relationship phrases and version/date forms. Authority, policy, modality and workflow metadata are explicit host assertions.

This design keeps extraction inspectable. The rules will miss paraphrases and may mistake text that resembles a path, identifier or version. Retrieved content is untrusted; extracted facets do not grant access, establish truth or override exact constraints.

## Structured profile

| View | Bits | Purpose |
|---|---:|---|
| Semantic | 96 | Approximate meaning |
| Task | 32 | Intended operation |
| Entity | 32 | Named context identity |
| Relation | 32 | Declared dependency or workflow language |
| Temporal | 16 | Version and date routing |
| Authority | 16 | Host-asserted source class |
| Policy | 16 | Host-asserted policy scope |
| Modality | 16 | Host-asserted content type |

The total is exactly 256 bits. The allocation is an initial ablation target. It was not selected from public retrieval results. Complete document profiles require all declared views; partial queries retain only observed views through `FacetQuery`. Encoder identity, profile schema, exact IDs, authorization, versions and source evidence remain outside the 32 bytes.

## Risk-budgeted routing

`RoutingPolicy` records the one-sided upper error bound and accepted validation count produced by disjoint calibration. `ProgressiveRouter.search` attempts the compact path only when:

- the policy is enabled and matches scope and result count;
- validation accepted at least one item;
- the certified upper error bound is at or below the caller's risk budget;
- a compact backend exists and there are enough candidates.

The result reports why it used exact, compact, precise or abstain behavior. A caller cannot convert an over-budget policy into a compact exit merely by supplying high similarity scores.

## Recorded controls

`evidence/capsule-controls-v1/results.json` records eight authored compiler cases, 100 structured-codec round trips and five routing paths: authorized exact, denied exact, unqualified fallback, over-budget fallback and qualified compact exit. All passed locally. These are small software controls, not independent facet accuracy, robustness or retrieval benchmarks.

The next evidence gate is a separately annotated corpus containing missing facets, paraphrases, negation, stale versions and adversarial identifier-like text. The structured allocation then needs equal-storage comparison against a single-view binary baseline and dense retrieval. A compact path should be enabled only after disjoint calibration qualifies it for the same scope and query shape.

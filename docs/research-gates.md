# Research gates after the local pilot

## What v0.4.0 closes

Partial query facets are now explicit rather than fabricated; incompatible schemas are rejected and calibration scopes include the observed facet mask and weights. Dependency-selective cache invalidation replaces unnecessary whole-session clearing and is tested against uncached graph behavior. Three-seed quantizer results and precise fallback address honest comparison and quality protection, not standalone compact superiority.

The v0.4 baseline additionally extracts bounded observed facets with provenance, defines a structured eight-view profile and enforces a caller-supplied compact-error budget. Its eight authored extraction controls are software checks, not independent extraction evidence.

Remaining gates include independently evaluated or learned facet construction, profile ablation, independent real coding/research tasks, a useful qualified compact exit, semantic gains over dense retrieval, and distributed/edge/model-internal validation. [Adaptive capsule design](adaptive-capsules.md) and [earlier measured behavior](selective-context.md).

The private candidate is a tested context library and experimental scorer. It is
not a pretrained multimodal foundation model. These gates separate implemented
mechanisms from the broader research programme.

| Question | Required comparison | Acceptance evidence |
|---|---|---|
| Does a spherical representation help beyond supplied labels? | Exact-field filtering, dense vectors, binary single-view, same-information multi-view, graph-only | Fresh naturally occurring tasks; equal information and storage budgets; paired task outcomes |
| Can the system discover useful facets? | Hand-labelled facets versus automatic extraction versus no facets | Independent annotation, extraction error rates and downstream sensitivity to missing/wrong facets |
| Does training improve activation? | Uniform scores, frozen ridge, monotone pairwise, stronger semantic reranker | Train/validation/test separation by source/workflow; calibrated no-answer tests; intervals over independent tasks |
| Does it improve coding work? | Full context, conventional retrieval, exact dependency system, spherical-assisted harness | Real repository issues, hidden tests, bounded execution, complete call/token/time logs |
| Does it help image generation? | Same generator with oracle conditioning, ordinary retrieval and spherical context selection | Public licensed prompt/task sets, independent prompt-adherence and perceptual evaluation, equal inference settings |
| Does it help voice generation? | Same synthesis model and resolved text controls | Intelligibility/transcription measures, pronunciation outliers, human evaluation; generation parity alone is insufficient |
| Does it help video generation? | Same model/seed/settings with controlled context inputs | Temporal coherence and prompt adherence; a three-prompt integration pilot cannot establish these |
| Does it change internal attention efficiency? | Full attention, established sparse/memory routing, spherical block routing | Actual model implementation and training; quality, GPU memory, prefill/decode latency and fallback cost |
| Does it scale as a system? | Optimized vector/binary index plus metadata store | Concurrent workloads, updates, authorization, durability and end-to-end latency; packed-code scan alone is insufficient |
| Does it help edge deployment? | Device-native baseline on the same hardware | RAM, energy and latency on ARM/mobile targets; laptop GPU results do not transfer automatically |

## Training discipline

Prioritize evidence-selection utility and complete task outcomes over standalone
similarity scores. Include difficult same-entity/wrong-task and same-task/wrong-entity
negatives, stale evidence, missing answers and mismatched source versions. Audit
shortcut features before tuning. Keep exact identity and access constraints outside
the learned scorer. Freeze selection criteria before evaluating a new test cohort.

Encoder fine-tuning is justified only if same-information baselines show a gap
that a learned representation can plausibly close. A larger generative model is
not a substitute for missing evidence or incorrect relationship metadata.

## Release discipline

Keep the repository private until the owner reviews the claimed scope and its
evidence. A narrower release may be appropriate if its supported behavior is clear;
unverified capabilities must remain marked as research directions. No successful
pilot should erase earlier failed experiments.

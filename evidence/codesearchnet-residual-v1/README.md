# Precision residual behind a 256-bit routing code: predeclared study

The 256-bit spherical stamp is a routing handle, not a container for source or every relationship. On an edge device, a separately stored precision layer may be preferable to trying to force all retrieval evidence into 32 bytes. This study tests that engineering tradeoff after the [matching-aware PQ adapter](../codesearchnet-matching-v1/README.md) failed validation. The frozen encoder, training repositories, 30-repository validation cohort and unopened 40-repository test cohort remain the same. No scores from that new cohort have been read.

Three retrieval routes are fixed before validation:

| Route | Stored vector bytes/function | Retrieval |
| --- | ---: | --- |
| Dense float32 | 1,536 | Exact inner-product search |
| Scalar-quantized SQ8 | 384 | 8-bit/component direct search |
| PQ32 → SQ8 | 32 + 384 | PQ selects up to 20 candidates; SQ8 reconstruction reranks them |

PQ and SQ8 codebooks are trained on CodeSearchNet Python training vectors only. Their shared serialized bytes count in total index storage. The two-stage route keeps a 32-byte coarse code per function but requires an external 384-byte residual; it is not a claim that the full context fits into 256 bits. Unlike a standard float-vector refinement index, it does not store full float32 document vectors. [Faiss's index descriptions](https://github.com/facebookresearch/faiss/wiki/Faiss-indexes) specify 384 bytes/vector for SQ8 at 384 dimensions, and [its refinement guidance](https://github.com/facebookresearch/faiss/wiki/Implementation-notes) motivates reranking compressed candidates.

The fixed metric is exact-function top-1, top-10 and MRR@20. A missed target outside the top 20 contributes zero to MRR@20. Warm CPU timing includes both search stages and SQ8 reconstruction, but excludes encoder inference and index construction. Validation qualifies a route for the fresh cohort only when top-1 is at least dense's and MRR@20 is within 0.005 of dense. Among qualifying routes, select the highest MRR@20, breaking ties by lower lookup time. If neither qualifies, leave the new cohort unopened.

Passing that quality gate would only justify a generalization test. Actual edge usefulness also requires lower **total** storage after codebooks, acceptable p50/p95 end-to-end latency, prompt-token savings at matched verified coding success, and measurements on the intended device. For a small isolated repository, shared PQ codebook overhead may erase its per-function saving; a shared multi-repository index has a different break-even point. Neither route changes the model's internal attention or makes an SLM self-improving by itself.

Reproduce from the prepared private cache of the earlier CodeSearchNet study:

```text
python experiments/codesearchnet_residual.py train --cache PATH_TO_PRIVATE_CACHE --out evidence/codesearchnet-residual-v1
```

The `test` phase is guarded by the validation gate. Copyright (c) 2026 Prashant Jagtap. Repository code is MIT-licensed; third-party source and model terms remain with their owners.

## Committed validation checkpoint before first fresh test

| Route | Top 1 / 90 | Top 10 / 90 | MRR@20 | Warm lookup median, ms/query | Shared index bytes |
| --- | ---: | ---: | ---: | ---: | ---: |
| Dense float32 | 63 | 88 | 0.807257 | 0.007983 | 45 |
| SQ8 direct | 63 | 88 | 0.807328 | 0.007367 | 3,153 |
| PQ32 → SQ8 top 20 | 63 | 88 | 0.807650 | 0.628033 | 396,455 |

Both compressed routes met the predeclared *quality* gate. The selection rule chose PQ32 → SQ8 on its slightly higher MRR@20. Its lookup time is about 79 times dense's in this local implementation, so it does **not** meet a deployment latency gate. SQ8 direct stores 384 bytes per function and showed near-identical ranking and timing to dense on validation, with much smaller shared overhead. These are development results; the new cohort has not yet been read.

## First test on the frozen 40-repository cohort

The validation checkpoint above was committed before any of the 120 new queries were embedded or ranked. The first test run evaluated all three predeclared routes. `results.json` stores every repository's ranks and the embedding-cache checksum; `analysis.json` replays paired differences and total vector-index bytes.

| Route | Top 1 / 120 | Top 10 / 120 | MRR@20 | Vector-index bytes for 4,504 functions | Short warm lookup median, ms/query |
| --- | ---: | ---: | ---: | ---: | ---: |
| Dense float32 | 81 | 116 | 0.782434 | 6,918,189 | 0.007867 |
| SQ8 direct | 81 | 116 | 0.782712 | 1,732,689 | 0.007750 |
| PQ32 → SQ8 top 20 | 81 | 116 | 0.781779 | 2,270,119 | 0.734967 |

SQ8 direct and dense had identical top-1 and top-10 decisions on all 120 observed queries; SQ8 improved the exact rank once and worsened it zero times. That is a useful **4× vector-index storage reduction at this pooled 40-repository size**, not proof of general equivalence. The selected two-stage route also tied top-1 but was about 93× slower than dense in the short lookup measurement. Its shared PQ codebook makes it unattractive for one small isolated repository.

The short timing sample suggested SQ8 and dense were similar. A subsequent alternating-order, 50-repeat profile found material run-to-run and thread-count sensitivity:

| CPU threads | Dense median / p95, ms/query | SQ8 median / p95, ms/query |
| --- | ---: | ---: |
| 1 | 0.0042 / 0.0529 | 0.0088 / 0.0479 |
| 4, run 1 | 0.2160 / 0.5998 | 0.2215 / 0.6237 |
| 4, repeat | 0.1399 / 0.5575 | 0.2223 / 0.5790 |

Those are three-query batch times divided by three, across the same 40 repositories. They exclude encoding, index build, resolver work, power and model generation. The sustained single-thread median is about twice as slow for SQ8; four-thread results are noisy. We therefore claim a measured storage saving and preserved observed ranking, **not** a latency improvement. The test cohort is now inspected and cannot qualify a revised route as fresh evidence. An edge release needs target-device p50/p95 and energy measurements and verified coding-task outcomes.

Reproduce the evidence and timing profile:

```text
python experiments/codesearchnet_residual.py test --cache PATH_TO_PRIVATE_CACHE --out evidence/codesearchnet-residual-v1
python experiments/analyze_codesearchnet_residual.py evidence/codesearchnet-residual-v1/results.json evidence/codesearchnet-residual-v1/analysis.json
python experiments/profile_codesearchnet_residual.py --cache PATH_TO_PRIVATE_CACHE --out evidence/codesearchnet-residual-v1/latency-profile-1thread.json --threads 1
python experiments/verify_codesearchnet_residual.py --directory evidence/codesearchnet-residual-v1 --cohort evidence/codesearchnet-matching-v1/cohort.json
```

The final verifier replays all published top-1, top-10, MRR@20 and byte totals from per-repository records without downloading the benchmark source.

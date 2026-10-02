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

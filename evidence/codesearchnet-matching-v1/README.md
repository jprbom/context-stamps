# Matching-aware 256-bit code: predeclared local study

The earlier [CodeSearchNet Python study](../codesearchnet-quantization-v1/README.md) found that unsupervised 32-byte codes trailed dense retrieval on an untouched repository cohort. This follow-up tests whether a small query-side adapter trained against **quantized document reconstructions** can recover exact-function ranking. It is a coding-context retrieval experiment, not a general model-improvement or agent-completion claim.

The pinned MiniLM encoder, AST docstring removal, 384-dimensional normalized vectors, repository-local candidate sets and exact-function scoring remain unchanged. The PQ codebook is fitted only on training documents. A rank-16 residual adapter maps a query `q` to a normalized `q + (qU)Vᵀ`; `U` and `V` are learned from training-query/document pairs. Training uses in-repository candidates and cross-entropy, with a penalty for moving too far from the original query. No target description or source body is distributed in Git.

Predeclared comparisons:

1. Frozen dense and frozen 32-byte PQ, both with the original query.
2. Full dense and 32-byte PQ with the *same* adapter trained on quantized reconstructions. This separates the adapter's effect from compression.
3. Full dense and 32-byte PQ with an adapter trained on uncompressed documents. This tests whether quantization-aware supervision is different from ordinary ranking supervision.

Train on CodeSearchNet Python training repositories. The earlier 30-repository validation cohort is development data. A new 40-repository, 120-query cohort is selected by `freeze_codesearchnet_matching_cohort.py` from the original Python test split, excluding the 30 previously inspected test repositories. There are 43 eligible unseen repositories under the existing 50–1,000-candidate rule; the first 40 in the fixed hash order are selected. `cohort.json` records the repositories, selection rule and raw-file hashes. The raw code, descriptions, cached embeddings and learned weights remain outside Git.

The adapter and its codebook will be opened on the new cohort **only if** its PQ route reaches at least dense's validation top-1 and comes within 0.01 of dense MRR. That is a go/no-go rule for generalization testing, not a deployment criterion. Deployment would additionally need noninferior exact-function ranking on the new cohort, favorable total index bytes after shared overhead, and measured end-to-end latency at matched task quality. The new cohort will stay uninspected if validation fails.

The fixed training recipe is rank 16, six epochs, learning rate 0.002, temperature 0.07, AdamW weight decay 0.01 and query-drift penalty 0.1, with seed 20261002. Each training step ranks a query against 5–64 unique-description functions from its own repository. Training the float and PQ adapters uses the same groups and recipe. No hyperparameter sweep on the new cohort is allowed. The 32-byte codebook remains 32 eight-bit subquantizers; warm timings include query-adapter computation and search, but exclude encoder inference and index construction.

The first CodeSearchNet cohort is now inspected and will not be used for tuning this candidate. Any result here applies only to the defined repository-local exact-function task. An agentic coding claim requires paired executable patch tasks, verified success, prompt-token accounting, and cold/warm end-to-end latency.

## Validation result and stop decision

Both adapters trained for six epochs on 585 repositories and 6,315 usable query–function pairs. The loss fell for both, from 0.3659 to 0.2878 against float documents and 0.5971 to 0.5325 against PQ reconstructions. Each rank-16 adapter has 12,288 float32 parameters (49,152 bytes), in addition to the 393,302-byte PQ codebook. The weights and source text remain in the private local cache; `validation.json` records their SHA-256 values and training traces.

| Route | Top 1 / 90 | Top 10 / 90 | MRR | Warm lookup median, ms/query |
| --- | ---: | ---: | ---: | ---: |
| Frozen dense | 63 | 88 | 0.8073 | 0.0060 |
| Frozen PQ, 32 B/function | 58 | 89 | 0.7734 | 0.0112 |
| Float-trained adapter + dense | 64 | 88 | 0.8180 | 0.0178 |
| Float-trained adapter + PQ | 61 | 89 | 0.7992 | 0.0329 |
| PQ-trained adapter + dense | 63 | 89 | 0.8132 | 0.0164 |
| PQ-trained adapter + PQ | 59 | 89 | 0.7787 | 0.0228 |

The primary PQ-trained route missed the top-1 threshold by four queries and the MRR threshold by 0.0286. It improved the exact rank for nine queries and worsened it for 18 relative to dense. The float-trained adapter performed better with both dense and PQ, but its PQ route also missed the predeclared top-1 gate and increased lookup time. This is not evidence that quantization-aware training helps. The PQ seed differs from the earlier study, explaining the 58 rather than 59 unadapted PQ top-1 count on the same development cohort.

**Stop:** the new 40-repository cohort has not been embedded, ranked or scored. It remains available for a genuinely revised, predeclared architecture. No route is activated. Warm timings include adapter computation and FAISS search, but exclude the common encoder and index construction; they are small-batch local CPU measurements, not device-wide latency or energy figures.

The paired repository bootstrap in `analysis.json` puts the primary route's top-1 difference from dense at −4.4 percentage points, with a 95% interval from −11.1 to +2.2 points. This does not establish equivalence. Reproduce the validation run using the pinned embeddings and raw data prepared for the earlier study:

```text
python experiments/freeze_codesearchnet_matching_cohort.py --source PATH_TO_PARQUET_DIRECTORY --previous-manifest evidence/codesearchnet-quantization-v1/manifest.json --cache PATH_TO_PRIVATE_CACHE --out evidence/codesearchnet-matching-v1/cohort.json
python experiments/codesearchnet_matching_adapter.py train --cache PATH_TO_PRIVATE_CACHE --out evidence/codesearchnet-matching-v1
python experiments/analyze_codesearchnet_matching.py evidence/codesearchnet-matching-v1/validation.json evidence/codesearchnet-matching-v1/analysis.json
```

The script's `test` phase refuses to run when the validation gate fails. The first step records only cohort identities and checksums in Git; raw rows and model weights stay in the private cache.

Copyright (c) 2026 Prashant Jagtap. Repository code is MIT-licensed; source snippets and documentation in CodeSearchNet retain their own licenses.

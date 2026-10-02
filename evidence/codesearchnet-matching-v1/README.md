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

Copyright (c) 2026 Prashant Jagtap. Repository code is MIT-licensed; source snippets and documentation in CodeSearchNet retain their own licenses.

# CodeSearchNet Python: 256-bit function-localization study

This is a **derived repository-local retrieval task**, not the official CodeSearchNet challenge. The source is the Python train, validation, and test Parquet conversion of `code-search-net/code_search_net`; exact downloaded SHA-256 values and cohort selection are in `manifest.json`. We do not redistribute function bodies or docstrings. Underlying source snippets retain their own licenses.

The target is the exact function whose CodeSearchNet documentation string is supplied as a query. A repository contains 50–1,000 cleaned candidate functions, and three queries are selected per repository by a fixed SHA-256 rule. The 30 validation repositories and 30 test repositories are disjoint from one another and from the 4,585 training repositories in the prepared data. The test cohort was selected before any test score was read. This task is useful for measuring code localization, but it is easier than corpus-wide search and exact-function labels can penalize a semantically equivalent result.

To avoid a trivial answer leak, AST parsing removes function and class docstrings before indexing. Rows that cannot be parsed, have short or long descriptions, or still contain the query verbatim after normalization are excluded. Remaining identifiers and executable string literals may still provide clues. All routes use the same pinned MiniLM encoder, cosine-normalized 384-dimensional embeddings, repository candidates, and queries.

The 256-bit routes store 32 bytes per candidate. Ordinary PQ uses 32 eight-bit subquantizers. OPQ learns a rotation before PQ. The query-weighted route derives a reversible transform from training-query covariance: if documents use matrix `A`, queries use `A^-T`, so exact float inner products are unchanged. Its only intended effect is to move quantization error away from directions common in training queries. The transform and codebook are shared overhead; they do not fit inside each 32-byte stamp. None of these routes stores relational facets, dependencies, or recoverable source in the 256 bits.

Validation, before the test run:

| Route | Top 1 / 90 | Top 10 / 90 | MRR | Warm lookup median, ms/query | Shared overhead |
| --- | ---: | ---: | ---: | ---: | ---: |
| Dense float32 | 63 | 88 | 0.8073 | 0.0117 | 45 B index header |
| PQ, 32 B | 59 | 89 | 0.7814 | 0.0276 | 393,302 B |
| OPQ, 32 B | 58 | 88 | 0.7828 | 0.0863 | 983,197 B |
| Query-weighted PQ, exponent 0.25, 32 B | 61 | 86 | 0.7911 | 0.0882 | 1,572,950 B |
| Query-weighted PQ, exponent 0.5, 32 B | 52 | 87 | 0.7169 | 0.0874 | 1,572,950 B |

These times include query matrix multiplication where applicable and FAISS search, but exclude encoding and index construction. The dense index stores 384 float32 values (1,536 bytes) per candidate; the table's dense overhead is only its serialized empty-index header. Quantization reduces per-candidate vector bytes by 48×, but dense is faster and more accurate on this validation cohort. For a repository with only 50–1,000 candidates, shared codebooks and transforms can erase much of the storage saving. No compact route is activated from these results.

Reproduce with Python, PyTorch, sentence-transformers, FAISS CPU, NumPy, and PyArrow. Download the three Python Parquet files at the hashes in `manifest.json` outside the Git checkout. Then run `experiments/codesearchnet_quantization.py` in order with `prepare`, `embed`, and `fit`. Pass `--source` as the directory containing `python-train.parquet`, `python-validation.parquet`, and `python-test.parquet`; pass `--cache` as a private directory outside Git. `embed` uses a CUDA device. `fit --run-test` adds the held-out test results. The supplied `validation.json` is the checkpoint before that first test run.

The single held-out test will compare the predeclared routes. It is not a license to tune on the test set. If the query-weighted route fails to improve dense top-1, MRR, and latency jointly, we will leave it research-only. The next research step would be supervised matching-aware quantization on training pairs, with a new untouched repository cohort for qualification.

Copyright (c) 2026 Prashant Jagtap. Repository code is MIT-licensed; third-party dataset licenses remain with their respective owners.

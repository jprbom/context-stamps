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

Reproduce with Python, CUDA PyTorch, sentence-transformers, FAISS CPU, NumPy, and PyArrow 21. Download the Python train, validation, and test Parquet files from [the CodeSearchNet dataset](https://huggingface.co/datasets/code-search-net/code_search_net) outside the Git checkout. Name them `python-train.parquet`, `python-validation.parquet`, and `python-test.parquet`. Verify their hashes against `manifest.json`. Install the repository's `experiment` extra plus `pyarrow==21.0.0` in a CUDA-capable environment, then run from the repository root:

```text
python experiments/codesearchnet_quantization.py prepare --source PATH_TO_PARQUET_DIRECTORY --cache PATH_TO_PRIVATE_CACHE
python experiments/codesearchnet_quantization.py embed --source PATH_TO_PARQUET_DIRECTORY --cache PATH_TO_PRIVATE_CACHE
python experiments/codesearchnet_quantization.py fit --source PATH_TO_PARQUET_DIRECTORY --cache PATH_TO_PRIVATE_CACHE --out evidence/codesearchnet-quantization-v1/results.json --run-test
python experiments/analyze_codesearchnet_quantization.py evidence/codesearchnet-quantization-v1/results.json evidence/codesearchnet-quantization-v1/analysis.json
```

`embed` uses a CUDA device; training and warm lookup use CPU FAISS. The supplied `validation.json` is the committed checkpoint before the first test run.

The first held-out test compared those same predeclared routes, with no post-test retuning:

| Route | Top 1 / 90 | Top 10 / 90 | MRR | Warm lookup median, ms/query | Shared overhead | Candidates needed to offset overhead |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Dense float32 | 52 | 82 | 0.6934 | 0.0119 | 45 B index header | — |
| PQ, 32 B | 50 | 81 | 0.6742 | 0.0270 | 393,302 B | 262 |
| OPQ, 32 B | 51 | 82 | 0.6789 | 0.0873 | 983,197 B | 654 |
| Query-weighted PQ, exponent 0.25, 32 B | 47 | 80 | 0.6530 | 0.0873 | 1,572,950 B | 1,046 |
| Query-weighted PQ, exponent 0.5, 32 B | 39 | 79 | 0.5903 | 0.0907 | 1,572,950 B | 1,046 |

The validation-favored query-weighted route regressed on the untouched test. Across the 30 test repositories, its top-1 difference from dense is −5/90, with a repository-cluster bootstrap 95% interval of −13.3 to +2.2 percentage points. Ordinary PQ is −2/90, interval −7.8 to +3.3 points. These intervals do not establish superiority or equivalence. The 0.5-exponent route has a clearly negative interval. Exact rank improves for 8 queries and worsens for 25 with the 0.25-exponent route. See `results.json` for per-repository ranks and `analysis.json` for the paired intervals.

The storage break-even column compares the shared quantizer overhead with 1,536 versus 32 vector bytes per candidate; it excludes identifiers, metadata, full source, encoder weights, and query representation. The 1,046-candidate threshold for query-weighted PQ exceeds every selected test repository's 1,000-candidate maximum. Its larger transform also increases CPU lookup latency. At these repository sizes, query-aware covariance weighting is not a useful default.

All compact routes remain research-only. These data answer a narrower question than the project goal: whether a 256-bit semantic vector code improves exact code-function localization over the same encoder's dense vectors. They do not evaluate the spherical multi-facet stamp's relation map, context activation, agent workflow, model attention, code patch success, or edge-device energy. A supervised matching-aware quantizer could be trained on the training pairs, but it needs a new untouched cohort and paired latency/storage gates before it can be called an improvement.

Copyright (c) 2026 Prashant Jagtap. Repository code is MIT-licensed; third-party dataset licenses remain with their respective owners.

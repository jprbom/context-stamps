# Retrieval repair and workflow efficiency

Private v0.3.1 candidate. The routing code remains 256 bits. Quality-preserving semantic retrieval now has a separate refinement layer; the original 256-bit-only losses remain recorded.

## Fixing the public retrieval loss

`ResidualIndex` stores rowwise int8 vector codes, scales, conservative residual bounds and exact vector digests. For reconstruction z of vector x, the stored radius bounds `||x-z||`. Cauchy-Schwarz bounds the true query score around the approximate score. Every candidate whose upper bound can reach the kth lower bound is fetched and scored exactly. The implementation includes conservative float32 scan-roundoff bounds and float64 final scoring. Uncertain or tied cases can require all candidates; no fixed shortlist is assumed safe.

| Dataset | Dense reference nDCG@10 | Repaired path nDCG@10 | Identical top-10 order |
|---|---:|---:|---:|
| SciFact | 0.6451 | 0.6451 | 300/300 |
| NFCorpus | 0.3167 | 0.3167 | 323/323 |
| ArguAna | 0.5041 | 0.5041 | 1,406/1,406 |

The repaired path matched every dense reference ranking across 2,029 public queries. This repairs the tested system-level accuracy loss; it does not make the original 256-bit hash lossless. The test sets were previously inspected, and no relevance-label tuning was performed for this fix. The same pinned embeddings are reused.

At 384 dimensions, persistent index arrays require 432 bytes per item versus 1,536 bytes for a float32 vector: **71.875% smaller**. Full backing vectors are still required. Combined storage is therefore larger; lower resident memory is possible only when the backing store is external or managed separately. This is array accounting, not a measured process-RAM reduction.

The mean fractions of full vectors fetched were approximately 0.279% on SciFact, 0.396% on NFCorpus and 0.144% on ArguAna. All timings below use memory-resident backing vectors; remote/disk speedup has not been measured.

| Dataset | Faiss flat IP median ms | Faiss SQ8 median ms | Repaired path median ms |
|---|---:|---:|---:|
| SciFact | 0.612 | 1.107 | 3.953 |
| NFCorpus | 0.359 | 0.798 | 2.586 |
| ArguAna | 0.831 | 1.873 | 6.835 |

**Optimized Faiss is faster in these in-memory tests.** The measured trade-off is exact reference ordering with a smaller resident representation and fewer full-vector accesses. It is not an all-purpose speed improvement. Faiss SQ8 is a strong compact baseline. Its nDCG@10 was 0.6442/0.3156/0.5013; float32 score precision and tie handling also distinguish Faiss flat IP from the stable float64 reference. Small metric differences must not be presented as learned relevance gains.

Scalar quantization and refinement are established methods: [Faiss indexes](https://github.com/facebookresearch/faiss/wiki/Faiss-indexes), [Faiss refinement](https://github.com/facebookresearch/faiss/wiki/Implementation-notes). This project implements an explicit residual-bound refinement path and checks its behavior; it does not claim invention of quantization or Cauchy-Schwarz.

```python
import numpy as np
from context_stamps import ResidualIndex

# Replace these sample vectors with outputs from one pinned encoder.
rng = np.random.default_rng(42)
vectors = rng.normal(size=(1000, 384)).astype(np.float32)
vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
index = ResidualIndex(vectors, encoder_id="my-encoder-v1")
hits = index.search(vectors[17], encoder_id="my-encoder-v1",
                    eligible=np.arange(len(vectors)),
                    fetch=lambda rows: vectors[rows], limit=10)
print(hits)
```

The host authenticates the eligible rows and provides an immutable backing snapshot in requested order. Stale, reordered or changed backing vectors fail digest validation. Hashes bind data to the local snapshot; they are not signatures. The optional dependency is NumPy (`pip install -e ".[learn]"`).

## Agent-workflow results

Eight fresh fictional workflows, two repeats, three methods, four stages: plan, configure, reject an invalid dependency/access state, then resume after revalidation. There were 192 stage attempts and 144 actual Qwen2.5-1.5B calls in the final run. All methods received the same records and explicit constraints. The exact-graph baseline constructs no stamps.

| Method | Complete workflows / 16 | Actual input tokens, total | Mean workflow ms | Median ms | p95 ms |
|---|---:|---:|---:|---:|---:|
| Full current context | 10 | 91,224 | 595.37 | 607.08 | 727.78 |
| Exact graph | 16 | 15,702 | 424.68 | 427.60 | 473.03 |
| Spherical guarded, 32-byte code | 16 | 15,702 | 442.30 | 438.62 | 539.34 |

Workflow times include setup, state changes, context assembly and model calls. For 16 attempts, nearest-rank p95 equals the maximum. The spherical path used **82.79% fewer input tokens** and **25.71% less mean workflow time** than full context in this run. Output tokens were 1,047/1,056/1,056 respectively. No paid model API was used. Commercial cost depends on input/output prices and caching; local token counts are not a bill.

The exploratory paired bootstrap averages two repeats per workflow and resamples eight workflows. Spherical-minus-full latency was -153.06 ms, with interval [-192.20, -115.84] ms. Spherical-minus-exact was +17.62 ms, with interval [-11.04, 42.82] ms. This does not establish a benefit over exact lookup. Small procedural samples cannot demonstrate production-agent performance.

The 32-byte code is transported between host components and resolved before generation. Schema bootstrap is additional. The model consumes selected original evidence, not an opaque hash. Token savings arise from supplying the required evidence rather than all distractors.

## Failures retained and fixed

- Workflow v1: every workflow failed its code stage because the model emitted values instead of the requested code field.
- Workflow v2: a JSON schema enforced the field but did not make the string valid code; complete-workflow success remained zero.
- Workflow v3: the application validated typed values and rendered bounded integer assignments deterministically. Selected-context workflows passed; a latency outlier remains recorded.
- Workflow v4: retained that declarative renderer, used fresh fixtures and the exact 32-byte transport, and matched warmup settings to actual inference. No previous run was deleted.

This renderer is for declarative configuration, not arbitrary software generation. It rejects unrecognized fields, unsafe names, booleans and out-of-range values. Generated code is parsed for evaluation, never executed. Real repository issue resolution remains an open benchmark.

## Scalability and choice of path

The residual component has been tested at 1k, 10k and 100k rows, with one and four concurrent readers. Its initial 100k-row follow-up measured 77.14 ms median at concurrency one and 35.91 queries/second at concurrency four, using 30 queries per configuration. A separate comparative run includes Faiss flat IP and SQ8; see `evidence/scale-comparison-v1`. These are in-process read-only tests, not production-service validation.

Use exact graph lookup when supplied keys determine the target. Use an optimized dense index when in-memory latency is the priority. Evaluate residual refinement when limiting resident representation size or full-vector reads matters and an external backing store is available. Use the 256-bit code for established-schema routing and context handoffs, with exact identity and access checks outside the code.

The separate 100k-vector comparison measured:

| Index | Median ms, one reader | Throughput, four readers | Identical dense order / 30 queries |
|---|---:|---:|---:|
| Faiss flat IP | 9.50 | 285.23 queries/s | 30 |
| Faiss SQ8 | 22.27 | 162.55 queries/s | 9 |
| Residual refinement | 76.93 | 33.77 queries/s | 30 |

Order equality is stricter than recall or task accuracy. SQ8's 9/30 does not mean only nine useful answers. This test again favors Faiss for throughput; residual refinement trades additional computation for exact ordering and reduced full-vector access. It is not evidence of production scalability.

Distributed consistency, concurrent mutation, cold-storage latency, real autonomous-agent tasks, mobile energy consumption and modified neural attention remain unverified. Fine-tuning a base model would not itself fix information discarded by a binary hash or remove these systems requirements.

Evidence: `residual-v1` (diagnostic), `residual-v2` (corrected counters and roundoff-bounded scan), `stamp256-v1`, `workflow-efficiency-v1` through `v4`, and `scale-comparison-v1`. The v1 SQ8 counter counted scanned codes rather than full-vector refinement; its manifest flags that error. The v2 counter is corrected. Common query encoding and eligibility-list construction are excluded equally from retrieval timings; per-index validation and refinement are included.

# Precision-layer decision for coding context

Prashant Jagtap · 2 October 2026 · exploratory engineering diagnostic

This measurement uses the already inspected CodeSearchNet-derived validation and test embedding caches from the earlier [code-localization study](../codesearchnet-quantization-v1/README.md). It is not new generalization evidence, an official benchmark score, or a coding-task result. The [probe](../../experiments/probe_coding_precision.py) normalizes the cached 384-dimensional vectors and compares exact float32 dense top-10 with the repository's `ResidualIndex`: row-wise int8 scalar codes, a conservative residual bound, and exact refinement from a host-held float32 backing store. All 90 top-10 lists match dense in each cache because the index refines every candidate that could cross the kth lower bound. The check also guards against implementation errors.

For normalized query `q`, original vector `x`, and quantized reconstruction `z`, Cauchy–Schwarz gives `|q·x − q·z| ≤ ||q||₂ ||x−z||₂`. If the kth largest lower bound exceeds another row's upper bound, that row cannot enter the exact top-k. Otherwise, fetch its backing vector and score it exactly. This is a deterministic ranking certificate given valid bounds and a current backing snapshot. It is **not** a certificate that the retrieved code is relevant or that the model can solve a task.

| Inspected cache | Rows / queries | Dense vector bytes | SQ8 index bytes | SQ8 plus exact backing | Mean rows refined / 10 requested | Median of three 90-query sweeps, dense → verified SQ8 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Validation | 3,063 / 90 | 4,704,768 | 1,323,216 | 6,027,984 | 13.73 | 0.061 → 0.265 s |
| Test | 4,716 / 90 | 7,243,776 | 2,037,312 | 9,281,088 | 13.20 | 0.248 → 0.715 s |

These short sequential CPU sweeps are sensitive to machine load and do not include encoder inference, index construction, source loading, or a coding model. The backing store could move to slower storage to reduce resident RAM, but that would require measuring fetch latency and device storage. On these small in-memory collections, certified SQ8 is neither a memory nor a speed improvement over flat dense. The earlier [direct Faiss SQ8 experiment](../codesearchnet-residual-v1/README.md) did show a roughly fourfold *vector-index* storage reduction without retaining full vectors, but it did not provide this exact-ranking certificate and its sustained single-thread timing was slower.

An exploratory 1/4/16-group row-wise scale diagnostic tested whether allocating more scales reduces quantization error. On the 4,716-row cache, mean possible refinements fell from **13.19 to 12.60 to 12.01**; four groups add 56,592 float32 scale bytes, and 16 add 282,960. This float64 simulation omits production quantized-scan cost and is not a speed result. The small candidate reduction does not justify implementing a grouped variant as the default. [Numeric record](results.json) includes the cache hashes, all three sweep times, byte totals and bound diagnostics. Raw code and embeddings remain outside Git.

The research direction is staged, not a claim of a new quantizer. [Anisotropic vector quantization](https://arxiv.org/abs/1908.10396) emphasizes errors in directions that affect inner-product ranking; [Matryoshka representations](https://arxiv.org/abs/2205.13147) train nested capacities; [Quick ADC](https://arxiv.org/abs/1704.07355) and [Quicker ADC](https://arxiv.org/abs/1812.09162) improve product-code scan throughput with SIMD. Their published gains occur under their own scale, hardware and training conditions. None repairs a coding model that fails with full source, and none makes 256 bits contain arbitrary source, authorization or relationships.

Decision for this project:

1. Keep the 256-bit spherical stamp as an exact, version- and role-bound **routing identity**, not a self-contained memory or answer. Keep direct schema as the control because it gave identical prompts and outputs on the inspected native task.
2. Default to precise flat retrieval for small local code indexes. Reconsider direct SQ8 or hardware-optimized PQ only at a measured corpus-size or device-memory threshold, including codebooks, encoder, source store and full latency. Use an exact residual layer only when its backing-store tradeoff is justified.
3. Before training another selector, establish a direct-context coding baseline that passes fresh native tasks. Add independently declared task/output invariants and execution feedback that do not expose hidden tests; the first new merger task showed that field names alone do not supply merge semantics. Compare direct context and stamp activation with the **same** invariant checker.
4. Train a compact code representation only on repository-disjoint training pairs after the direct baseline works. A candidate may combine hard-negative ranking loss with teacher-score distillation and relation-completion loss, but must clear held-out retrieval recall *and* native pass-rate gates before activation. Preserve older-task retention and count p95 latency, full tokens, RAM, disk, update cost and energy on the target edge device.

The admission rule is a constrained comparison, not one weighted headline score. For each fresh task, record paired native outcomes `y_stamp` and `y_direct`, then estimate the repository-clustered interval for their mean difference. A compact route is eligible only if its lower confidence bound clears a predeclared non-inferiority margin, it introduces no unacceptable security or retention failures, and at least one predeclared cost measure improves with its own uncertainty bound. Model generation time, resolver time and repeated tool calls belong in the same end-to-end cost. A failed direct baseline is a diagnostic task for model/scaffold development, not evidence that quantization helped.

At measurement time, local C: and G: each had less than 250 MB free. No new weights or benchmark containers were started. More GPU training or a larger coding model should wait until workspace capacity is restored and the direct-baseline protocol is frozen.

Reproduce after preparing the earlier private embedding cache:

```text
python experiments/probe_coding_precision.py --cache PATH_TO_PRIVATE_CACHE --out evidence/coding-precision-decision-v1/results.json
```

The three timing sweeps will vary; exact top-10 agreement, byte accounting and cache hashes provide the stable checks. Third-party datasets and models retain their own licenses.

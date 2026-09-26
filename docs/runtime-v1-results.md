# Runtime-v1: lower execution cost and bounded workflow composition

By Prashant Jagtap · 26 September 2026

The corrected optimized cross-encoder preserves **3,677/3,677 top-10 rankings** against the frozen full-fusion reference across five previously inspected public collections. The deployed scope decisions remain unchanged: fusion for SciFact/FiQA and dense for NFCorpus/ArguAna/SciDocs. FiQA remains **0.4125 nDCG@10**, versus the earlier dense 0.3687 and hybrid 0.3888 controls.

In the paired local FiQA timing run, full fusion falls from **191.34 ms median to 115.95 ms with a cold passage-token cache and 89.17 ms warm**. Warm execution is 2.15× faster (53.4% less stage latency). This is retrieval/reranking execution on the same workload, not a new model-quality or universal speed result. Dense retrieval remains cheaper.

## What changed

The scorer retains FP32 weights and BF16 autocast, the full candidate union and the 512-token pair limit. Stable length sorting reduces padded work. A bounded exact-content cache avoids retokenizing passages. Long queries delegate full pair truncation to the pinned tokenizer. No relevant candidate is dropped for speed, and no shorter evidence limit is substituted.

The first candidate incorrectly handled the odd token in longest-first truncation when both sequences were long. It changed 231 ArguAna top-10 rankings and was rejected there. [Initial evidence and source](../evidence/runtime-v1/initial-candidate), [correction](../evidence/runtime-v1/correction.json). The corrected implementation was rerun across every query and timing scope. Small logit differences can still occur from GPU batching; ranking parity is an observed result on this regression suite, not bitwise equivalence on every future input.

Full FP16/BF16 weight conversion was faster in the seven-query development pilot but changed scores, so it was not promoted. Previous whole-layer student int8 was slower on the measured CPU. Precision is placed where the measured accuracy and operator behavior support it; lower bit width alone does not establish a speed benefit. See [Sentence Transformers inference guidance](https://www.sbert.net/docs/cross_encoder/usage/efficiency.html), [PyTorch numerical behavior](https://docs.pytorch.org/docs/main/notes/numerical_accuracy.html) and [tokenizer truncation semantics](https://huggingface.co/docs/transformers/main/pad_truncation).

## Quality and latency

| Dataset | Fusion nDCG@10 | Optimized nDCG@10 | Identical top-10 | Existing policy retained |
|---|---:|---:|---:|---:|
| scifact | 0.7269 | 0.7269 | 300/300 | 0.7269 |
| nfcorpus | 0.3609 | 0.3609 | 323/323 | 0.3167 |
| arguana | 0.5204 | 0.5204 | 1406/1406 | 0.5041 |
| scidocs | 0.1981 | 0.1981 | 1000/1000 | 0.2164 |
| fiqa | 0.4125 | 0.4125 | 648/648 | 0.4125 |

| Dataset | Original fusion p50 / p95 ms | Optimized cold p50 / p95 ms | Optimized warm p50 / p95 ms |
|---|---:|---:|---:|
| scifact | 148.40 / 174.79 | 104.86 / 120.40 | 96.64 / 109.83 |
| nfcorpus | 150.36 / 167.00 | 102.18 / 114.79 | 91.97 / 105.23 |
| arguana | 161.03 / 199.40 | 124.34 / 181.11 | 111.53 / 181.69 |
| scidocs | 152.65 / 181.31 | 85.64 / 111.41 | 70.66 / 88.39 |
| fiqa | 191.34 / 223.60 | 115.95 / 141.82 | 89.17 / 102.87 |

![Paired latency comparison](assets/runtime-v1-latency.png)

![Uploadable numeric comparison](assets/runtime-v1-table.png)

Each timing cell uses 20 evenly spaced queries with three measured repeats after warmup. Methods rotate in order and GPU work is synchronized. Cold/warm refers only to passage token IDs. Query encoding, model/index cold load, downstream generation, network and concurrent load are excluded. Full-fusion timings on scopes that retain dense are diagnostic, not their deployed cost. The historical four collections and FiQA are all regression data now; this optimization round has no new holdout.

## Expert routing was trained but did not qualify

A ridge model learns hybrid-minus-fusion relevance benefit from 14 observable candidate statistics, using 3,134 SciFact/NFCorpus training queries and equal domain weights. A fixed regularization coefficient of 10 is used. Threshold selection uses 290 tuning queries and requires nonnegative mean benefit in each tuning domain. A separate calibration gate requires at least 30 cheap exits, nonnegative mean benefit and a corrected lower gain bound of at least −0.001 against existing fusion.

The selected threshold, 0.01, produced only five proposed cheap exits on FiQA calibration, with mean change −0.000374. SciFact had none. Both failed; **no cheap-route scope is enabled**. The latency improvement comes from execution changes, not an unqualified retrieval shortcut. The int16-rounded router coefficient experiment preserves boundary decisions through a rounding-error bound and FP64 fallback. It retains original coefficients and is not an integer-kernel speed claim.

## Unified runtime and local readers

`ContextRuntime` composes `prepare_context`, `resolve`, `invalidate`, `record_outcome` and `run_verified`. It checks source/dependency revisions, roles, whole-packet byte/token budgets, bounded missing-evidence recovery, no-progress conditions and host-owned result verification. It supports explicit registered experts and exact result reuse. It neither updates its weights during inference nor executes plans found in context.

The two-reader fixture uses 12 fictional research configurations, each with initial/repeat/source-edit/repeat events. All modes see the same task and correct bindings. Full-scope input includes ten unrelated project notes. Prepared context keeps the target dependency closure; verified reuse additionally caches only previously verified exact computations. Readers are the pinned local Qwen2.5 1.5B Q4_K_M model and the existing locally fine-tuned Cortex 1.7B BF16 model. No new reader-model fine-tuning occurred. Local custom weights are not redistributed.

| Local reader | Mode | Verified | Model calls | Reported input tokens | p50 / p95 ms |
|---|---|---:|---:|---:|---:|
| cortex-harness:1.7b-v11 | full_scope | 36/48 | 48 | 85,246 | 428.4 / 608.3 |
| cortex-harness:1.7b-v11 | prepared_context | 42/48 | 48 | 14,974 | 316.3 / 379.2 |
| cortex-harness:1.7b-v11 | verified_reuse | 42/48 | 27 | 8,425 | 293.5 / 341.1 |
| qwen2.5:1.5b | full_scope | 48/48 | 48 | 86,062 | 258.7 / 353.9 |
| qwen2.5:1.5b | prepared_context | 48/48 | 48 | 15,790 | 177.4 / 207.8 |
| qwen2.5:1.5b | verified_reuse | 48/48 | 24 | 7,895 | 76.8 / 194.5 |

![Local reader tokens and verified outcomes](assets/runtime-v1-readers.png)

Input/output token counts come from the local model server. They are not assumed tokenizer estimates or dollars. The 50% repeated-request ratio is designed into this narrow fixture. Timing covers packet preparation and local HTTP/cache operations, with one run per mode and ordinary server prefix caching. These results do not establish general coding productivity, neural recursive intelligence or service-scale latency.

The Cortex reader improves from 36/48 verified outputs with full scope to 42/48 with prepared context, but six outputs still fail. They are retained and `run_verified` does not cache or release them as verified results. An [exact parser/calculator control](../evidence/runtime-v1/exact-tool-control.json) handles all 48 requests with zero model calls and rejects three malformed/ambiguous packets. It parses the declared source grammar and uses the existing exact-decimal tool; it does not read evaluation answers. For this structured task, direct calculation is the strongest control. This does not repair or fine-tune the reader weights, and the grammar is not a general natural-language solver.

## Spherical contribution: an explicit control

The 80-task synthetic ablation uses eight equal-semantic candidates per task; distractors differ in one declared facet. Same candidate set and top-1 budget throughout. Semantic float and semantic 256-bit selection each succeed 9/80. Full float facets and full 256-bit facets each succeed 80/80; removing relation yields 35/80 and removing entity 43/80. **Exact metadata also succeeds 80/80.**

![Same-budget synthetic facet ablation](assets/runtime-v1-facets.png)

This fixture isolates the value of supplied facet information and exercises the 32-byte representation. It does not show that the stamp beats exact metadata, dense semantic retrieval on natural documents or a graph system. Source text, schema and relationship maps remain external. A natural-workflow ablation and independent replication are still required.

## Reproduce and use

[Complete API example](unified-runtime.md) · [remaining milestone gates](unified-context-roadmap.md).

```powershell
$Work = (Resolve-Path '..\..\work').Path
$Python = Join-Path $Work 'controller-venv\Scripts\python.exe'
$env:OMP_NUM_THREADS = '4'
$env:MKL_NUM_THREADS = '4'
$env:OPENBLAS_NUM_THREADS = '4'
& $Python experiments/train_expert_router.py --work $Work
& $Python experiments/audit_runtime_latency.py --work $Work
# Run separately from GPU timing. Requires the two pinned local Ollama models.
& $Python experiments/benchmark_unified_runtime.py
& $Python experiments/ablate_spherical_runtime.py
& $Python experiments/verify_runtime.py
& $Python examples/unified_context.py
```

Reuse the verified caches and environment from the [controller-v2 runbook](controller-methodology-v2.md). Run reproduction in a separate checkout: experiment scripts write derived evidence. The archived initial failure is preserved in the published release and is not regenerated by the corrected scorer. The verifier checks recorded artifact/source hashes and reported gains; numerical reruns on different hardware require a new manifest rather than expected identical hashes. No GPU job overlaps the local-reader benchmark in the recorded run.

Evidence: [protocol](../evidence/runtime-v1/protocol.json), [latency](../evidence/runtime-v1/latency-summary.json), [router training](../evidence/runtime-v1/router-training.json), [reader observations](../evidence/runtime-v1/workflow-observations.json), [facet ablation](../evidence/runtime-v1/spherical-ablation.json). Implementation and documentation © 2026 Prashant Jagtap; MIT source, separately attributed evidence.

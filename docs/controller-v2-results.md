# Controller-v2: measured gains and remaining gaps

By Prashant Jagtap · 26 September 2026

The clearest new gain comes from a fixed hybrid/cross-encoder blend on **FiQA**, a locally fresh evaluation collection. The frozen blend scored **0.4125 nDCG@10**, against **0.3687 dense** and **0.3888 hybrid**. Its gate passed on 500 FiQA development queries before evaluating the 648 test queries. This is a measured retrieval improvement for that setup, not a universal model or stamp-only improvement.

The small learned model remains experimental. A 98,817-parameter no-attention model won neural-model tuning; the more expensive frozen fusion won overall method selection. On test data, the selected student beats hybrid on SciFact but trails it on the other four collections. More stable recurrence and richer supervision did not establish a broadly stronger learned ranker.

## What changed and what actually trained

The implementation adds balanced domain sampling, a top-10 metric-aware pair loss, separate query/document role features, larger candidate unions, pretrained teacher supervision, contractive recurrence and actual dynamic-int8 evaluation. [Mathematics, research references and complete local commands](controller-methodology-v2.md).

- Reused 552 SciFact and 2,582 NFCorpus training queries, with no new normalized query overlap found against FiQA calibration/test.
- Retained 290 tuning queries and 288 previously inspected calibration queries; added 500 FiQA calibration queries. No FiQA training data was used.
- Trained three recipes × three seeds × twelve epochs on an RTX 5080 Laptop GPU. Selection included epoch zero and used tuning data only.
- Generated 1,183,295 frozen teacher pair scores locally. Recorded retrieval/teacher preparation took 1,192.82 seconds; initial FiQA encoding/download is additional. Recorded training loops and tuning took 325.71 seconds; final evaluation is additional.
- The selected student is `balanced_ndcg-17`, epoch 11, with 98,817 parameters and a 395,708-byte safetensors checkpoint. Recurrent variants contain 100,929 parameters. Encoder, teacher, document vectors and index are additional.
- Evaluated 3,677 test queries across five collections. SciFact, NFCorpus, ArguAna and SciDocs are regression collections, not new holdouts.

Teacher predictions are auxiliary supervision, not human judgments. The recipe comparison isolates recurrence and distillation within v2; it does not separately attribute every v1-to-v2 change to one cause. All nine checkpoints, epoch histories and per-query results are retained.

## Ranking quality

Linear-gain nDCG@10; higher is better. The selected student is the same checkpoint for every dataset. “Gated” applies the method decisions frozen before test evaluation.

| Dataset | Dense | Hybrid | Selected student | Cross-encoder | Fixed fusion | Gated |
|---|---:|---:|---:|---:|---:|---:|
| SciFact | 0.6451 | 0.7225 | 0.7316 | 0.6869 | 0.7269 | 0.7269 |
| NFCorpus | 0.3167 | 0.3503 | 0.3441 | 0.3522 | 0.3609 | 0.3167 |
| ArguAna | 0.5041 | 0.5373 | 0.5171 | 0.4110 | 0.5204 | 0.5041 |
| SciDocs | 0.2164 | 0.2043 | 0.2003 | 0.1689 | 0.1981 | 0.2164 |
| FiQA — fresh local test | 0.3687 | 0.3888 | 0.3853 | 0.3676 | 0.4125 | 0.4125 |

![Retrieval results suitable for article upload](assets/controller-v2-results-table.png)

On FiQA, the fusion gain over dense is **+0.04384**, with a descriptive paired 95% bootstrap interval **[+0.02672, +0.06105]**. Against hybrid, the gain is **+0.02375**, interval **[+0.01375, +0.03403]**. FiQA's frozen cross-encoder alone scores slightly below dense; combining complementary scores helps here. This is not evidence that adding any cross-encoder always improves retrieval.

On SciFact, fusion improves over dense but its **+0.00441** gain over hybrid has interval **[−0.01331, +0.02256]**. Its calibration success therefore does not guarantee a statistically clear hybrid improvement on a new test sample. NFCorpus fusion looks better on test, but the earlier calibration gate was inconclusive, so the recorded policy still uses dense. ArguAna has no scoped calibration and prefers dense by policy; hybrid is the strongest measured test control. SciDocs fusion regresses, and dense fallback avoids that observed regression.

The gated policy is no worse than dense on these five aggregate test scores. That outcome follows from two accepted scopes and three fallbacks; it is not a theorem that the policy wins on all future domains or all queries. Existing SciFact/NFCorpus calibration is development evidence reused from v1. Only the FiQA development/test split was fresh locally in this round.

## Candidate coverage improved; ranking still has headroom

Expanding the union from 48 dense / 48 hybrid / 32 BM25 candidates to 96 / 96 / 64 raises the available top-10 ceiling on every collection. It uses no label-based positive injection.

| Dataset | Old oracle nDCG@10 | Expanded oracle nDCG@10 |
|---|---:|---:|
| SciFact | 0.9583 | 0.9787 |
| NFCorpus | 0.6177 | 0.6732 |
| ArguAna | 0.9787 | 0.9851 |
| SciDocs | 0.5548 | 0.6336 |
| FiQA | 0.7339 | 0.7969 |

The oracle sorts candidates using evaluation relevance labels. It is a diagnostic upper bound, never a deployable method. FiQA candidate recall rises from 69.12% to 76.08%; the remaining distance between actual ranking and the oracle shows that candidate recall alone is not the whole problem.

![Candidate ceiling comparison](assets/controller-v2-candidate-ceiling.png)

## The recurrence failure was addressed

The previous recurrent implementation accumulated residual changes and failed badly when given extra passes. The new model uses a fixed-input row-stochastic attention matrix and a contraction bounded by 0.5 per pass in the maximum row Euclidean norm. It trains across depths 2, 4 and 8; selected inference uses four passes.

For the tuning-selected recurrent trial, four-, eight- and 32-pass test nDCG is identical at reported precision on four collections. FiQA changes from 0.387119 to 0.387096. Across the first 32 queries of each collection, the largest eight-pass hidden-state distance from the 32-pass reference is below **4 × 10⁻⁷**. This is numerical stability evidence. The recurrent architecture still loses overall neural tuning to the simpler model, and stable states can still rank evidence incorrectly.

## Runtime and quantization are separate results

The quality improvement has a cost. Warm single-user retrieval plus reranking on FiQA takes **6.78 ms p50 / 7.80 ms p95** for dense, **13.69 / 17.95 ms** for hybrid, **19.60 / 23.73 ms** for the student and **193.81 / 213.97 ms** for fusion. Each method uses 20 fixed queries with three measured repeats. Query encoding, reader generation, network and concurrent load are excluded. These results do not demonstrate a latency or reader-token saving from fusion.

![Quality against online runtime](assets/controller-v2-quality-latency.png)

Whole-layer dynamic int8 was not numerically faithful to the FP32 student. Its input concatenates small embedding values with larger retrieval statistics. A follow-up engineering fix splits the input matrix into two algebraically equivalent FP32 projections, then quantizes only the embedding projection. Retrieval features and output remain FP32. No weights are retrained and no deployment decision is changed.

Across five now-inspected regression collections, split precision reduces mean absolute logit error **28.4–121.8×** compared with whole-layer int8. Its nDCG differs from FP32 by less than 0.0006 on each collection. Exact top-10 order is still not preserved for every query, and dynamic activation quantization remains batch-dependent. This is a numerical-fidelity result, not new independent retrieval superiority.

Forward-only median time remains about **0.60–0.62 ms** for split precision versus **0.52–0.53 ms** for FP32 on the tested CPU. It therefore does not earn a speed claim or replace FP32 by default. [Full numerical and timing audit](../evidence/controller-v2/split-quantization.json). The original whole-layer int8 results remain available.

## How to use these findings

Keep the existing precise retrieval path unless a method qualifies on your own scope. Add the spherical reference, exact source identity, dependency checks and selective reuse where those interfaces solve your workflow problem. The learned controller and teacher reranking are optional layers around that path.

Run the [local reproduction and inference example](controller-methodology-v2.md#run-on-the-local-rtx-setup) to inspect the trained checkpoint and scope decision. A financial QA corpus result does not authorize automated financial decisions; the benchmark here measures document ranking only. No answer-generation or action quality claim follows from nDCG.

The next useful experiments are cost-aware expert routing, a token-interaction student or encoder adapter, broader licensed training-only task coverage, and end-to-end workflow labels for verified evidence and actions. A same-budget ablation must also isolate whether the eight spherical facets and typed relations add value over the identical system without them. The present neural study does not establish that causal benefit. The [unified context runtime roadmap](unified-context-roadmap.md) defines acceptance gates for token efficiency, iterative reasoning, expert selection, multimodal use and edge deployment.

Evidence: [protocol](../evidence/controller-v2/protocol.json), [training](../evidence/controller-v2/training.json), [selection](../evidence/controller-v2/selection.json), [full metrics](../evidence/controller-v2/summary.json), [model card](../evidence/controller-v2/MODEL_CARD.md), [artifact terms](../evidence/controller-v2/ATTRIBUTION.md).

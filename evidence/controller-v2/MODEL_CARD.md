# Context Stamps controller-v2

Author: **Prashant Jagtap**. Research artifact, 26 September 2026.

## Intended use

An optional small evidence ranker for experiments around an existing retriever. It consumes a query embedding, candidate embeddings, six retrieval features and a trusted eligibility mask. It outputs ranking scores. It does not generate text, infer permission, verify truth, recover a document from a 32-byte stamp or modify a foundation model's attention.

**The learned model remains experimental.** The frozen deployment-method selection preferred the hybrid/cross-encoder blend. That blend qualified on SciFact and FiQA calibration; other scopes use dense. The selected student trails hybrid on four of five test collections. No production, general agent, edge-energy or industry-standard superiority claim is supported.

## Architecture and checkpoints

| Recipe | Parameters | Description |
|---|---:|---|
| `balanced_ndcg` | 98,817 | Width-64 no-attention residual ranker |
| `contractive` | 100,929 | Fixed-input attention with explicit bounded recurrence |
| `distilled` | 100,929 | Same recurrence with auxiliary centered teacher-score supervision |

All recipes use 384-dimensional normalized query/document vectors, their product and absolute difference, and six retrieval features. The score correction is bounded to ±1 around hybrid. The recurrence has a max-row-Euclidean contraction factor at most 0.5 for fixed inputs; this is a numerical bound, not a ranking or factuality bound. Training depths are 2/4/8; frozen evaluation depth is 4. Research forward calls allow up to 32, while the serving method restricts depth to the trained set.

The neural tuning winner is **`models/balanced_ndcg-17.safetensors`**, epoch 11, 395,708 bytes. All nine tuning-selected checkpoints are retained, with hashes and complete histories in [training.json](training.json). Recurrent files are 404,380 bytes. These sizes exclude external encoder/teacher weights, source text, embeddings and indexes.

## Data and optimization

Use 552 SciFact and 2,582 NFCorpus training queries. Reuse the earlier 290-query tuning set and 288-query calibration set, and add 500 FiQA development queries for calibration only. No FiQA training data enters fitting or tuning. Normalized query text is disjoint across fitting/tuning/calibration/test; this is not semantic deduplication or document-disjoint generalization.

Three seeds (7, 17, 29), twelve epochs, AdamW learning rate 0.0003, weight decay 0.01, clipping norm 1.0, BF16 training forward and FP32 loss. Every optimizer step samples eight queries per training domain with replacement. The main loss weights logistic document-pair errors by their delta linear-nDCG@10, with a 0.01 squared residual anchor. Distillation adds 0.1 centered MSE against normalized teacher predictions. No-positive shortlists receive no fabricated positive label.

Frozen encoder: `sentence-transformers/all-MiniLM-L6-v2`, revision `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`. Frozen teacher: `cross-encoder/ms-marco-MiniLM-L6-v2`, revision `233902d25c440f23af6f7d6e94d2946bac0bee0a`. Encoder truncation is 256 tokens; teacher truncation is 512. Text beyond those limits may be lost. Both models must be obtained separately under their terms.

The candidate set is the union of dense top96, hybrid top96 and BM25 top64, at most 256. Full-corpus eligibility and normalization happen before ranking; self IDs are excluded. The model does not consume the spherical facet/relation representation in this study. Stamp-specific causal benefit remains untested here.

## Evaluation

| Dataset | Dense | Hybrid | Selected student | Fixed hybrid/teacher fusion |
|---|---:|---:|---:|---:|
| SciFact | 0.6451 | 0.7225 | 0.7316 | 0.7269 |
| NFCorpus | 0.3167 | 0.3503 | 0.3441 | 0.3609 |
| ArguAna | 0.5041 | 0.5373 | 0.5171 | 0.5204 |
| SciDocs | 0.2164 | 0.2043 | 0.2003 | 0.1981 |
| FiQA | 0.3687 | 0.3888 | 0.3853 | 0.4125 |

Metric: linear-gain nDCG@10. FiQA is a fresh local test in this round; the other four are historically inspected regression sets. Absence from upstream pretrained-model data is not established. Incomplete relevance judgments, English-dominant tasks and only two training domains limit generalization. No choice of checkpoint, seed, recipe or deployment gate uses test scores.

The complete [summary](summary.json), compressed per-query rankings, [selection](selection.json), [candidate ceilings](candidate-coverage.json), [recurrence stress](recurrence.json) and [online runtime](online-runtime.json) include the failed comparisons. Actual CPU dynamic-int8 execution is an evaluation ablation, not a separately published quantized checkpoint or a default serving configuration. Its results and batch-shape differences are recorded separately.

A subsequent [split-precision audit](split-quantization.json) preserves the six retrieval features in FP32 while quantizing the embedding input projection. It reduces mean logit distortion 28.4–121.8× against the whole-layer int8 control, but still changes some rankings and is slower than FP32 in the measured forward test. No model fitting or frozen deployment-gate change follows from this post-evaluation engineering check.

## Loading and local use

Install the repository's `controller` extra in a working PyTorch environment. From the repository root:

```python
from context_stamps.contractive_controller import load_contractive

model = load_contractive(
    "evidence/controller-v2/models/balanced_ndcg-17.safetensors"
)
# Tensors from the same pinned retrieval profile:
# query_vectors: [batch, 384]
# candidate_vectors: [batch, candidates, 384]
# features: [batch, candidates, 6]
# eligible_mask: bool[batch, candidates], derived by the trusted host
# scores = model.score_candidates(
#     query_vectors, candidate_vectors, features, eligible_mask
# )
```

For a complete example using the verified local public caches, run:

```powershell
python examples/inspect_trained_controller.py --work '..\..\work' --dataset scifact --row 0
```

The [runbook](../../docs/controller-methodology-v2.md) defines feature order and normalization, training commands, hardware, source hashes and remaining experiments. Do not reuse this scope decision on another encoder, corpus, task or permission regime. Review checkpoints and dependencies before loading; safetensors avoids pickle execution but is not a sandbox. Host authorization, source versioning and prompt-injection defenses remain separate application responsibilities.

## Rights

Source implementation: copyright 2026 Prashant Jagtap, MIT. Derived public-data checkpoints and evidence: [CC BY-SA 4.0 with dataset/model acknowledgements](ATTRIBUTION.md). No private Cortex data, raw public text, external pretrained weights or participant feedback is included.

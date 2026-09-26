# Closing the controller gaps: methodology and local RTX reproduction

By Prashant Jagtap · 26 September 2026

This experiment follows the failed generalization and unstable extra recurrence in [controller-v1](local-rtx-controller.md). It changes the training objective, sampling, supervision, recurrence and evaluation controls. The model remains an optional evidence ranker around an existing retriever. The 256-bit spherical stamp remains an external routing reference; these experiments require full embeddings, retrieval features and source text for the teacher.

## Gaps and the changes implemented

| Observed gap | Implemented change | What the change cannot establish |
|---|---|---|
| NFCorpus supplied about 82% of training queries although model selection weighted domains equally | Eight queries from each training domain per optimizer step, sampled with replacement | Equal domain exposure does not create broad topic diversity or new labels |
| The previous listwise loss did not directly weight errors by their effect on top-10 quality | Linear-gain delta-nDCG@10 weighted pairwise logistic loss | Unjudged candidates remain uncertain negatives; this is not complete relevance supervision |
| Product/difference features removed some query/document role information | Separate normalized query and document vectors, their product and absolute difference, plus six retrieval features | A frozen single-vector encoder can already have discarded useful token-level distinctions |
| The previous teacher was a heuristic retrieval distribution | Pinned pretrained cross-encoder scores, an explicit cross-encoder baseline, and a distilled student ablation | Teacher predictions can be wrong; distillation does not turn them into human judgments |
| A ranker cannot recover a document absent from its shortlist | Expand dense/hybrid/BM25 union from 48/48/32 to 96/96/64; report recall and oracle top-10 quality | Extra candidates cost compute; the oracle is an upper bound, not achievable quality |
| Extra recurrent passes caused a severe transfer regression | Fixed-input attention with a contractive update, trained at depths 2/4/8 | Convergence does not guarantee relevance, factual correctness or recursive self-improvement |
| Four benchmark test sets had already been inspected | Add FiQA with development-only calibration and a frozen, locally fresh test evaluation | This does not prove the pretrained encoder/teacher never saw related data |
| Comparing only against dense retrieval could hide a weaker result than simple fusion | Include dense, BM25, hybrid, full cross-encoder and a fixed hybrid/teacher fusion | This is a bounded control set, not a survey of every current commercial or research system |
| Forward-only timing omitted retrieval and teacher tokenization | Measure online retrieval plus ranking; separate offline teacher labeling | Query encoding, reader generation, concurrency and networking still require application-level tests |
| The earlier int8 export reduced storage but reconstructed FP32 for execution | Evaluate real dynamic-int8 CPU Linear operators as a separate ablation | Quantization is not automatically faster or more accurate |

The [frozen protocol](../evidence/controller-v2/protocol.json) precedes v2 training. The run configuration records optimizer details and source hashes before optimization. All seeds, selected epochs and failed gates remain in the evidence directory. Four old datasets are regression checks; only FiQA test was locally uninspected before this round. Do not use these now-inspected test sets for another tuning round and still call them fresh holdouts.

This is a bundled methodology change, not a full factorial causal study. The three v2 recipes isolate adding contractive recurrence and teacher distillation on top of the common v2 setup. They do not separately isolate the effect of balancing, role features and ranking loss against v1. Candidate expansion has its own before/after ceiling measurement. Learned candidate attention is also distinct from the explicit typed `RelationMap`; this experiment does not supervise all eight spherical facets or learn a complete context graph.

## Research basis and the implementation differences

The metric-aware objective is motivated by Wang and colleagues' [LambdaLoss framework](https://research.google/pubs/the-lambdaloss-framework-for-ranking-metric-optimization/). Here, the weight for a preferred document pair is the absolute change in **linear-gain** nDCG@10 from swapping their current positions. A logistic pair penalty supplies gradients. This implementation is LambdaRank-inspired; it does not reproduce every LambdaLoss configuration.

[Balanced topic-aware sampling](https://arxiv.org/abs/2104.06967) demonstrates the importance of how training queries and margins are sampled. Our change is simpler: balance the two dataset domains, without topic clustering or that paper's dual-teacher training recipe. Topic-aware sampling remains a useful next experiment if additional training-only data is introduced.

The [Sentence Transformers distillation guidance](https://sbert.net/examples/cross_encoder/training/distillation/README.html) describes learning teacher scores or score differences. Our distilled variant minimizes centered per-list score MSE against normalized teacher logits. Centering removes arbitrary additive offsets; the same quantity is proportional to the mean squared mismatch of all within-list pair margins. The teacher is `cross-encoder/ms-marco-MiniLM-L6-v2`, frozen at the revision recorded in the protocol. It scores actual query/document text pairs, unlike the student, which sees cached embeddings and retrieval statistics.

[Deep Equilibrium Models](https://arxiv.org/abs/1909.01377) motivate examining fixed points rather than assuming extra depth is beneficial. Our implementation uses explicit, bounded iteration. It does not implement their root solver, implicit differentiation or constant-memory training claim.

## The recurrence and its guarantee

Let `u` be the bounded input projection for every candidate. Attention matrix `P` is computed once from `u`, with padded keys excluded. Its entries are nonnegative and each valid row sums to one. Let `a = tanh(d)` for a learned diagonal vector `d`. Each iteration is:

```text
h_next = 0.5 × u + 0.5 × tanh(P @ (h × a))
score  = hybrid_score + tanh(output(h))
```

Use the maximum row Euclidean norm across candidates. Row-stochastic averaging cannot increase that norm; multiplication by `a` has operator norm at most one, and elementwise tanh is 1-Lipschitz. Therefore:

```text
distance(F(h1), F(h2)) <= 0.5 × distance(h1, h2)
```

For fixed inputs and parameters, this is a contraction with a unique fixed point. The correction to the retrieval score is also bounded to ±1. The guarantee concerns numerical states, not ranking monotonicity: crossing a score tie can still change document order. Learned weights do not update during inference. The host must still enforce authorization, provenance and evidence sufficiency outside this model.

Training samples recurrence depth from 2, 4 and 8. Selection uses depth 4. A 32-pass stress test examines convergence after training, without using that test result to choose depth. The no-attention model is retained as a control because recurrence must earn its additional cost.

## Selection, calibration and statistical limits

Train only on the inherited SciFact/NFCorpus training partitions. Keep their tuning partitions separate. Select a checkpoint, recipe and seed by equal-domain mean tuning nDCG@10, including the untrained epoch-zero model. No test score participates in selection.

Select a candidate deployment method on tuning from dense, hybrid, teacher, a fixed equal blend of standardized hybrid/teacher scores, and the selected student. FiQA inherits the global tuning winner, then uses its development set for calibration. Each candidate must pass paired-bootstrap lower gain bounds against dense, hybrid and teacher, with one-sided alpha `0.05/3` and at least 50 calibration queries. Comparison with the identical method is exact equality and exempt. Failure uses dense. ArguAna and SciDocs have no scoped calibration and also use dense.

This correction addresses three comparisons within a scope. It is not a blanket family-wise guarantee across all domains, model-development cycles or future traffic. SciFact/NFCorpus calibration had already been inspected in v1; it is separate from fitting but is not pristine prospective evidence. Bootstrap intervals concern mean query-level retrieval scores under sampling assumptions. They do not certify every query or protect against distribution shift.

## Run on the local RTX setup

Use the working Python 3.12 CUDA environment described in the [v1 runbook](local-rtx-controller.md). The recorded environment is PyTorch 2.11.0+cu128 on an RTX 5080 Laptop GPU with approximately 16 GB VRAM. Reuse the verified v1 document/query caches. This pipeline uses direct Transformers loading with `trust_remote_code=False`, safetensors and pinned revisions; it does not need the SciPy extension blocked by the older environment's Windows application-control policy.

From a reviewed clone, using a PowerShell session:

```powershell
$Work = (Resolve-Path '..\..\work').Path
$Python = Join-Path $Work 'controller-venv\Scripts\python.exe'
$env:OMP_NUM_THREADS = '4'
$env:MKL_NUM_THREADS = '4'
$env:OPENBLAS_NUM_THREADS = '4'
& $Python -m pip install -e '.[controller]'
& $Python -c "import torch; print(torch.__version__); print(torch.cuda.get_device_name())"

# Use the hf executable belonging to the same Python installation.
hf download cross-encoder/ms-marco-MiniLM-L6-v2 `
  --revision 233902d25c440f23af6f7d6e94d2946bac0bee0a `
  --include '*.json' --include 'model.safetensors' --include 'vocab.txt'

# Downloads the public FiQA archive; reuses old data and cached encoder weights.
# Teacher scores and raw data stay in Work; only derived evidence enters Git.
& $Python -u experiments/prepare_controller_v2.py --work $Work
& $Python -u experiments/train_controller_v2.py --work $Work
& $Python -u experiments/audit_controller_v2.py --work $Work
& $Python -u experiments/audit_split_quantization.py --work $Work
& $Python -m unittest discover -s tests -p 'test_*controller.py' -v
& $Python experiments/verify_controller_v2.py
& $Python examples/inspect_trained_controller.py --work $Work --dataset scifact --row 0
```

For a fresh machine, first follow v1 preparation to obtain the exact public collections and frozen encoder cache; do not substitute different embeddings under the recorded hashes. `prepare_controller_v2.py` verifies old fingerprints, extracts only four whitelisted FiQA files and checkpoints expensive teacher partitions. Training resumes completed trials only when source/configuration and checkpoint hashes agree. A different GPU, kernel version or precision can produce numerical differences; compare metrics and document the new environment instead of promising bitwise CUDA identity.

The new training run uses 3 recipes × 3 seeds × 12 epochs, width 64 and AdamW at 0.0003. Each batch contains 8 SciFact and 8 NFCorpus queries. No-positive shortlists remain eligible for the anchor/distillation objectives. They are never assigned invented positive labels. Initial FiQA encoding, teacher labeling, model training and online inference are separate costs.

The inference example verifies local cache hashes, loads the tuning-selected safetensors checkpoint and prints ranked document IDs alongside the actual scope decision. It does not override a failed deployment gate. For your own retriever, call `load_contractive()` and then `score_candidates(query_vectors, candidate_vectors, features, eligible_mask)`. The six features, in order, are cosine similarity, eligible-corpus standardized dense score, the 0.75/0.25 hybrid score, eligible-corpus standardized BM25 score, reciprocal dense rank and reciprocal BM25 rank. Build them with the same pinned encoder and retrieval profile; changing those inputs defines a new scope needing separate calibration. The model accepts at most 256 candidates per query. Full embeddings and correct host eligibility are required; passing a 32-byte hash as if it were an embedding is not supported.

For the optional numerical-fidelity conversion, use `split_precision_controller(model)` from `context_stamps.controller_quantization` on a CPU FP32 checkpoint. Choose a supported PyTorch quantized backend first; the recorded Windows environment exposes oneDNN. Only the embedding input matrix is quantized. This preserves retrieval-feature precision, but still changes rankings and was slower than FP32 in this small forward benchmark. The separate post-evaluation audit preserves both whole-layer and split-precision outcomes without changing the frozen model/policy selection.

## Remaining work with the highest expected value

1. Add licensed, training-only task diversity and topic/margin-balanced sampling. Keep a new untouched external evaluation set and publish every failed trial. More copies of the same two domains are unlikely to solve transfer.
2. Train an encoder adapter or a token-interaction student if frozen pooled embeddings are the bottleneck. [ColBERTv2](https://arxiv.org/abs/2112.01488) is a relevant design reference: it retains token-level representations and uses residual compression to reduce index cost. Compare a small implementation's gain against its extra index size, encoding cost and an unchanged cross-encoder before integrating it. This is a proposed experiment, not an implemented ColBERT backend or a borrowed performance claim.
3. Build real evidence-sufficiency and workflow-action labels: missing dependencies, stale revisions, contradictions, permissions, next-tool decisions and verified completion. Relevance judgments cannot supervise these claims.
4. Evaluate a frozen reader on end-to-end factuality, source coverage, input tokens, total latency and tool-call count. Add coding tests and controlled mutations before extending claims to coding agents, image, speech or video generation.
5. Measure concurrent load, cancellation, memory growth, cache invalidation races, tenant separation and an ANN candidate backend. The current in-process exhaustive benchmark is not production-scale validation.
6. Measure whether spherical facets and relation maps improve decisions over the same system with those features removed. The current reranker experiment does not establish the unique causal benefit of the 32-byte stamp.

These are further experiments, not features already delivered. The source remains MIT with Prashant Jagtap's notice. Public-dataset-derived artifacts have separate attribution and terms in the experiment directory. Source text, private research, credentials and external pretrained weights are not published.

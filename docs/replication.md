# Frozen selector replication

> Historical v0.2 evidence. For the private v0.3.3 candidate, see [current results](selective-context.md), [scenario coverage](scenario-matrix.md) and [failures](failures-and-fixes.md).

The coverage/diversity gain on SciFact did **not** generalize to the two additional datasets. We recommend keeping a retrieval baseline and validating any reranking on the intended domain.

The selector was frozen at release commit `9b1900923de28bce3633bb9138e929db5405cc85`. The [protocol](../evidence/replication-v1/protocol.json) was committed before reviewing results. No new model fitting, parameter search or dataset-specific tuning was performed. The [runner](../experiments/run_replication.py), [manifest](../evidence/replication-v1/manifest.json), [summary](../evidence/replication-v1/summary.json) and [per-query rankings](../evidence/replication-v1/per-query.jsonl) are available.

## Results

All official test queries: NFCorpus has 3,633 documents and 323 queries; ArguAna has 8,674 documents and 1,406 queries. Values are nDCG@10, with linear relevance gains as in trec_eval's default graded nDCG.

| Method | NFCorpus | ArguAna |
|---|---:|---:|
| Faiss dense | 0.3159 | 0.5014 |
| BM25 reference | 0.3111 | 0.4191 |
| Coverage/diversity | 0.3038 | 0.4906 |
| Classical MMR | 0.2818 | 0.4832 |
| Frozen SciFact linear selector, seed 7 | 0.3194 | 0.2168 |

The paired coverage/diversity minus dense differences have exploratory 95% query-bootstrap intervals of **[-0.0211, -0.0017]** on NFCorpus and **[-0.0178, -0.0039]** on ArguAna. There are 2,000 resamples, with a fixed seed; these intervals are not adjusted for multiple comparisons. The small NFCorpus linear-selector gain has an interval spanning zero, [-0.0135, +0.0212]. The large ArguAna loss is evidence against treating that model as a general reranker.

## Low lexical overlap

Before evaluation, low overlap was defined as maximum word-set Jaccard similarity between a query and its positive documents at most 0.1. This measure depends strongly on document length: it includes 321/323 NFCorpus queries but only 34/1,406 ArguAna queries. It is a diagnostic stratum, not proof of semantic reasoning or an equally difficult subset across datasets.

| Method | NFCorpus low overlap: 321 | ArguAna low overlap: 34 |
|---|---:|---:|
| Dense | 0.3136 | 0.3645 |
| BM25 | 0.3101 | 0.0089 |
| Coverage/diversity | 0.3014 | 0.3073 |
| MMR | 0.2793 | 0.3279 |
| Frozen linear | 0.3173 | 0.0000 |

Coverage/diversity again lost to dense retrieval; the ArguAna subset is small. The full records include all per-query metrics and intervals.

## Data handling and controls

Every neural method uses the same pinned `all-MiniLM-L6-v2` encoder, normalized vectors and a top-100 dense shortlist. BM25 searches the full corpus. Identical query/document IDs are excluded for every method, following BEIR evaluation conventions. The model's default input truncation remains in effect; long arguments are not given a special encoder or chunking treatment.

The first run stopped because five ArguAna queries have positive document IDs absent from the supplied corpus. The corrected runner retains all five in overall metrics, preserving their relevance denominators, and marks their overlap as unknown. It does not count them in the low-overlap subset. Their IDs are recorded in `missing_positive_ids`. This was an evaluator correction, not a selector change. The completed run reused identical cached embeddings; manifest encoding times are cache-load times, not encoder throughput.

Measurements ran on Windows, Intel Core Ultra 9 275HX, with CPU embeddings. A separate local SLM experiment overlapped part of the run, so timing records are diagnostic and should not be treated as controlled speed comparisons. Per-query retrieval/selection latency excludes query encoding; index build time is separate. No raw dataset text or cached vectors are published. See [data attribution](../evidence/replication-v1/ATTRIBUTION.md).

## Reproduce

Download NFCorpus and ArguAna from the [official BEIR catalogue](https://github.com/beir-cellar/beir/wiki/Datasets-available), verify its archive checksums, and extract into separate `nfcorpus` and `arguana` directories beneath a local data directory. Each requires `corpus.jsonl`, `queries.jsonl` and `qrels/test.tsv`.

```bash
python -m pip install -e ".[experiment]"
python experiments/run_replication.py --data /path/to/data --cache /path/to/cache --out evidence/replication-reproduction
python experiments/verify_followup.py
```

The last command validates the committed records, not an arbitrary reproduction directory. Compare input hashes, model revision and metrics with the committed manifest. The runner reads the committed frozen protocol and rejects a changed selector hash. Rerun from the recorded source revision if the implementation later changes.

## What this changes

Version-aware evidence handling remains the practical focus. The ranking heuristic is domain-dependent. A subsequent research iteration should examine query/document-length effects, candidate recall and semantic coverage on a new development split, then test once on untouched data. These two test sets have now been inspected and must not be presented as untouched validation after tuning on their outcomes. More training is justified only with an explicit target, a leakage-resistant split and an improvement over a strong baseline.

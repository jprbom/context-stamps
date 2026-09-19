# Experimental selector model cards

> Historical v0.2 evidence. For the private spherical v0.3 candidate, see [current results](spherical-results.md), [scenario coverage](scenario-matrix.md) and [failures](failures-and-fixes.md).

Author: Prashant Jagtap. Code and synthetic coefficient exports: MIT. SciFact-derived evidence and coefficient exports: CC-BY-SA-4.0; see their directory attribution notice. These are six-parameter logistic rerankers (five feature weights and a bias), not generative SLMs or fine-tuned embedding models.

## Feature contract

Feature version 1 uses query-word coverage, query/document Jaccard overlap, a supplied retrieval score, bounded text byte length and full query-word containment. Inference needs only the standard library. The JSON loader validates schema, coefficient count and finiteness. Scores are uncalibrated ranking utilities; do not interpret them as correctness probabilities.

`base_score` must have the meaning used in training. Synthetic models use lexical-encoder binary agreement. SciFact models use pinned MiniLM cosine similarity. Loading a SciFact model on a default lexical store is out of distribution. The caller must match the retrieval setup; no automatic calibration is provided.

## Synthetic family

`evidence/synthetic-v1/selector-seed-{7,19,43}.json` uses binary relevance labels on original code, support and edge fixtures. Training, validation and test groups have separate fictional entity IDs. Test wording differs from training. Policy and API domains are held out as an additional distribution shift. Templates are related and do not establish broad language understanding.

`selector-restoration.json` is trained on whether restoring a document improves a deterministic required-fact coverage oracle after one fact is omitted. Every intervention is recorded. It does not measure a real generator's response or establish causal benefit in deployed agents. Models are compared with unchanged deterministic baselines on the same cases; unsuccessful models remain published.

## SciFact family

`evidence/scifact-v1/selector-seed-{7,19,43}.json` fits binary relevance from official training qrels and MiniLM candidates, with labeled positives added only during training. Unjudged candidates are treated as negatives, which introduces label uncertainty. Queries sharing a labeled relevant document with official test queries are excluded from training/validation; remaining connected groups are split by a fixed hash. Candidate documents are a shared retrieval corpus, not a document-disjoint corpus.

These tiny rerankers learn on biomedical retrieval labels and are unsuitable as safety classifiers. The base MiniLM encoder is frozen; no language-model or encoder fine-tuning was performed. Corpus text is not embedded in the coefficient files. Dataset provenance, hashes, split IDs, seeds, losses and measured outcomes are under `evidence/scifact-v1/`.

## Usage and limits

```python
from context_stamps.selection import LinearSelector, select_evidence

# Use only with the matching encoder / score semantics described above.
selector = LinearSelector.load("evidence/synthetic-v1/selector-seed-7.json")
# result = select_evidence(memory, query, reranker=selector.score,
#                          revisions=current_versions, required=essential_sources)
```

Models are opt-in research artifacts. Poor held-out results do not justify enabling them by default. Hard freshness and source requirements take precedence over learned relevance. They do not detect prompt injection, secrets, misleading claims or all material changes. No mobile/ARM device or energy benchmark has been measured. Separate local generated-answer fixtures are documented in local-tasks.md and spherical-results.md; they do not validate these historical selectors generally. Standard-library selector inference can run on CPU, but edge deployment still requires device-specific validation.

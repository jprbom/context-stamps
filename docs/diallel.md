# Diallel assessment and evidence selection

The useful question is whether combining signals selects a better evidence set at the same cost. Terminology alone does not create an independent signal.

For query-document relevance m, a mean prior g and residual s = m - query_mean - g + global_mean, the raw combination

`alpha*m + beta*g + gamma*s`

equals `(alpha+gamma)*m + (beta-gamma)*g + query_constant`. When beta equals gamma and alpha+gamma is positive, ranking is unchanged. Independently normalizing terms can change rankings but does not make them independent observations. The experiment includes this cancellation control and tests the identity. Classical MMR is attributed as MMR; its selected order is preserved in exported rankings.

The exact non-self mean similarity is `(D @ D.sum(axis=0) - row_squared_norms) / (n-1)`. This avoids materializing an n-by-n matrix. The implementation and explicit-Gram parity test are in `context_stamps/baselines.py` and `tests/test_selection.py`. This is a memory-efficient algebraic reformulation; it establishes no retrieval-quality improvement by itself.

## Integration

Context Stamps offers optional marginal query-term coverage and a redundancy penalty after hard freshness filtering. It can invoke a supplied reranker when the leading scores are close. These are disclosed heuristics inspired by established relevance/diversity selection, not a new diallel algorithm. The public experiment compares these operations with ordinary dense ranking and classical MMR. Validation can choose zero coverage weight; a negative result is retained.

The current greedy budgeted selector has **no claimed 1-1/e guarantee**. The cardinality-constrained guarantee for particular monotone submodular objectives does not automatically apply to this score, hard requirements or variable token costs. Log-determinant optimization, LP certificates and broad model ensembles are not necessary to ship the small runtime and are not represented as implemented here.

## Research interpretation

Treat main effects and interactions as an experiment-design question: compare individual components, combinations, and the no-freshness control. A better combined score must earn its complexity through held-out results. Statistical diallel models from genetics require assumptions and replication; our ablation matrix is not a biological GCA/SCA estimator.

References: [diallel linear-model review](https://doi.org/10.1007/s00122-020-03716-8), [original MMR paper](https://doi.org/10.1145/290941.291025), [Faiss binary indexes](https://github.com/facebookresearch/faiss/wiki/Binary-indexes). Private discussions, original notebooks and historical claims are not redistributed as validation evidence.

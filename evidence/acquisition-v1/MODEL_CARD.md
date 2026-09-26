# Judged-pool acquisition predictor

By Prashant Jagtap

**Status: failed qualification; experimental research artifact.** No scope passed the registered 5% conditional-error gate. Use the full candidate pool through the guarded API. Do not deploy the raw early-stop output as context sufficiency.

The selected `mlp32-seed29.json` has 1,187 learned parameters: 33 standardized features, 32 tanh hidden units and three sigmoid heads. It predicts complete judged-pool retention, remaining judged gain and next-batch judged gain. It is not an LLM, a semantic encoder, a trained answer verifier or a frontier-model replacement. Its size excludes the upstream encoder, index, corpus and runtime.

Training used 3,134 SciFact/NFCorpus queries, separate from 290 tuning queries and 788 calibration queries for this run. All data collections have prior project exposure. Six linear/MLP fits across three seeds were retained. The selected checkpoint was frozen before the calibration/regression phase. See `selection.json` for every seed and loss curve; see `protocol.json` for the fixed configuration and hashes.

The raw 0.90 policy lost recall on all five regression datasets. SciDocs and FiQA transfer failed severely; no default route was changed. Whole-trajectory risk bounds reject every tested scope. The evidence contains no new reader accuracy, actual model-token savings, latency improvement or production qualification.

Input rows must follow `hybrid-prefix-33-v1`; the target is `nonempty-judged-pool-retention-v1`. Only finite bounded float parameters in reviewed JSON files are accepted. The content digest is an integrity check, not a signature. Coefficients and public-data-derived evidence use [CC BY-SA 4.0 attribution](ATTRIBUTION.md). Original core source remains MIT.

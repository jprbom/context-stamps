# Local lexical relation selector

Developer: Prashant Jagtap. Original implementation and learned JSON policy: MIT. MuSiQue-derived data: CC BY 4.0 with upstream attribution. Experimental; inactive.

## Intended use

Research on choosing a small subset of paragraphs for local readers. Inputs are an explicit question and candidate titles/text. Twelve lexical/graph features feed a pointwise score; a thirteenth parameter controls four query-restarted diffusion steps. `pointwise.json` is the 12-parameter control. `diffusion.json` is the selected 13-parameter candidate. Neither file contains a language model or compressed source corpus.

## Training and evaluation

400 public answerable MuSiQue training cases; 64 separate calibration cases choose epochs/policy; 64 source-separated development question pairs evaluate answerable and unanswerable behavior. Both fits run locally on RTX in FP32. The selected diffusion alpha is 0.53216. No reader weights are updated. See `training-registration.json`, `training.json` and the [protocol](../../docs/multihop-local-learning.md).

Complete-support retention: BM25 17/64, pointwise 30/64, diffusion 38/64. On Qwen2.5 1.5B, exact answers are 8/64, 7/64 and 10/64 respectively; full context reaches 12/64. Diffusion answers 27/64 unanswerable cases. On a post-hoc Qwen2.5-Coder 7B reference, diffusion exact answers fall to 3/64 versus full context 8/64. End-to-end benefit is unqualified.

## Limits and deployment

Literal title links are fallible lexical cues. Propagation mass is not answer confidence or proof of sufficiency. The selector cannot recover missing source passages, authorize access, validate facts, defeat prompt injection or replace a domain verifier. Apply authorization before candidate construction. Text length, feature computation, candidate count and graph storage add cost beyond thirteen parameters. No large-corpus scaling, quantized rank parity, older-skill retention, energy measurement, complete device-memory bound, multimodal quality or production edge result is established. Do not activate these weights as an unattended self-improving policy.

# Runtime experiment attribution

Original implementation, documentation, fictional workflow/facet fixtures and exact-tool control: copyright 2026 Prashant Jagtap, MIT License. Preserve the owner's copyright and source license.

Public-dataset-derived router parameters, development features, ranking records, latency observations and result figures are distributed under CC BY-SA 4.0 with the same dataset/model acknowledgements as [controller-v2](../controller-v2/ATTRIBUTION.md). This includes SciFact, NFCorpus, ArguAna, SciDocs and FiQA and their original authors; the frozen MiniLM encoder/cross-encoder are obtained separately. Dataset and upstream model terms are not replaced by the source license.

The original synthetic reader/facet records and corresponding original figures remain MIT. They contain fictional research configurations, constrained integer outputs and response hashes, not private source documents or unconstrained model-generated text. The local Qwen2.5 1.5B model and existing Cortex 1.7B local fine-tune were used for inference only. No weights, private training data, system prompt templates or external pretrained models are distributed here. Their exact local identifiers/digests and numerical formats are in the workflow manifest.

The `initial-candidate` directory preserves the rejected long-query truncation behavior, its source snapshot and full first-pass measurements. The current corrected source and second full regression are outside that directory. Current and original runs are engineering regressions on already inspected datasets; neither is represented as an untouched test or a universal guarantee.

The new ridge router is an experimental policy, not a foundation model or a replacement for the controller-v2 checkpoint. It approves no cheap-route scopes after calibration failure. No reader/base-model fine-tuning, neural mixture-of-experts training or self-updating weights occurred in this runtime experiment.

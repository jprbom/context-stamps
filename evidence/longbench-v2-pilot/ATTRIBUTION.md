# LongBench v2 pilot attribution

Experiment implementation and commentary: copyright 2026 Prashant Jagtap, MIT.

The three prompt files in `upstream/` and the answer extraction logic in
`experiments/longbench_eval.py` originate from LongBench v2, copyright 2023
THU-KEG & Zhipu AI, MIT. Their complete notice is retained in
[upstream/LICENSE](upstream/LICENSE). The prompts are unmodified; the parser
preserves the upstream match precedence and case sensitivity.

Code revision: `ef5ccc4bdcb1d505455517e4b419a50bb959862a` in
[bys0318/LongBench-v2](https://github.com/bys0318/LongBench-v2/tree/ef5ccc4bdcb1d505455517e4b419a50bb959862a).

Dataset: [zai-org/LongBench-v2](https://huggingface.co/datasets/zai-org/LongBench-v2/tree/2b48e494f2c7a2f0af81aae178e05c7e1dde0fe9),
revision `2b48e494f2c7a2f0af81aae178e05c7e1dde0fe9`. The dataset card declares
Apache-2.0. Its 465,490,535-byte `data.json` is downloaded separately and not
redistributed here. Published evidence contains task identifiers, labels,
configuration, hashes, short model responses and measurements. No source
contexts or question/choice text is bundled. The Hub split name is `train`;
that name does not mean these benchmark examples were used for gradient training.

Tokenizer/model family: [Qwen2.5-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct/tree/989aa7980e4cf806f80c7fef2b1adb7bc71aa306),
Apache-2.0. Tokenizer revision `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`.
Installed Ollama GGUF weights are a separately identified Q4_K_M local control;
their exact digest and metadata hashes are in the plan. Neither tokenizer nor
model weights are redistributed. These are third-party models, not models
trained by this project. Original third-party rights remain with their owners.

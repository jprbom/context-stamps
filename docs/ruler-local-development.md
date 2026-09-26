# RULER: local context treatment protocol

This development experiment asks whether a small local reader benefits from explicit context operations, and whether the runtime adds value beyond the same operations used directly. It does not qualify a self-improving model or establish an effective context length.

## Protocol

Use all 13 RULER-v1 tasks at nominal 4,096 and 16,384 token limits, initially two generated examples per task and length. This gives 52 inputs and 208 paired reader calls. The small sample count is for implementation and failure diagnosis. Longer lengths, fresh seeds, larger samples, model comparisons and deployment-device measurements remain separate milestones.

The four arms use the same installed Qwen2.5 1.5B Q4_K_M model, tokenizer, raw ChatML template, deterministic decoding and 256-token output cap:

| Arm | Context given to the reader |
|---|---|
| Full | The complete native question, including native worked examples and answer prefix |
| BM25 | Ranked 512-token chunks with 64-token overlap, within a 4,096-token complete-input budget |
| Direct operations + reader | Literal key matches, ordered variable dependencies or full-scan word counts; otherwise full context |
| Runtime + reader | The same selected material, managed as versioned context nodes with explicit dependency closure, provenance and a resolved 32-byte receipt |

A fifth control returns the deterministic operation's answer with zero model calls where its explicit grammar applies, and abstains on document QA. This control must be reported even if it outperforms the reader.

The operations are deliberately benchmark-aware. A correct exact lookup or word counter is useful, but is not new model intelligence. The runtime does not pretend that substring matching, a stamp collision or a retrieval score proves semantic sufficiency. QA keeps all source text. Ambiguous counts, missing keys and unsupported assignments also fall back. No answer key or answer-position metadata is available to selection or generation.

The receipt is a reference to locally retained evidence. The reader receives materialized text, including provenance. Report its actual tokenizer count; do not call the 32-byte receipt a 32-byte model context. No computation-result reuse is enabled in this experiment.

## Native sources and deviations

Upstream now recommends its [RULER-v1 NeMo pipeline](https://github.com/NVIDIA/RULER/tree/rulerv1-ns). The pinned NeMo preparation source still invokes the generators in the main RULER branch. This experiment calls those generators directly using subprocess argument arrays, without running the wrapper's shell-based package installation or unpinned clone.

- RULER generator revision: `c3f5e3b4f87f97e048793bb510a3a6b19a46bf3a`.
- NeMo preparation/scoring revision: `f4a3fd8e524acd9abd1fea4387e8f179f6d51cf3`.
- Existing Qwen tokenizer revision: `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`.
- Local model digest: `65ec06548149b04c096a120e4a6da9d4017ea809c91734ea5631e89f96ddc57b`.
- Preparation seed and generation seed: 71. Treatment order is deterministically shuffled within each input.
- NeMo's 50-token template reserve and the generator's base-template reserve are retained. This means nominal 4K CWE follows the native generator's below-4,096 branch, with different repetition counts. Report this detail when comparing other protocols.
- QA uses the first two native questions at both lengths. These observations are correlated across lengths and may have appeared in model pretraining. They are not independent, contamination-free domain holdouts.
- The original NIAH `index` is an answer character offset. Our unique IDs use task, length and row position instead; the offset is excluded from public inputs.

The metric matches the pinned NeMo scorer: case-insensitive substring recall for structured tasks and any accepted answer alias for QA. It can give partial credit and false-positive credit. A score of 1 is called **complete native credit**, not verified factual correctness. Ten adversarial metric cases check parity with the two original scorer functions. Raw input truncation is forbidden; output truncations and execution errors remain in the denominator.

This is a custom local backend and context-treatment experiment. Its shorter output cap, sample count and model wrapper differ from the official NeMo invocation. It must not be submitted or described as a directly comparable official leaderboard result.

## Reproduce on a local RTX system

Use a Python environment with the project's development dependencies and the pinned preparation packages in `experiments/ruler-requirements.txt`. GPU inference uses the already installed local Ollama model; no remote model, paid provider or judge is used. Public dataset downloads are a separate explicit preparation step. Keep supporting preparation and scoring on CPU and only one inference workload on the GPU.

Clone RULER outside this repository and check out the generator revision above. Clone NeMo-Skills and check out the scoring revision. Set `$Source`, `$Assets`, `$Data`, `$Tokenizer`, `$Scorer` and `$Run` to absolute paths outside this publishing checkout. The tokenizer directory must contain the reviewed local Qwen tokenizer/configuration; it need not download new weights.

```powershell
python experiments/ruler_native.py fetch --source $Source --output $Assets
python experiments/ruler_native.py prepare --source $Source --assets $Assets --tokenizer $Tokenizer --output $Data --count 2 --seed 71 --lengths 4096 16384
python experiments/ruler_local_eval.py prepare --data $Data --tokenizer $Tokenizer --scorer $Scorer --output $Run
python experiments/ruler_local_eval.py run --output $Run
python experiments/ruler_local_eval.py score --data $Data --output $Run
```

Each output directory is fresh. Failed attempts remain intact; do not overwrite them to make a run look successful. Preparation records exact source, input, tokenizer and package hashes. Generation does not open scoring keys, and refuses changed source/model/input identities. HTTP generation is restricted to loopback with proxy and redirect handling disabled. Scoring follows generation and retains all four arms for every task.

Report native score by task and length; complete-credit counts; new failures; complete input/output token totals; model-call time plus CPU preparation time; output truncations and transport/protocol errors. Warm-up is recorded separately. This first protocol does not measure streamed time-to-first-token, energy, concurrency throughput, thermal sustainability or edge-device performance.

## Data and rights

NVIDIA RULER and NeMo code are Apache-2.0, with their attribution and [license](RULER-APACHE-2.0.txt) retained. SQuAD and HotpotQA are [CC-BY-SA-4.0](https://hotpotqa.github.io/); see also the [SQuAD distribution notice](https://rajpurkar.github.io/SQuAD-explorer/). Paul Graham essays retain their underlying rights. Downloaded source text, generated long prompts and complete local response logs remain outside this repository. The public evidence contains reviewed short predictions and references, with QA attribution, plus metrics and source hashes. It excludes long prompts and reconstructable context-token IDs; the project's MIT license does not relicense these corpora.

## Local improvement after this experiment

Failure analysis should first improve the context operation, routing policy or verifier when that is the cause. Weight updates are appropriate only when independently verified failures persist with sufficient context and tools. Freeze a new candidate, evaluate on new domain cases and retention cases, and measure learning cost before activation. Passing this development set after inspecting its failures is regression coverage, not fresh generalization evidence.

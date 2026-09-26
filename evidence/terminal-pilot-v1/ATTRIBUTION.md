# Terminal pilot sources and scope

Authored driver, sandbox wrapper, tests and documentation: copyright 2026 Prashant Jagtap, MIT license.

The benchmark material and portions reproduced in prompts, tool traces and native test reports come from [Terminal-Bench 2.1](https://github.com/harbor-framework/terminal-bench-2-1), revision `7131e4375048a0e408a8fb404b5f499d726b695b`, under its [Apache-2.0 license](upstream/LICENSE). Task metadata credits Alex Shaw for `cancel-async-tasks` and Orfeas Menis Mastromichalakis for `log-summary-date-ranges`. The source manifest binds every downloaded instruction, fixture, grader and reference-solution file. Benchmark canaries and evaluation-only restrictions remain applicable; none of this task material or these traces may be silently repurposed as training data.

The raw task tree, reference solutions and model weights are not bundled. The published records include public task instructions, generated commands/answers, selected task output, native grader reports, hashes and resource observations. They contain no private Cortex documents or user corpus. Read third-party terms separately from the MIT core.

The local Qwen2.5 1.5B model and its tokenizer originate from [Qwen](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct), under Apache-2.0. Exact local GGUF and tokenizer identities are in each plan. The model was used for inference only; no model weights were modified or redistributed.

Harbor is an external Apache-2.0 framework; Python and verifier packages retain their own licenses. Runtime containers are local and are not published as repository artifacts. The old and patched dependency audits are retained, including the duplicate report of one advisory. These are dated observations, not guarantees of future security.

Both attempts are development evidence. The corrected pilot passed zero of two tasks. No runtime treatment, local learning gain, SLM fine-tuning, frontier comparison or official leaderboard score is claimed.

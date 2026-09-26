# Stronger local coding reference

Authored harness and documentation: copyright 2026 Prashant Jagtap, MIT.

These are development controls on the same two Terminal-Bench 2.1 tasks as the [earlier pilot](../terminal-pilot-v1/ATTRIBUTION.md). That file identifies the upstream revision, authors, [Apache-2.0 task license](../terminal-pilot-v1/upstream/LICENSE) and evaluation-only boundary. Public task instructions, generated code and native test reports are included here. No task material, model output, reference solution or grader was used for training. Raw task sources and solutions remain outside Git.

The separately installed `qwen2.5-coder:7b` model is Q4_K_M, identified by its full local digest in `plan.json`. Its [upstream model](https://huggingface.co/Qwen/Qwen2.5-Coder-7B-Instruct) is Apache-2.0. The separate tokenizer revision and hashes are recorded in `tokenizer-source.json`; the license hash matches the installed model's license. Weights and tokenizer files are not bundled. This is inference only, not a Context Stamps trained model.

The stronger reference passed one task and failed one, with 12 steps used on each. No official leaderboard, runtime improvement, autonomous learning, edge-device qualification or independent replication is claimed. This run uses the hardened supervisor and rebuilt verifier images; native assertions and model/task budgets match the corrected earlier control.

# Minimal schema context on an inspected coding task

Prashant Jagtap · 2 October 2026 · **development evidence only**

The earlier [path-contract diagnostic](../terminal-coding-contract-v1/README.md) found that Qwen2.5-Coder 7B copied irrelevant legacy behavior and missed the requested mean-temperature computation. We tested a smaller view on the same already inspected Terminal-Bench 2.1 task, `modernize-scientific-stack`. The host read only the authorized CSV header, checked exact source hashes, and supplied the same path contract. No CSV data rows or configuration values entered this view.

In the first four-arm replay, instruction and paths alone failed: the generated script guessed a `station` column where the CSV has `station_id`. The 32-token larger schema view passed the native verifier. Full source and full stamp relationship packets still failed after their bounded path retries. A second replay again passed with direct schema and with exact 32-byte stamp-bound activation of the **same schema bytes**. The two schema prompts and responses matched exactly in that replay; the stamp did not cause a model-quality gain.

| Inspected-task arm | Native pass | Total input tokens |
| --- | ---: | ---: |
| Instruction + path contract only | 0/1 | 606 |
| Direct CSV schema | 1/1 | 638 |
| Direct three-file content, with repair | 0/1 | 4,048 |
| Full stamp relationship closure, with repair | 0/1 | 4,848 |
| Exact stamp-bound CSV schema, second replay | 1/1 | 638 |

The second replay's host stamp-schema binding and activation took 0.95 ms once, excluding tokenizer/model work; this is not a reliable latency estimate. The 32-byte stamp is a handle to a host-held source and schema, not a compressed CSV. Source bytes, digest, role and schema definition remain outside it. All generations used the same local Qwen2.5-Coder 7B, temperature 0 and seed 7. Empty submissions failed and the reference solution passed the isolated native verifier. The source task was already inspected and the prompt was developed against it; these passes are **not fresh generalization evidence**. [Numeric run record](results.json) contains external plan/summary hashes and all arm outcomes.

The first unopened task-family check is [multi-source-data-merger](../terminal-coding-schema-fresh-v1/README.md). It must decide whether schema-only context remains useful beyond this small scientific script.

Reproduction uses `terminal_modern_contract.py` with `--strategy requirements_first`, `--include-schema-only`, and optionally `--include-stamp-schema`. All official task files and raw outputs remain outside Git.

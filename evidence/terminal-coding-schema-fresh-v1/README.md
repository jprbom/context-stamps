# First unopened schema-context coding task

Prashant Jagtap · 2 October 2026 · negative native result

The [task and gate](protocol.md) were recorded before opening the official Terminal-Bench 2.1 `multi-source-data-merger` task at revision `7131e4375048a0e408a8fb404b5f499d726b695b`. An empty submission failed and the reference passed the isolated native verifier. The unchanged local Qwen2.5-Coder 7B received the same task and path contract in four arms: contract only, direct field schema, full source, and schema activated through exact 256-bit stamps. None passed. The frozen run used one generation per arm and no verifier-guided repair.

| Frozen arm | Native pass | Input tokens | Output tokens | Host view bytes |
| --- | ---: | ---: | ---: | ---: |
| Contract only | 0/1 | 704 | 1,013 | 36 |
| Direct schema | 0/1 | 827 | 808 | 507 |
| Full source | 0/1 | 974 | 784 | 715 |
| Stamp-bound schema | 0/1 | 827 | 1,008 | 507 |

The schema arms showed the same field names to the model and had identical prompt hashes. The stamp route also built three 32-byte handles (96 bytes total), with source bytes, revision, role, authorization and field schema held by the host. The single packet build was 20.3 ms; it is not a latency distribution or a full workflow measurement. The smaller view did not make this task pass, and the stamp supplied no model-visible information beyond the direct schema control.

After inspecting the failed run, three separate development probes were made. A generic execution-feedback loop allowed up to three generations per arm; all four still failed native grading. A generic record-fold coding hint also failed in all four arms. A single direct-schema probe with local Qwen3.5 4B produced an executable script, but its native output contained only two of four required users and no conflicts, so it failed. These probes are tuning data, not independent validation. A 30B coding model could not be loaded on the available 16 GB RTX laptop GPU and host memory; no score is assigned to that attempt. No paid frontier inference was used.

The first 7B scripts made incorrect assumptions about merged column names and failed during execution. Runtime feedback did not reliably correct those assumptions. A schema can prevent an absent-field guess on the earlier [inspected scientific task](../terminal-coding-schema-development-v1/README.md), but field names alone do not specify join semantics, precedence, null handling, or conflict reporting. This task requires all of those. The observed limitation is task competence and supplied semantics, not only retrieval precision. Neither a 32-byte handle nor a larger prompt can guarantee a correct transformation.

The [numeric record](results.json) preserves run order, pinned model digests, arm outcomes, token counts, attempt counts, code and prompt hashes, and hashes of the external run plans and summaries. Raw benchmark files, generated scripts, verifier logs and model outputs remain outside Git. `experiments/record_terminal_merger_schema.py` reconstructs this sanitized record from local runs. To reproduce, prepare pinned task images with `experiments/prepare_terminal_coding_images.py`, build the packet with `experiments/terminal_merger_packet.py`, then use `experiments/terminal_merger_schema.py` and the native verifier in the task image. Parquet schema parsing is opt-in for trusted input. This is one task, not an official aggregate Terminal-Bench score.

The candidate stays inactive. A next experiment needs a model that can solve this task family with direct authorized context, then a paired comparison of direct schema and stamp-bound schema on fresh tasks with full resolution, authorization, token and latency costs. The direct baseline must pass the same native checks. Code generation, exact context routing and quantized storage are separate capabilities; success in one cannot be used as evidence for another.

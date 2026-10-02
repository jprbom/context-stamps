# Public-input invariant feedback on an inspected coding task

Prashant Jagtap · 2 October 2026 · development evidence only

The first [predeclared multi-source merger test](../terminal-coding-schema-fresh-v1/README.md) failed in every frozen context arm. After inspecting that result, we added a bounded host-declared checker for one failure that native grading exposed: a generated program can exit successfully while omitting required records. The checker reads only the task's public declared inputs and the candidate's output in the isolated agent image. It compares the union of record IDs with the generated Parquet IDs, checks duplicates, required columns and whether the report's declared conflict count equals its list length. It does **not** verify source-priority values, conflict correctness, date semantics or the whole task. The native verifier remains the final outcome check; its tests and reference solution are absent from the agent image.

The [generic checker](../../experiments/public_record_invariants.py) and [local harness option](../../experiments/terminal_merger_schema.py) are optional. The original frozen prompt and result remain unchanged. Two synthetic checker tests cover a complete output, missing/extra/duplicate IDs, absent columns and a report count mismatch.

| Inspected-task development probe | Local model | What happened | Native result |
| --- | --- | --- | --- |
| Record-fold hint + runtime feedback, three attempts | Qwen3.5 4B | All generated programs exited nonzero before outputs; the public checker could not run | 0/1 |
| Original prompt + public invariant feedback | Qwen3.5 4B | First program exited zero but returned 2 of 4 expected IDs; checker reported two missing IDs. Second program crashed with a pandas indexing error. | **Not run:** stopped before third attempt and native grading for disk headroom |

The first probe used 8,339 input and 7,780 output tokens across three attempts. Its external plan SHA-256 is `ae5510148e877dd6741aff793193d7071ade35360b07892fef05395e18e2cf82`; summary SHA-256 is `4a323da599b50ff93b07ee4a11d29c31420513783c17e9665307785c18a2cab0`. The interrupted second probe's plan SHA-256 is `2f612cea961c49b69dd3d7072eb60d99c5375b2cf2bf071ec07b4d4babbc4f2e`. Its first two local execution records have SHA-256 values `c8362fdc1ed4306bb77af9ac230e221ad655dcfddec3db9dc094473e18388afc` and `ca8a68bd74d6cc8e580de04760b9821650ee71149c44e1316d2a9b76a165fa62`. Raw code, benchmark files, model responses and logs remain outside Git.

The interrupted probe had empty/reference native controls that failed/passed as expected, but **no final model native score**. The checker detected a failure the old execution-only loop missed; it has not yet improved task pass rate. This is post-inspection tuning, not fresh validation and not a stamp-specific effect. Both probes used direct schema; no paired stamp arm was justified while the direct baseline failed.

Host storage was a practical constraint. The pinned 2.96 GB TechQA download archive was removed only after matching its recorded SHA-256 and confirming the selected extracted data and license remained; `experiments/fetch_techqa.py` retains the source and checksum for recovery. Docker's writable disk image then grew rapidly across the isolated attempts. The second probe was stopped around 1 GB free. The harness now checks a configurable host free-space floor before starting each isolated run (`--min-free-gb`, default 1.5), so future runs fail before launching another sandbox when space is low.

The next competent-model gate still requires adequate disk and memory, a direct-context pass on new native tasks, and a paired direct/stamp comparison under the same public checker. Do not train on this inspected task or promote this checker as a proof of correctness.

# Frozen next-task choice and gate

Prashant Jagtap · 2 October 2026 · recorded before opening the task's source, tests, or reference solution

The next official Terminal-Bench 2.1 task is `multi-source-data-merger` at repository revision `7131e4375048a0e408a8fb404b5f499d726b695b`. It was chosen by its directory name because the development failure concerned source data shape. `modernize-scientific-stack` and `fix-code-vulnerability` are inspected development tasks and cannot serve as this test.

The candidate is a source-revision-checked, role-scoped, stamp-bound *minimal schema view*. The 256-bit stamp remains a host-held routing handle; the source path, CSV header, digest, policy and any full content remain outside its 32 bytes. Controls are task instruction plus exact file contract without source body, the same authorized schema view supplied directly, and the direct full-source packet. The same frozen local `qwen2.5-coder:7b` model, system instruction, generation limits, static path check and bounded repair allowance apply to every arm. No benchmark tests, reference patch or verifier feedback will be used to tune the candidate before all arms complete.

Before inference, inspect only the public task instruction and environment to map its required input/output files. If the task cannot be represented by the one-shot code artifact harness or has no supported schema type, record that as an ineligible task rather than selecting a replacement after looking at results. Keep task files, model output and native verifier logs outside Git. Run negative and reference grader controls. Record every arm, order, full input/output tokens, packet bytes, host schema/index build and activation time, native pass/fail, wall time and any path violation. A stamp-specific benefit requires a better outcome or cost than the **same schema view without stamps**, not merely a better outcome than a verbose full-source prompt. A single task cannot establish generalization; this test is only the first unopened task-family check.

No paid frontier calls and no training on this task are permitted in this protocol.

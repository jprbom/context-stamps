## Typed file tools and the schema-ordering regression

The next development experiment adds `read_file`, `write_file`, `shell` and `finish` actions. File operations run inside the isolated container, accept bounded literal UTF-8 data and stay under `/app`. Directory descriptors reject symlink traversal, reads reject FIFOs and oversized files, and writes replace a path atomically. Eleven unit tests pass on Windows Python 3.12/3.13 and Linux Python 3.12; twelve live file probes pass. These checks do not prove containment against every possible exploit.

The first union-schema implementation caused both models to finish immediately. Requiring a bounded submission before accepting `finish` changed that into repeated rejected completion, still without solving either task. The root-cause experiment compared identical copy requests with four response formats. No generated commands were executed in these canaries.

| Response format | Qwen 1.5B correct actions | Coder 7B correct actions |
|---|---:|---:|
| Recursively sorted union schema | 0/3 | 0/3 |
| Order-preserving union schema | 3/3 | 3/3 |
| Plain JSON mode | 3/3 | 3/3 |
| Sorted single-action schema | 2/3 | 3/3 |

Sorting the request recursively moved fields such as `command` before the `tool` discriminator. With the recorded Ollama version, models, prompts and greedy decoding, preserving discriminator order corrected the measured copy behavior. This observation does not establish a general rule for every server or model. A semantically equivalent JSON Schema can still produce a different constrained-generation process; hash and bind the exact wire representation when registering a learned policy.

The corrected `typed-files-v3` interface preserves schema insertion order and retains the artifact-presence gate. Its task results are:

| Model and task | Native task pass | Calls | Input / output tokens | Agent elapsed seconds |
|---|---:|---:|---:|---:|
| Qwen 1.5B — cancellation | No | 2 | 933 / 183 | 19.55 |
| Qwen 1.5B — logs | No | 1 | 575 / 1,024 | 16.87 |
| Coder 7B — cancellation | No | 12 | 15,315 / 1,128 | 46.82 |
| Coder 7B — logs | No | 9 | 17,907 / 1,924 | 52.25 |

The 1.5B cancellation code passes callables where awaitables are required; its log action reaches the output budget before execution. The 7B cancellation artifact fails on a missing `Callable` import; its log artifact has valid structure but incorrect counts. Artifact presence is not correctness. Both models pass 0/2 whole tasks, and the typed 7B result regresses from the legacy shell control's 1/2. The shell interface remains the default; these typed variants are experimental. Timing is descriptive, without randomized repetitions.

All failed and corrected runs are retained: 76 task calls across three interface versions, plus 24 schema-canary calls. These repeatedly inspected tasks remain evaluation-only development cases. No task material is used for training, and no context-runtime or learned-policy treatment is present. Sources are archived before changes; replay verifies prompt hashes, request bytes, token parity, native grader reports and file probes without executing model output.

```bash
python experiments/test_terminal_tools.py -v
python experiments/verify_typed_terminal.py
```

[Complete records and registrations](../evidence/terminal-typed-v1/manifest.json).

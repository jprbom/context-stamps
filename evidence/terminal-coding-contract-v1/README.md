# Host path contract: inspected coding-task diagnostic

Prashant Jagtap · 2 October 2026 · **candidate inactive**

This replay tests a narrow repair to the [earlier Terminal-Bench coding pilot](../terminal-coding-development-v1/README.md). The task `modernize-scientific-stack` had already been inspected. It is development data, not an independent benchmark or a source of training examples. The official Terminal-Bench 2.1 source is pinned to `7131e4375048a0e408a8fb404b5f499d726b695b`; task source, reference solution, generated code, verifier tests and raw model responses stay outside this repository.

The new host contract binds three explicit input paths to SHA-256 revisions and gives the model two exact output paths. Both arms see the same instruction, source bytes, contract and complete-prompt limit. The direct arm renders the three files plainly. The stamp arm resolves the already declared dependency closure and carries extra relation/revision metadata. Thus, any difference cannot be credited to one arm receiving more task files. The 32-byte stamps remain host routing handles; their payload does not contain the three files.

The first generation in **both** arms used relative paths for `config.ini` and `sample_data/climate_data.csv`. A bounded static diagnostic caught those literals. One fresh generation with the same generic exact-path reminder removed the flagged literals in both arms. Neither final program passed the native executable verifier. The contract repaired this path symptom, but did not make the model implement the requested mean-temperature behavior. The direct program still retained legacy visualization/anomaly work; the stamp program also retained the old computation. Static path checking is not proof that computed paths, execution or task logic are correct.

| Arm | First flagged literals | After one repair | Native pass | Total input tokens | Total output tokens |
| --- | ---: | ---: | ---: | ---: | ---: |
| Direct three files + contract | 2 | 0 | 0/1 | 3,906 | 962 |
| Stamp closure + contract | 2 | 0 | 0/1 | 4,706 | 789 |

The stamp arm used **800 more input tokens** across the two requests. Request wall times are recorded in [results.json](results.json), but one ordered pair on a warm local model is insufficient for a latency comparison. Native grader controls behaved as expected: empty submission failed, reference solution passed. The model was frozen `qwen2.5-coder:7b` (Q4_K_M), temperature 0, seed 7; no paid provider calls or benchmark-material training occurred. Both submissions ran in the same offline, non-root verifier profile. The exact external plan and summary file hashes are in the numeric record.

This result keeps the coding quality gate closed. A path contract is useful as a fail-fast interface, but source routing and 256-bit quantization still lack a verified task-outcome gain. A future quantized route should compare dense, SQ8 and stamp-to-SQ8 under the same host contract, source authorization and full cost accounting. It should open new task families only after development controls pass; this inspected task cannot serve as fresh confirmation.

Reproduce with the pinned official task checkout and local verifier prepared by the earlier pilot:

```powershell
$env:HARBOR_TELEMETRY = 'off'
python experiments/terminal_modern_contract.py --source PATH_TO_PINNED_SOURCE --packets PATH_TO_PRIVATE_PACKETS --verifier-image PATH_TO_VERIFIER_ID --token-python PATH_TO_TOKENIZER_PYTHON --tokenizer PATH_TO_TOKENIZER_JSON --out PATH_OUTSIDE_GIT
```

Copyright (c) 2026 Prashant Jagtap. Repository code is MIT licensed; upstream benchmark and model terms remain with their owners.

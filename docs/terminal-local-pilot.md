# Local coding-agent pilot

By Prashant Jagtap

The current small-model baseline is not ready for the intended domain workflow. After a bounded interface correction, local Qwen2.5 1.5B Q4_K_M passed **0 of 2 selected Terminal-Bench 2.1 tasks**. Both reference solutions passed their native assertions, and empty submissions failed. This establishes an executable task baseline and identifies failures; it does not establish a Context Stamps improvement.

## Observations

| Run | Task | Outcome | Model calls | Input / output tokens | Agent elapsed seconds |
|---|---|---|---:|---:|---:|
| Initial | Async task cancellation | Invalid action format; no submission | 2 | 611 / 48 | 17.43 |
| Corrected interface | Async task cancellation | Agent declared completion without creating the file | 3 | 1,109 / 66 | 17.00 |
| Corrected interface | Dated-log aggregation | Repetitive generation exhausted the 1,024-token output budget | 1 | 524 / 1,024 | 16.26 |

The initial run stopped after its action-format error. The corrected run allows at most two format-repair messages, never executes a rejected action, and still collects partial submissions after an error. The initial failure is retained. Both tasks are now development/regression cases; they cannot later be called untouched final evaluation. No training or fine-tuning used their prompts, outputs, reference solutions or graders.

All six model responses matched the independently counted complete input prompt. Actual server input/output tokens, commands, outputs, statuses, model identities, device telemetry and native test reports are retained under [terminal-pilot-v1](../evidence/terminal-pilot-v1). No paid model or judge was called. Host peak RAM and energy are unknown. Agent elapsed time includes setup, token counting, model calls, tool execution and teardown, but excludes the separate grading phase. These one-task timings are not latency quantiles or a controlled speedup comparison.

## Protocol and security boundary

The task collection is pinned to `harbor-framework/terminal-bench-2-1` revision `7131e4375048a0e408a8fb404b5f499d726b695b`. Selection preceded generation and used task type and local resource fit: `cancel-async-tasks` and `log-summary-date-ranges`. The model uses a frozen raw ChatML prompt, temperature 0, seed 7, at most 12 calls, a 16,384-token context and a 15,000-token complete-input ceiling. A pinned Qwen tokenizer runs in the existing permitted local environment. Windows Application Control was not disabled or bypassed; the blocked Inspect/pandas environment is unused.

The host loop uses Harbor 0.23.0's Docker environment API. Model-written commands execute only inside the task container. The model server runs on fixed host loopback and rejects redirects or cloud-backed models. A task container receives no provider keys, host mounts, Docker socket, grader files or solution. Only a regular, bounded `run.py` or `summary.csv` artifact is transferred to a newly created verifier container after the agent container has been destroyed.

The tested agent profile uses UID 1000, a read-only root filesystem, dropped capabilities, no new privileges, one CPU, 2 GiB RAM, 64 processes and bounded temporary filesystems. Explicit Docker `network_mode: none` leaves only loopback. Native Harbor controlled-egress rules permit DNS/ICMP, so this additional restriction is necessary for this pilot's offline boundary. Ordinary commands, excessive output and a timed-out command with a detached child were tested; task-owned containers were absent after teardown. These checks do not establish protection against all container/kernel exploits or all forms of grader manipulation.

The original verifier pin, pytest 8.4.1, has a known temporary-directory issue. The audit returned two entries for the **same** advisory, not two distinct vulnerabilities. The pilot uses pytest 9.0.3 and pytest-json-ctrf 0.3.5 with all seven dependencies pinned and hashed. The initial patched audit omitted `packaging` from its output. A separate explicit audit of all seven exact pins found no known findings or skipped packages; both reports are retained. This is a dated package audit, not a Docker base-image vulnerability scan. [Advisory and patched version](https://github.com/advisories/GHSA-6w46-j5rx-g56g)

This is a **modified development pilot**, not the full native Harbor job/leaderboard protocol. Changes include the custom host agent loop, a patched pytest, preinstalled offline verifier dependencies, separate verifier containers, restrictive filesystem/network settings and a shorter model/tool budget. Native task instructions, log generation and test assertions are unchanged. The reference and empty-submission controls pass their expected outcomes, but that is not a proof of equivalence for every possible submission.

## Reproduction and next experiment

Install the pinned [Harbor environment](evaluation-programme.md), prepare the licensed task files at the recorded revision and build isolated verifier images from the pinned Python digest and [hashed requirements](../evidence/terminal-pilot-v1/verifier-requirements.txt). Verifier images contain only their own task's tests; agent images contain none. Keep raw task files and downloaded models outside the repository. The [complete driver](../experiments/terminal_pilot.py) exposes `prepare`, `qualify-graders` and `run`, each with a fresh output directory for a changed protocol. `--help` lists the explicit source, tokenizer and local environment paths. The current driver also accepts `--model qwen2.5-coder:7b` with its matching tokenizer for a stronger local reference; no result for that model is included here.

To replay the published records without Docker, network access or a model:

```bash
python experiments/test_terminal_pilot.py -v
python experiments/verify_terminal_pilot.py
```

For a fresh local run, start in a reviewed research checkout. Use the separately
installed Harbor environment from the evaluation runbook and a tokenizer-only
environment with `tokenizers==0.23.2`. Docker must target the local Linux engine.
The following PowerShell example keeps data and build outputs outside Git:

```powershell
$env:HARBOR_TELEMETRY = 'off'
$harborPython = '../harbor-eval-venv/Scripts/python.exe'
$tokenPython = '../general-learning-cuda/Scripts/python.exe'

& $harborPython experiments/prepare_terminal_assets.py fetch --out ../terminal-source-new
& $harborPython experiments/prepare_terminal_assets.py build --source ../terminal-source-new --out ../terminal-images-new
& $harborPython experiments/harbor_sandbox.py --out ../terminal-sandbox-new

$runArgs = @('--source', '../terminal-source-new', '--token-python', $tokenPython,
             '--tokenizer', '../qwen25-15b-tokenizer/tokenizer.json',
             '--out', '../terminal-run-new')
& $harborPython experiments/terminal_pilot.py prepare @runArgs --model qwen2.5:1.5b --images ../terminal-images-new/images.json
& $harborPython experiments/terminal_pilot.py qualify-graders @runArgs
& $harborPython experiments/terminal_pilot.py run @runArgs
```

Install the local Ollama model separately. Obtain the tokenizer from
`Qwen/Qwen2.5-1.5B-Instruct`, revision
`989aa7980e4cf806f80c7fef2b1adb7bc71aa306`; the recorded tokenizer hash is in
each plan. Inspect every command's exit status and stop on a failed preparation
or qualification. Use fresh output names after changing a protocol. The driver
binds the actual local model digest, tokenizer, source and verifier images before
generation. A mutable Ollama tag may resolve differently later; compare digests
with the published plan before calling a run a reproduction. Docker image IDs
can differ across rebuilds; native positive/negative controls must pass locally.
The published source archives preserve the earlier interface and supervisor;
the commands above exercise the current hardened implementation.

Sixteen boundary tests cover action ambiguity, protocol delimiter injection, command/output limits, cancellation, changed container isolation, bounded format repair, partial-artifact grading, trusted-helper import isolation and unknown usage after provider failure. The sandbox probe and native grader qualification are separate live checks.

After these model runs, a review found that Python helpers could import an agent-created module from `/app`. The current supervisor and artifact helpers use Python's isolated mode, and an additional live probe creates hostile module names before verifying supervisor operation. The earlier source is archived with the failed baseline records; their agents created no submission. This is a harness correction, not a model-quality improvement.

The next comparison needs a stronger local coding reference before model/runtime effects can be separated. For the smaller model, investigate typed file-edit/tool interfaces and training on independently verified **non-benchmark** action examples. Compare that with ordinary prompt/scaffold corrections first. The context runtime and local statistical policy then need their own paired treatment on fresh task families. Extra context, more compact stamps and a successful statistical gate cannot supply missing code-generation competence. This is why local adaptation must test absolute task quality as well as savings and retention.

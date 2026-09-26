# Evaluation programme

By Prashant Jagtap

The question is whether Context Stamps improves a particular model's task outcome or reduces its total resource use while preserving required quality. A model leaderboard alone cannot answer that question. Compare the same model, agent scaffold, tools and task budget with and without the runtime; report comparisons between different models separately.

**Status, 26 September 2026:** the independent [LongBench v2 reader pilot](longbench-local-pilot.md) completed 30 local calls on ten short-context tasks: full context 4/10, BM25 4/10, no context 3/10. This establishes baseline plumbing, not a runtime benefit or frontier ranking. Seventeen evaluation-boundary tests pass. Harbor 0.23.0 starts locally with Docker 29.7.2; no coding-agent trial has run, and sandbox/task qualification remains. The separate native Inspect ARC preparation remains blocked before generation by Windows Application Control on a pandas DLL. The 304-test acquisition revision passed all six GitHub CI jobs. Engineering checks and model accuracy remain separate evidence.

## Harness selection

| Tool or collection | Intended role | Decision and limits |
|---|---|---|
| Inspect AI + Inspect Evals | Native task definitions, model adapters, scoring and per-sample logs | Selected for the first local model track. Installed versions are pinned; execution is not yet qualified on this host. |
| Harbor | Isolated agent tasks, verifier execution and trajectories | Version 0.23.0 installed; CLI version/help and local Docker preflight pass. Adapter and sandbox qualification remain. |
| Cline Bench | Engineering tasks derived from coding sessions; Cline as a scaffold control | Harbor-compatible. Inspected revision has no root license file or declared repository license; task rights need resolution before inclusion. No tasks copied or executed. |
| Terminal-Bench 2.1 | Broader terminal and coding-agent tasks | Preferred initial Harbor collection after task/resource/license review. Keep 2.0 and 2.1 results separate. |
| EleutherAI LM Evaluation Harness | Conventional language-model capability controls | Useful secondary cross-check when its task protocol fits. It does not by itself test persistent enterprise context. |
| Benchmark authors' native runners | Retrieval, long context and long-term memory | Preserve native scoring and version-specific task definitions; mark every modified protocol or subset. |

Sources: [Inspect providers](https://inspect.aisi.org.uk/providers.html), [Inspect Evals](https://github.com/UKGovernmentBEIS/inspect_evals), [Harbor](https://docs.harborframework.com/), [Cline Bench](https://github.com/cline/cline-bench), [Terminal-Bench 2.1](https://github.com/harbor-framework/terminal-bench-2-1), [LM Evaluation Harness](https://github.com/EleutherAI/lm-evaluation-harness). Framework code licenses do not automatically license every dataset, model or task dependency.

The Terminal-Bench authors revised tasks to address external dependency drift, resource mismatches and specification problems. Freeze the collection revision and container digests, validate the task environment, and report infrastructure failures separately from incorrect model answers. Do not combine scores from different benchmark versions. [Revision explanation](https://www.tbench.ai/news/terminal-bench-2-1)

Published Cline or leaderboard results can help choose controls. They are external results, not measurements of this repository, and cannot establish a runtime improvement. Our own paired runs must retain their task identities, configurations and failures. No automatic leaderboard upload is planned.

For the Harbor implementation, use an external `BaseAgent` when the runtime and
local model connection need to remain on the host. The agent sends task commands
only through Harbor's `BaseEnvironment`; it must not execute model-written shell
commands on the host. Keep identical model, tool, turn and time budgets for the
baseline and runtime treatment. Harbor also supports installed agents, including
the Cline CLI control. [Custom agent API](https://docs.harborframework.com/core-concepts/agents/custom-agents)

Qualify the local Docker boundary before running coding tasks: pin images and
dependencies, separate verifier inputs, use task-owned writable volumes, exclude
host credentials, test resource limits and cancellation, and validate denied
network access. Harbor distinguishes setup, agent and verifier phases; a policy
applied only during the agent phase does not restrict setup. Docker network
policies require its egress-control support. These checks are pending, not
capabilities already proved by this repository.
[Network policy semantics](https://docs.harborframework.com/core-concepts/tasks/network-policies)

Set `$env:HARBOR_TELEMETRY='off'` before invoking Harbor for these local runs.
Its default usage telemetry includes job-level metadata and measurements; no
upload is needed to perform a local comparison. Keep the raw run artifacts in
the task workspace and publish only reviewed evidence.
[Telemetry controls](https://docs.harborframework.com/telemetry/telemetry)

The tested local CLI environment is recorded in
[harbor-preflight.json](../evidence/enterprise-evaluation-v1/harbor-preflight.json).
To reproduce that installation separately from model-serving dependencies:

```powershell
uv venv ../harbor-eval-venv --python 3.12
uv pip install --python ../harbor-eval-venv/Scripts/python.exe -r experiments/harbor-requirements.txt
$env:HARBOR_TELEMETRY='off'
& ../harbor-eval-venv/Scripts/harbor.exe --version
& ../harbor-eval-venv/Scripts/harbor.exe run --help
```

These are installation checks, not an agent trial or container isolation test.

## Evaluation tracks

| Track | Collections / scenarios | Primary question |
|---|---|---|
| Retrieval | Existing BEIR collections as regression; new untouched licensed scopes | Does candidate selection preserve relevance at measured index/query cost? |
| Long context | Version-pinned RULER and LongBench v2 | What changes with context length, distractors and distributed evidence? |
| Persistent memory | LongMemEval, LoCoMo, BEAM after individual rights review | Can the runtime handle temporal updates, contradictions, missing evidence and long histories? |
| Coding and tools | Terminal-Bench 2.1, SWE-bench Verified; Cline Bench conditional on rights | Does context management improve verified task completion and recovery at equal tool/time budgets? |
| General controls | ARC, IFEval, MMLU-Pro and mathematics tasks where applicable | Does the adapter preserve ordinary capability, formatting and instruction compliance? |
| Enterprise reliability | Revocation, stale policy, delayed correction, conflict, injected instructions, missing dependencies, restart and partial tool effects | Does the system reject unsafe or unsupported decisions and retain an inspectable failure? |
| Multimodal | Separate licensed image/OCR, audio and video task protocols | Do canonical adapters improve end-task results? Text success is not multimodal evidence. |

This is a coverage plan, not a statement that all these suites are installed, licensed for every use, or completed. LongMemEval's oracle evidence is a labelled upper-bound control, never a normal retrieval input. Native model-judge tasks need a separately recorded judge; local surrogate scoring must be labelled and cannot be presented as the native score.

## Model tiers and access

1. **Existing local controls:** use exact digests for Qwen2.5 1.5B and the existing Cortex checkpoint after its provenance audit. Qwen2.5 is a compatibility/control model, not a claim about today's strongest open weights. Its model card declares Apache-2.0. [Model card](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct)
2. **RTX candidates:** choose a current small model, a quantized general reader and a VLM after reviewing official cards and profiling actual residency. Qwen3.5 4B is one candidate, not a selected winner. A locally installed 17–18 GB weight file is not evidence that the complete workload fits the 16 GB GPU. Record CPU offload explicitly. [Candidate card](https://huggingface.co/Qwen/Qwen3.5-4B)
3. **Large open-weight and frontier controls:** prepare the same tasks/adapters for exact provider model revisions. Selection follows official availability and independent task-specific results at registration time. No paid inference, hosted GPU job or remote judge is authorized in this phase. Provider API charges are separate from development-assistant credits.

Use one resident GPU workload at a time. Keep dataset checks, state, audit and result aggregation on CPU. Increase batching only after a quality/latency/parity sweep. Preserve an unquantized or higher-precision control when claiming quantization gains. Do not keep the GPU busy with unrelated load to inflate utilization.

## Paired protocol and promotion rules

Before seeing final outputs, register model/tokenizer/quantization revisions; model template and reasoning settings; task IDs, dataset hashes and licenses; context/output limits; seeds/trials; adapter source hashes; hardware; concurrency; and acceptance thresholds. There are three distinct stages: plumbing smoke, development/calibration, and untouched final evaluation. Smoke and development tasks cannot be renamed as untouched final tests.

Use the same original evidence and permissions in each applicable treatment:

- model with full authorized context, where the complete input fits;
- dense and qualified hybrid retrieval;
- exact metadata/dependency control when those labels are legitimately available;
- runtime compilation without reuse or prefetch;
- runtime plus exact reuse;
- runtime plus prefetch or a learned selector only as separate ablations.

Include zero-repeat, repeated and changing-state streams. Randomize paired run order, isolate sessions, and report cold initialization separately from warm request latency. Keep answer keys, oracle solutions and grader-only metadata outside model/tool-visible inputs. Never silently truncate a long-context baseline. If it cannot fit, report unsupported length, not a fabricated loss or equal-budget comparison.

Select development thresholds once. The default research promotion gate is a task-success noninferiority margin of at most one percentage point, evaluated with a paired 95% interval clustered by original task, plus a positive reduction in the preregistered cost or latency metric. High-risk tasks may require a stricter margin. An inconclusive interval is not a pass. Report raw paired counts and every dataset separately; do not hide a regression behind a pooled average. No new unauthorized-source release or verifier-accepted stale decision is permissible in the security regression suite. Passing that suite is not proof of zero production risk.

For repeated stochastic trials, cluster intervals by task rather than treating retries as independent examples. Record all attempts and their compute. Keep failed tuning candidates. After inspecting final-test failures, treat that set as development for further changes and qualify the revised method on a newly frozen scope. Fix engineering defects independently of benchmark answer content.

## Measurements

| Dimension | Record |
|---|---|
| Quality | Native score, task success, evidence correctness, missing-fact/contradiction errors, malformed outputs |
| Decision risk | False acceptance, abstention, risk versus coverage; Brier/ECE only for genuine calibrated probabilities |
| Tokens and calls | Actual input/output/reasoning/cache token counts where exposed, auxiliary model/judge/tool calls, retries and fallback work |
| Cost | Provider charges at dated pricing, cache charges, local energy only if measured; never report unknown cost as zero |
| Latency | Whole-request p50/p95/p99, TTFT when streamed, model load, context preparation, inference, verification and wasted prefetch |
| Resources | Sampled GPU utilization/VRAM and host RAM, batch/concurrency, throughput, queue delay and OOM/failure rates |
| Workflow | Verified completed tasks, steps, redundant calls, recovery, invalidated/reused results and stale releases |

Record unsupported metrics as null. A short pilot cannot justify a stable p99 or a scalability claim. Report maximum tested workload and the saturation curve. Data preparation and indexing costs belong in the report even when amortized separately from request latency.

## Local pilot

The first runner deliberately supports only Inspect's native ARC-Challenge multiple-choice solver and scorer, at most 20 samples, and installed local GGUF weights through loopback Ollama. It has no remote-model option. Model digests, prompt templates, package versions, task source hashes and selected sample hashes are checked before execution. Each run directory is consumed once so a failed attempt cannot be overwritten by a successful retry. This runner is preparation code until a full local smoke succeeds.

From the research checkout in PowerShell:

```powershell
uv venv ../enterprise-eval-venv --python 3.12
uv pip install --python ../enterprise-eval-venv/Scripts/python.exe -r experiments/eval-requirements.txt
& ../enterprise-eval-venv/Scripts/python.exe experiments/local_eval.py prepare --output ../enterprise-eval-runs/arc-qwen-pilot --model qwen2.5:1.5b --limit 10
# Run only after preparation and host security checks succeed:
& ../enterprise-eval-venv/Scripts/python.exe experiments/local_eval.py run --output ../enterprise-eval-runs/arc-qwen-pilot
```

The observed Windows failure occurred while importing pandas 3.0.6 (`ops.cp312-win_amd64.pyd`): Application Control blocked the file. Do not disable the control or relocate the blocked binary to evade it. Use the host's normal security/admin review. The failure occurred before dataset preparation completed and before any generation. [Recorded setup status](../evidence/enterprise-evaluation-v1/preflight.json)

ARC's pinned dataset card declares CC-BY-SA-4.0. Raw Inspect logs can contain its questions and answers; keep them outside the repository until a separate export/licensing review. A source-code MIT license does not relicense those records. [Dataset card](https://huggingface.co/datasets/allenai/ai2_arc)

## Gap review and next implementation

The immediate gaps are a qualified evaluation environment, native context/memory adapters, reliable whole-request instrumentation, and a sandboxed coding-agent integration. A small ARC run can validate the plumbing but cannot demonstrate a context benefit. The learned selector still needs generalization evidence, and the measured simple prefetch policy remains slower than ordinary working-set reuse in the storage fixture.

After the local smoke passes, register the first matched context pilot, audit errors by retrieval/compilation/model/verification stage, and implement a fix only where the trace supports it. Then rerun the affected development controls and evaluate the frozen candidate on untouched tasks. Keep the existing qualified retrieval routes available throughout. Frontier comparisons remain prepared but unexecuted until provider access and an explicit spending limit are supplied.

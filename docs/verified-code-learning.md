# Verified local code adaptation

By Prashant Jagtap

Context Stamps aims to let a local domain system improve through verified
experience. The runtime can update evidence and context policies, and capable
devices can train a small model adapter. These are separate changes with separate
versions and evaluation requirements. An edge device without a suitable training
backend can still learn through memory and policy updates.

The current public-code experiment establishes the next training and evaluation
path. It does **not** yet establish an improvement from Context Stamps, domain
AGI, or a candidate suitable for automatic deployment. The earlier
[authored-state adapter](local-adapter-rtx.md) remains inactive.

The [public-code candidate](../evidence/mbpp-code-adapter-v1/MODEL_CARD.md) has
completed one local fit: 368 examples, 184 optimizer steps, 65.42 seconds and
7.34 GiB peak Torch CUDA allocation. Its 1,089,536 trainable parameters produce
a 4.17 MiB adapter; the base fingerprint is unchanged. These are training
measurements. Model-quality comparisons are separate, and activation remains
disabled.

## Learning contract

1. Record the task, authorized context, model/runtime versions and an independent
   outcome. A model saying it succeeded is not a verified outcome.
2. Admit only examples with an appropriate verifier and permitted data use.
   Preserve rejected examples and the reason for rejection.
3. Fit a candidate using training data only. Keep an immutable baseline.
4. Compare the candidate and baseline on separate tasks using the same precision,
   prompt, tools, token limits and hardware. Measure regressions as well as gains.
5. Check retention, memory, latency and failure handling. Activate only through
   the local registry after the complete domain-specific gate passes.
6. Monitor fresh outcomes after activation; retain a known-good version for
   rollback. Automatic model deployment and drift-driven rollback are not
   qualified by this experiment.

The [existing local registry](local-domain-learning.md) implements evaluation
registration and reversible policy activation. Its published activation example
uses simulated tasks and costs. These code experiments do not automatically
connect an adapter to that registry or change a serving model.

## Public training data

The source is [MBPP](https://github.com/google-research/google-research/tree/f46ca8374b4cddef97ca4208ad986049d74d296a/mbpp)
by Austin et al. (2021), using the published training IDs 601–974. IDs 1–600
remain reserved for prompting, validation or testing. The source revision,
file hashes, dataset card and license evidence are retained in
[the source manifest](../evidence/mbpp-training-v1/source.json).

Each of the 374 training references is checked against its three published
assertions in a fresh offline container. An empty-program control must fail.
Exact normalized-description or full-AST overlaps with reserved MBPP records
are excluded. A second audit compares training code with HumanEval+ references
after removing docstrings. This is a conservative exact-match filter, not
semantic decontamination or evidence that a pretrained base never saw a task.

Passing three assertions is an admission filter. It does not prove general
correctness or resistance to deliberate grader manipulation. The manually
sanitized MBPP variant is downloaded and pinned but is not silently substituted
for the published training split.

## Train on a local RTX

Use the pinned CUDA environment and base download in the
[RTX setup guide](local-adapter-rtx.md). Put data and new run directories outside
the repository. A failed or partial run is retained; use a new directory for a
new attempt. The Harbor environment is separate from the CUDA environment and
needs Docker Desktop with Linux containers.

```powershell
# CPU environment: download the exact source; no code from the data runs here.
python -c "import sys; from pathlib import Path; sys.path.insert(0,'experiments'); from mbpp_training_data import fetch; fetch(Path('../mbpp-source'))"

# Harbor environment: execute reference programs inside isolated containers.
$env:HARBOR_TELEMETRY = "off"
python experiments/qualify_mbpp_training.py --source ../mbpp-source --out ../mbpp-qualified

# CPU environment: download pinned benchmark sources and build the grader.
python experiments/code_benchmark_sources.py fetch --out ../evalplus-source
python experiments/code_benchmark_sources.py build --source ../evalplus-source --out ../evalplus-build

# CUDA environment: train a candidate; serving remains unchanged.
python experiments/mbpp_adapter.py --source ../mbpp-source --qualification ../mbpp-qualified --heldout-source ../evalplus-source/HumanEvalPlus.jsonl.gz --base ../qwen-base --base-manifest evidence/local-adapter-v1/base.json --out ../code-adapter-run
```

The fixed recipe starts from Qwen2.5-1.5B-Instruct, not the earlier fixture
adapter. Rank-eight query/value updates use BF16, two epochs, batch size four
and a learning rate of 0.0001. The base remains frozen and is fingerprinted before
and after training. Training covers the assistant response, including its end
marker; prompt and padding positions are masked. Overlong examples are refused
rather than truncated. There is no test-directed checkpoint search.

This is ordinary supervised adaptation used as a control. A future claim about
Context Stamps must also compare the same model with and without its verified
memory, context compiler and learning policy. A generic fine-tuning gain alone
would not establish that contribution.

## Evaluate separately

The pinned HumanEval+ v0.1.10 release contains 164 tasks, 1,570 base inputs and
122,683 additional inputs. The minimal image keeps the native EvalPlus grading
functions and special oracles unchanged. It installs only pinned NumPy/psutil
wheels from a verified local build context; no provider SDK or model download
is needed for grading. The image runs non-root, offline, with a read-only root,
no host mounts or GPU, and bounded CPU, memory, processes, output and time.

```powershell
# Harbor environment: first qualify controls, then audit the full reference set.
$image = (Get-Content ../evalplus-build/image-id.txt -Raw).Trim()
$env:HARBOR_TELEMETRY = "off"
python experiments/humaneval_local.py --image $image --out ../grader-smoke --smoke
python experiments/humaneval_local.py --image $image --out ../reference-audit

# CUDA environment: inspect a four-task pipeline smoke before the full run.
python experiments/code_adapter_generate.py --dataset ../evalplus-source/HumanEvalPlus.jsonl.gz --base ../qwen-base --run ../code-adapter-run --out ../generation-smoke --smoke

# Harbor environment: grade retained outputs in fresh per-program containers.
python experiments/grade_code_generation.py --image $image --run ../generation-smoke --out ../grading-smoke
```

For the full comparison, repeat generation without `--smoke`, using a fresh
output directory, and grade that run. The same BF16 model executes each prompt
with the adapter enabled and disabled, with paired order randomized by a fixed
seed. Only public prompts reach the model. Hidden inputs and reference programs
are not training or generation context. Generation is greedy with a fixed
1,024-token ceiling; malformed or truncated outputs count as failures.

This is a custom paired generation protocol using a native benchmark grader,
not a reproduction of a leaderboard's prompting and inference stack. Report
all 164 tasks, both base and extended-test results, gains, regressions, errors
and generation limits. Batch-generation timings are not per-request latency,
time to first token, energy measurements or edge-device results.

## A numerical limitation found before model evaluation

The first control run failed at file transfer: Docker refused copying a request
into the read-only container. The revised runner writes bounded data through
the checked isolated guest interpreter without relaxing the container profile.
Both attempts remain evidence.

HumanEval/32's published Newton solver passes its base inputs but fails an
extended polynomial-root input. An independent bracket-and-bisection diagnostic
returned the same root, approximately 17.3124550475086. At that root and 32
adjacent floating-point values in each direction, the smallest observed absolute
residual was approximately 0.002315, above the official tolerance of 0.0001.
This is a local numerical diagnostic, not a proof that no acceptable root exists.
The authored diagnostic is not a model output or training example.

Keep the official oracle and denominator unchanged when reporting model scores.
Report reference failures and any sensitivity analysis separately. Do not fix a
benchmark by silently deleting a difficult task or loosening its tolerance.

## Attribution and limits

MBPP records retain [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/),
as identified by the pinned official dataset card. Attribute Austin et al.,
*Program Synthesis with Large Language Models* (2021), link the source and
identify filtering or formatting changes. The upstream repository's Apache
code license does not replace the dataset license.

EvalPlus is by Liu, Xia, Wang and Zhang, *Is Your Code Generated by ChatGPT Really
Correct? Rigorous Evaluation of Large Language Models for Code Generation*
(NeurIPS 2023). Its Apache-2.0 license and original HumanEval MIT notices are
retained. The Qwen base remains Apache-2.0. Context Stamps code is MIT licensed,
copyright Prashant Jagtap. Upstream attribution is not an endorsement.

No paid provider calls, user documents, proprietary project code or production
experience enter these public experiments. Before adapting on actual local
work, define permitted data, retention, verifier trust and evaluation splits.
Do not publish private experience or adapters trained on it by default.

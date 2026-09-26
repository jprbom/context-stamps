# Local long-context reader controls

By Prashant Jagtap

This experiment establishes a reader baseline for later Context Stamps work.
It compares the same installed Qwen2.5 1.5B Q4_K_M model with full source context,
BM25-selected chunks and no source context. None of these three controls is a
Context Stamps compiler, trained sufficiency policy or runtime-assisted treatment.

The existing acquisition learner predicted whether a prefix retained the judged
positives in a retrieval pool. It failed its calibration gate and did not establish
whether a reader could answer a question. Here the outcome is an actual model
answer scored against the public benchmark's multiple-choice key. A correct
answer can still reflect guessing or prior knowledge, so it is not a sufficiency
certificate. The no-context control helps expose that ambiguity.

## Observed results, 26 September 2026

All 30 calls completed on the RTX 5080 Laptop GPU. Every planned input-token
count matched the server count exactly; no malformed answers or request errors
occurred. All ten selected tasks are in the dataset's **short** length category,
despite spanning all six domains. Medium and long categories were not tested.

| Control | Correct / tasks | Total input tokens | Total output tokens | Request median | Request p95 |
|---|---:|---:|---:|---:|---:|
| Full context | 4 / 10 | 191,647 | 94 | 3.273 s | 4.548 s |
| BM25, 4,096-token budget | 4 / 10 | 39,121 | 94 | 1.396 s | 13.422 s |
| No source context | 3 / 10 | 2,108 | 106 | 0.900 s | 1.143 s |

These are measured controls, **not a Context Stamps improvement**. BM25 used
79.6% fewer input tokens and matched full-context correctness on these ten
tasks: four jointly correct, six jointly wrong. That tiny sample cannot establish
quality noninferiority. Full context was correct on two tasks where no context
failed; no context was correct on one task where full context failed. Correctness
therefore cannot be treated as a direct label for evidence sufficiency.

The first call happened to be BM25 and included **11.949 seconds of model load**.
Its 13.422-second request is retained in the table and determines that arm's p95.
With ten observations, p95 is the largest observation, not a reliable production
tail estimate. Total request times were 33.197 s full, 26.244 s BM25 and 9.323 s
no-context; the complete execution loop took 69.561 s. Those totals include the
unbalanced cold load and do not support a controlled speedup claim. Context
preparation and model/identity calls are included per request; the corpus length
audit and downloads are separately recorded setup costs.

Across 61 device telemetry samples, peak utilization was 100%, peak device
memory was 2,230 MiB and peak temperature was 64 C. Ollama reported the resident
model fully in VRAM with a 32,768-token context capacity. These are one-workload
observations, not a saturation or scalability experiment. Paid inference calls:
zero. Local electricity cost was not measured.

Evidence: [frozen plan](../evidence/longbench-v2-pilot/run-v2/plan.json),
[request summary](../evidence/longbench-v2-pilot/run-v2/summary.json),
[paired and stratified report](../evidence/longbench-v2-pilot/run-v2/analysis.json).
Run `python experiments/verify_longbench.py` to replay the accounting without
model weights, network access or the raw dataset. Seventeen evaluation-boundary
tests pass (ten LongBench, seven existing local-provider checks).

## Protocol

The dataset is LongBench v2, pinned to revision
`2b48e494f2c7a2f0af81aae178e05c7e1dde0fe9`. The original collection contains 503
tasks spanning six domains. All 503 complete full-context prompt lengths are
measured with the pinned Qwen tokenizer. A task is eligible only when that prompt
has at most 28,000 tokens; the server context capacity is 32,768, with 128 output
tokens. No source is silently shortened to make the full-context baseline fit.

Select at most two eligible tasks in each domain using a frozen hash of a seed
string and the task ID. This excludes unsupported lengths before any generated
answer is observed. It is a small **development subset**, not a representative
full benchmark score, a leaderboard submission or an untouched final evaluation.
Downloaded benchmark keys are used only by the scorer; they cannot enter the
model input or retrieval query. Previous exposure during third-party model
pretraining is unknown.

The first length-only registration allowed 12,000 tokens and admitted just seven
tasks from one domain. It was superseded before any generation. The revised cap
admits 97 of 503 tasks across all six domains; 10 tasks are selected (one each
where a domain has only one eligible task, two otherwise). The original 562.17-second
CPU length audit is retained and reused by exact hash; revised preparation takes
6.25 seconds. No benchmark answers or generated outcomes informed that change.

The three upstream non-CoT prompts are retained byte for byte. The adapter uses
the upstream answer parser, including case sensitivity and malformed-output
failure. Temperature is 0.1, seed 7, and each task/treatment runs once without
retries. Treatment order is deterministically shuffled within each task. The
adapter renders one user message in Qwen ChatML and uses Ollama raw generation;
the stored model system prompt is not added. It requires exact agreement between
the planned complete-input token count and the server's reported input count.
Generation stops on a mismatch, with the failed attempt preserved.

BM25 uses Unicode word tokens, k1=1.5 and b=0.75. It ranks 512-token source chunks
with 64 tokens of overlap using the question and all four choices. Selected
chunks are restored to source order. The entire rendered input, including
question, options, chunk labels and ChatML, must fit 4,096 tokens. This is our
explicit retrieval control, not a claim to reproduce an unspecified upstream
retrieval implementation.

The upstream `result.py` repeats the `short` condition in its medium-length
branch. This runner groups by the dataset's actual labels instead. Per-item
native correctness is unchanged. Placeholder replacement is performed once so
placeholder-like strings in source text remain literal. These adapter differences
are recorded in the frozen plan.

## Reproduce on the local RTX

Use a permitted Python environment with `tokenizers` and the Hugging Face CLI.
No Torch import, pandas, paid provider or remote judge is required by this runner.
The independent Inspect ARC environment remains blocked by Windows Application
Control pending review; this experiment does not change that environment or the
security policy.

```powershell
hf download zai-org/LongBench-v2 data.json README.md --type dataset --revision 2b48e494f2c7a2f0af81aae178e05c7e1dde0fe9 --local-dir ../longbench-v2-data
hf download Qwen/Qwen2.5-1.5B-Instruct tokenizer.json tokenizer_config.json config.json LICENSE README.md --revision 989aa7980e4cf806f80c7fef2b1adb7bc71aa306 --local-dir ../qwen25-15b-tokenizer
python experiments/test_longbench_eval.py -v
$env:RAYON_NUM_THREADS='2'
python experiments/longbench_eval.py prepare --data ../longbench-v2-data --tokenizer ../qwen25-15b-tokenizer --output ../my-longbench-pilot
python experiments/longbench_eval.py run --data ../longbench-v2-data --tokenizer ../qwen25-15b-tokenizer --output ../my-longbench-pilot
```

Ollama must already serve the exact local `qwen2.5:1.5b` digest recorded in the
runner. There is no automatic model pull or selectable remote endpoint. The
model, server, source files, tokenizer implementation, dataset content and prompt
hashes are checked before execution. A directory is consumed once, including a
failed run. Create a new directory for any intentional retry and retain the old
one. Code changes require a newly registered plan.

CPU performs dataset auditing, selection, scoring and reporting; the GPU runs one
reader workload. The first model call includes loading. Record that separately
from later warm requests. GPU telemetry samples device-wide utilization and VRAM
once per second; it is not a process allocation trace. Unknown electricity cost,
RAM peak and streaming first-token latency cannot be inferred from those samples.
Input/output counts include every successful call, while failed attempts retain
any usage the server returned. A timeout may leave server-side work uncertain.
Prefix-cache effects are not independently disabled. This pilot does not establish
production throughput, concurrency limits or a stable tail-latency distribution.

## Next qualification step

Use these controls to select a capable reader and verify task plumbing. Add dense
retrieval, the qualified hybrid and actual runtime compilation as separate matched
treatments. Freeze new development/calibration/final scopes before tuning a
reader-aware acquisition policy. Include distributed evidence, contradictions,
missing facts, temporal changes, authorization boundaries and changed source
versions. Retain the complete authorized task scope for conflict checks even when
only a small working set is resident.

Promotion requires a paired quality interval and a measured end-to-end resource
benefit; fewer prompt tokens alone are insufficient. Frontier adapters remain
preparation only until separate provider access and spending are authorized.
Coding-agent evaluation follows the separate Harbor track in the
[evaluation programme](evaluation-programme.md).

Sources and licensing: [experiment attribution](../evidence/longbench-v2-pilot/ATTRIBUTION.md),
[official dataset](https://huggingface.co/datasets/zai-org/LongBench-v2),
[native evaluation code](https://github.com/bys0318/LongBench-v2/blob/ef5ccc4bdcb1d505455517e4b419a50bb959862a/pred.py).

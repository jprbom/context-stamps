# Local SLM evaluation

Selected evidence reduced measured input tokens by about 75% while matching a full-current-context baseline's correctness on short fictional tasks. Latency was mixed. This experiment tests a narrow integration pattern with caller-supplied required sources; it does not establish general coding-agent performance, security, or edge-device efficiency.

## Executed setup

Qwen2.5-1.5B-Instruct, Ollama `qwen2.5:1.5b`, Q4_K_M, 4,096-token context, temperature 0 and fixed seed. The exact model digest and Ollama version are in the [manifest](../evidence/local-tasks-v1/manifest.json). The local model ran on an NVIDIA GeForce RTX 5080 Laptop GPU, with Intel Core Ultra 9 275HX host CPU. This is a laptop GPU experiment, not mobile/ARM testing. The weights were downloaded from the public [Ollama model distribution](https://ollama.com/library/qwen2.5:1.5b); see the [upstream model card](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct). No private model, private corpus or API service was used.

There are 24 fictional cases, two repetitions and three modes: 144 calls. Twelve cases ask for a changed numeric specification. Twelve restricted code-edit cases ask for a Boolean expression after a comparison operator changes. Those twelve contain only **four distinct rule patterns**, repeated three times. Repetitions are not independent quality samples. Code output is graded on 20 boundary points through a restricted AST interpreter; generated code is never executed with `eval`, `exec`, imports or a shell.

Each case updates a source, invalidates a declared dependent plan, and adds eight irrelevant records. The modes are:

- **Full current:** all current original records, excluding the known stale plan. This is a strong correctness baseline.
- **Selected:** a 700-byte packet from `select_evidence`, with the correct required-source ID supplied by the caller. The packet contains original text and source headers.
- **Stale cache control:** the old derived plan without version checking. This deliberately broken control demonstrates the consequence of stale evidence; it is not a competitive retrieval baseline.

The input budget uses bytes, while reported prompt tokens come from Ollama's actual `prompt_eval_count`. The same system instruction and task are used in every mode.

## Results

Each table row summarizes 24 calls: 12 fixtures, twice. Accuracy is the fraction with the expected numeric answer or a passing Boolean expression. Citation accuracy checks the returned source filename separately.

| Task | Context | Correct | Stale answer | Citation correct | Mean input tokens | Mean request ms | Mean pipeline ms including ingestion |
|---|---|---:|---:|---:|---:|---:|---:|
| QA | Full current | 100% | 0% | 100% | 1,162.75 | 181.6 | 253.3 |
| QA | Selected | 100% | 0% | 100% | 297.25 | 262.9 | 334.7 |
| QA | Stale cache control | 0% | 100% | 100% | 177.42 | 129.1 | 193.9 |
| Code expression | Full current | 100% | 0% | 70.8% | 1,162.25 | 158.9 | 229.6 |
| Code expression | Selected | 100% | 0% | 100% | 293.25 | 130.7 | 202.6 |
| Code expression | Stale cache control | 0% | 100% | 100% | 172.00 | 126.8 | 191.5 |

Input-token reductions are **74.4% for QA** and **74.8% for code expressions** versus full current context. Both methods answered correctly, so this is a context-size result rather than a task-accuracy gain over current evidence. A correct filename also does not imply a current or faithful answer, as the stale control demonstrates.

QA mean latency increased despite fewer tokens. Do not infer faster inference or production savings from prompt length alone. Mode order was shuffled and the model was warmed; ordinary Ollama caching remained enabled. A CPU retrieval replication ran concurrently, so the host was not isolated. Stored per-call timing includes request time, mode-specific packing time and shared ingestion time. The reported end-to-end value is their sum for a request; it excludes installation, model download and warm-up. Ingestion uses the same local memory construction for all modes. It is not a claim about an optimized application's full lifecycle.

## Inspect and reproduce

- [Fictional inputs](../evidence/local-tasks-v1/fixtures.json) and [protocol](../evidence/local-tasks-v1/protocol.json).
- [All prompts, responses, grades and timings](../evidence/local-tasks-v1/per-call.jsonl).
- [Aggregate results](../evidence/local-tasks-v1/summary.json) and [runner](../experiments/run_local_tasks.py).

Install Ollama, start its local service, and download the public model. This uses approximately 1 GB for the quantized download; runtime memory differs. The runner connects only to `127.0.0.1:11434`.

```bash
ollama pull qwen2.5:1.5b
python -m pip install -e .
python experiments/run_local_tasks.py --out evidence/local-tasks-reproduction
```

Check the downloaded digest against the manifest before comparing results; a tag can change. Use a fresh output directory. The experiment records all calls and refuses to append to an existing call log. For higher-confidence performance results, repeat on an otherwise idle host with a stated cache policy and test multiple model sizes and actual target edge hardware.

## Next validation

The next coding evaluation needs real repository changes, realistic distractors, multiple files and tests beyond four comparison rules. The next answer-quality evaluation needs open-ended questions, missing required evidence and independent grading. Source discovery should be tested separately from required-source enforcement. These fixtures must not become the training set for a claimed held-out robustness result. No generative model was trained or fine-tuned in this follow-up.

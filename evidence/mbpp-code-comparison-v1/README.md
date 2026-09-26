# Full local code-adaptation comparison

By Prashant Jagtap

The adapter does **not improve aggregate correctness** in this experiment.
Both arms pass 49/164 tasks (29.88%). Nineteen tasks improve and nineteen regress.
The adapter reduces output-format failures but increases generated tokens and
measured generation time. It remains inactive.

| Measurement | Unchanged base | Local adapter |
|---|---:|---:|
| Pass all native base and additional tests | 49/164 | 49/164 |
| Output-format failures | 20 | 5 |
| Fail native base tests | 91 | 101 |
| Pass base tests, fail additional tests | 4 | 9 |
| Truncated generations | 0 | 0 |
| Driver errors / container boundary failures | 0 / 0 | 0 / 0 |
| Input tokens | 27,260 | 27,260 |
| Generated tokens | 10,192 | 10,936 |
| Summed generation batch time | 131.77 s | 179.17 s |
| Median generation batch time | 2.64 s | 3.40 s |
| p95 generation batch time | 5.64 s | 8.43 s |

Generation has 41 batches of four tasks per arm. Batch timings are counted once,
not four times. The token increase is about 7.3%; summed generation time increases
about 36.0%. These timings exclude loading, context preparation, grading and
training. They are not per-request latency, TTFT or total workflow cost. Peak
Torch allocation is approximately 3.04 GiB in both arms and excludes driver and
system memory. Energy remains unmeasured.

The paired accuracy change is 0.00 percentage points. A seeded 10,000-resample
paired bootstrap gives a descriptive 95% interval of −7.32 to +7.32 percentage
points; exact two-sided McNemar p = 1.0. Equal point estimates do not establish
equivalence, and this interval does not establish the programme's one-percentage-
point noninferiority target. The statistical interpretation assumes independent
tasks; related task families and four previously inspected smoke tasks limit it.

## Protocol and evidence

- Qwen2.5-1.5B-Instruct, the same BF16/SDPA backend and base parameters; adapter
  enabled versus disabled. Randomized paired arm order, greedy decoding, one
  training seed, fixed batch size and token limits. No paid model calls.
- A custom complete-function prompt uses only the public HumanEval problem.
  Hidden tests and reference solutions are not sent to the model. This is a
  local paired protocol, not a published leaderboard reproduction.
- Native pinned EvalPlus base and full additional tests run in fresh, non-root,
  offline containers, with bounded resources and no host mounts. All 328 outputs
  are retained; 303 enter native grading and 25 fail the declared format check.
  No containers remain according to the per-run cleanup checks.
- The reference numerical failure on HumanEval/32 remains in the denominator.
  The official oracle and tolerance are unchanged. See the
  [grader audit](../humaneval-grader-v1/README.md).
- `generations.jsonl.gz` preserves the original generation-log bytes.
  `records.json.gz` retains both plans, completion records, all native outputs,
  isolation profiles and failure records. `summary.json` contains derived
  quality/resource statistics. `manifest.json` binds files and implementation.

The [nineteen regressions](failure-review.md) are now development observations.
Further changes informed by them need a fresh evaluation scope. The training
checkpoint, recipe and completed run are not overwritten.

```bash
python experiments/test_code_statistics.py -v
python experiments/verify_full_code_eval.py
```

For a new completed local run, export before reviewing/publicizing its evidence:

```bash
python experiments/export_code_comparison.py --generation ../code-generation-run --grading ../code-grading-run --out ../code-comparison-export
python experiments/verify_full_code_eval.py --evidence ../code-comparison-export
```

Replay checks data and recorded native outcomes; it does not rerun model
generation or execute candidate programs. Full reproduction requires the pinned
base, adapter, GPU environment, benchmark release and container image described
in the [local guide](../../docs/verified-code-learning.md).

This experiment isolates generic supervised adaptation. It contains no paired
Context Stamps memory/compiler treatment, older-domain retention assessment,
quantized edge deployment or frontier-model comparison. Native benchmark grading
is not an adversarially tamper-proof learning oracle. Source and dataset notices
remain in the linked training/grader evidence; project code copyright belongs
to Prashant Jagtap under the repository license.

# ChartQA local training diagnostic

By Prashant Jagtap. This is a **TRAIN-only** diagnostic, not a ChartQA leaderboard
score, a held-out evaluation, or a qualified visual-memory improvement.

The [aggregate record](summary.json) was exported from a completed, source-pinned
local run. It includes no chart images, question text, answer keys, case IDs or
raw model responses. Those artifacts remain outside this MIT repository under
the [ChartQA publisher's terms](https://github.com/vis-nlp/ChartQA), pinned at
`044eabfc306abfe9340c5741f0093aefc5973d06`.

The selected TRAIN cohort contains 64 charts and 143 questions (54 human, 89
augmented), with two to four questions per chart. The same unchanged
`qwen3.5:4b` local reader (4.7B, Q4_K_M) answered three matched arms:

| Arm | Relaxed correct | Stripped exact | Entire charts correct | Model calls | Tokens, including shared extraction | Summed request wall time, seconds |
|---|---:|---:|---:|---:|---:|---:|
| Direct image | 112/143 | 94/143 | 39/64 | 143 | 84,405 | 92.2 |
| Extracted text memory | 79/143 | 68/143 | 21/64 | 207 | 207,561 | 650.6 |
| Extracted table + program | 51/143 | 39/143 | 13/64 | 207 | 241,129 | 661.3 |

The native relaxed scorer accepts numeric predictions within 5% and uses a
stripped-string fallback; exact agreement is shown separately. Each memory or
program plan pays for one extraction and memory build per chart, including
failed extractions. These are summed measured request costs for hypothetical
single-arm plans, not end-to-end throughput or device energy. The paired
collection itself made 493 calls in 841.1 seconds. CPU checks ran concurrently
at times, so these timings are not isolated GPU performance claims.

The predeclared four-cell CPU policy family used only the question available
before answering. Three penalties were fitted with four chart-group folds,
retaining 12 fold policies and three full fits. **Every fold selected direct**;
all cross-validated policies equal the direct baseline on this cohort. The
selection gate chose no candidate and nothing was activated. Only one memory
answer and two program answers rescued a direct error, while 28 questions were
wrong in all three arms. An oracle allowed to inspect the labels could reach at
most 115/143 from these fixed responses. This is a diagnostic upper bound, not
a deployable routing result.

The main failure is upstream of arithmetic: extraction, chart grounding and
question-to-cell interpretation do not preserve enough of the direct model's
answer quality. The later [host-declared numerical API](../../docs/numerical-tasks.md)
is a separate exact execution path. It was **not** swapped into this frozen
reader, and its arithmetic tests cannot change this measured score.

Validation (32 charts, 78 questions) and test (64 charts, 145 questions) were
left unopened by the reader and fitter. Exact decoded-pixel overlap between the
selected cohorts is zero; near duplicates or model pretraining exposure remain
possible. No model weights changed. Peak memory, energy, target-edge latency
and production scale were not measured.

The local source and data hashes, model digest, full cost denominators, subgroup
counts and four question-cell aggregates are in `summary.json`. The original
run retains all 2,118 recorded artifacts and 17 archived source files; their
hashes were checked against `complete.json` before scoring. To verify this
published aggregate against those exact retained local artifacts:

```powershell
python experiments/export_chartqa_training_summary.py `
  --data ../chartqa-native-prepared-v2 `
  --runs ../chartqa-training-reader-v2 `
  --policy ../chartqa-training-policy-v1 `
  --summary evidence/chartqa-training-v1/summary.json
```

A new inference run can differ despite the fixed inputs and model settings;
the command above verifies the specific recorded run, not every future replay.
See the [protocol and preparation history](../../docs/chartqa-local-learning-draft.md).

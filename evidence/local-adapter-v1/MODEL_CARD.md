# Local workflow adapter: experimental, inactive

Research and authored fixtures: Prashant Jagtap.

This is a 2,193,720-byte LoRA adapter for the separately obtained
`Qwen/Qwen2.5-1.5B-Instruct` base at revision
`989aa7980e4cf806f80c7fef2b1adb7bc71aa306`. It is not a standalone SLM.
The base remains unchanged. The adapter is **not approved for deployment**.

## Intended research use

Reproduce a bounded local fine-tuning experiment on six explicit workflow
decisions: abstain, refresh stale evidence, acquire missing evidence, repair,
verify and finish. Inputs are authored synthetic host-state records. Output is
the highest-scoring next token among six letters; no free-form agent actions
are executed. The deterministic contract is fully specified in the prompt and
can be implemented exactly without a language model.

## Training

- One RTX 5080 Laptop GPU, Python 3.12.10; exact package versions in `plan.json`.
- Rank 4, alpha 8, query/value projections, 544,768 trainable parameters.
- BF16 base, FP32 adapter parameters; AdamW, learning rate 0.0002, two epochs.
- 144 authored training rows in 24 fixture-project clusters; 72 optimizer steps.
- 48 calibration rows, 72 test rows and 24 arithmetic retention controls remain
  outside fitting. These share templates; they are not real project holdouts.
- Training loop: 27.73 seconds. Peak Torch CUDA allocation: 3,501,706,752 bytes
  (3.26 GiB), excluding untracked driver/system allocations. Energy unmeasured.
- Frozen parameter fingerprints agree before and after fitting. Reloading the
  saved adapter reproduces all eight base/adapter predictions and six-choice
  logit vectors in one recorded four-row batch exactly.

## Recorded outcomes

| Partition | Base correct | Adapter correct | Exact rule correct |
|---|---:|---:|---:|
| Calibration workflow | 11/48 | 35/48 | 48/48 |
| Test workflow | 12/72 | 52/72 | 72/72 |
| Arithmetic retention format | 2/24 | 5/24 | Not scored here |

One previously correct test row becomes incorrect. Missing-evidence decisions
remain especially weak (2/12 correct). The adapter fails 20/72 workflow rows.
The exact rule wins and needs no SLM call. This result does not justify using
this adapter instead of that rule or activating a new production policy.

The baseline's multiple-choice retention accuracy is unexpectedly low. The
tokenizer matches the pinned tokenizer file; separate direct-number controls
answer 2+2 and 36+12 correctly in BF16 and FP32. Multiple-choice failures persist
in those diagnostics. SDPA/eager attention and batch-shape checks show logit
differences and some attention-backend prediction changes. Do not interpret
this narrow letter-selection task as general arithmetic competence, broad
retention, a public benchmark score or validated precision parity.

## Limits and rights

No public benchmark prompts, solutions, graders, local user records or model
self-assessments enter training. No hosted-provider calls or training uploads
occur. Local fixtures are MIT licensed with the project. Qwen base attribution
and Apache License 2.0 terms remain applicable to the base and this research
adapter; see `adapter/BASE_MODEL_LICENSE.txt`. The adaptation was produced by
the local experiment described here. Original Qwen ownership is unchanged.

The 256-bit Context Stamp remains a reference to external evidence. This
experiment does not train a decoder for facts hidden in a hash, improve base
attention universally, qualify an embedded device, or implement autonomous
weight promotion. Do not use a model prediction as permission or verification.

[Local RTX reproduction and integration requirements](../../docs/local-adapter-rtx.md).

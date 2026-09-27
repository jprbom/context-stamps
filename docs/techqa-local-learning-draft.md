# TechQA local learning — protocol and completed result

By Prashant Jagtap.

The [completed study](../evidence/techqa-local-v1/README.md) records 1,774 measured
requests and three warmups on the local `qwen3.5:4b` reader. All three fitted
policies fail calibration. On 310 native development questions, checked quotations
lower false positives from 95/150 for the direct exact-span control to 31/150,
while answerable F1 falls from 13.64 to 7.10 and summed request time rises 2.33
times. Always abstaining has higher combined native F1. No candidate activates.

The purpose is to test whether a small local policy can reduce unsupported answers
while preserving useful extractive answers. This is separate from changing the
language-model weights. It is also separate from proving that the system saves
tokens: a gate evaluated after generation cannot recover generation cost already
incurred.

## Data and source boundaries

The [original TechQA implementation](https://github.com/IBM/techqa) links to the
[PrimeQA archive](https://huggingface.co/datasets/PrimeQA/TechQA). The pinned archive
has 600 native training questions and 310 public development questions. Candidate
document IDs are legitimate task inputs. Answerability, the correct document,
answer text and answer offsets are labels and must not enter context selection.

The archive README specifies **CDLA-Permissive-1.0** for the data. Its notice takes
precedence in this workflow over the conflicting Hub metadata. Keep the raw
archive, selected documents, question keys and raw model prompts outside the Git
repository. Preserve the original licence, attribution and source revisions.

The first streaming attempt failed at its 128 MiB selected-member cap. It did not
retain a complete archive or validate the final archive checksum. The subsequent
attempt retains the compressed archive first and allows at most 512 MiB for each
whitelisted member. It never extracts arbitrary archive paths or runs upstream
training code.

## Statistical approach

Group normalized questions before assigning cohorts. Withhold 23 native training
questions that duplicate development questions. A fixed group-hash order assigns
400 fitting cases and 177 calibration cases while keeping seven training
duplicate groups together. Retain all 310 native development questions, including
two duplicate groups. There are no normalized question duplicates across the
three resulting cohorts. Candidate documents still overlap: 2,619 between fit
and calibration, 3,379 between fit and development, and 1,842 between calibration
and development. This is not a source-disjoint sample; unknown base-model
pretraining exposure cannot be excluded.

The actual files use `QUESTION_TEXT`, although the archive README describes
`QUESTION_BODY`. Two questions repeat a candidate document ID; only the repeated
reference is deduplicated. Numeric answer offsets are strings, as accepted by
the native scorer. Eleven development `ANSWER` strings differ from the text at
their native character offsets. Preserve those original keys, report the issue
and use the declared native offset metric. Four failed preparation attempts and
their exact source versions are retained before the successful fifth attempt.

Use BM25 over overlapping 200-word source windows, with a 150-word stride. Select
at most eight windows that fit a 7,000-token prompt budget for both registered
tokenizers and both response modes. Keep the entire question or record a compiler
failure. Never silently truncate the native question or drop failed cases.

The two response modes are an answer-only direct control and a quote-first
treatment. Context text is identical. The response interface differs. A fixed
check verifies exact quotation, current authorisation and answer inclusion.

Fit a small ridge model from twelve observable input/output features, including
retrieval score margins, query overlap, answer length and quotation coverage.
Its target is the utility of answering relative to abstaining: native character
span F1 on answerable questions, and minus one on unanswerable questions. The
score is not a calibrated probability or proof of entailment.

The fit uses CPU linear algebra on a 13-coefficient system, including the
intercept. Three regularisation strengths and six thresholds are selected using
only calibration outcomes. A candidate must preserve calibration positive F1,
reduce false positives and improve aggregate native F1. Otherwise retain the
fixed citation check. Calibration selection is exploratory, not a formal risk
guarantee. No candidate is automatically activated.

An int16 coefficient export measures storage approximation and threshold flips.
Serving reconstructs floating-point arithmetic. It does not demonstrate integer
kernel acceleration or a speed advantage.

## Local commands

Use a local environment with `numpy`, `tokenizers`, `huggingface_hub` and the
repository installed. Pin dependency versions in the eventual result manifest.
The core library's offline citation example has no model dependency. These
experiment commands reproduce the completed modern-reader workflow. Source and
measured checkpoint identities are retained with the result.

```powershell
hf download PrimeQA/TechQA TechQA.tar.gz --repo-type dataset --revision 60437bc79ab217679682217598a3693cab78365b --local-dir ../techqa-archive-v2
python experiments/extract_techqa.py --archive ../techqa-archive-v2/TechQA.tar.gz --output ../techqa-data-v2
python experiments/techqa_prepare.py --data ../techqa-data-v2 --output ../techqa-prepared-v5
python experiments/techqa_reader.py prepare --data ../techqa-prepared-v5 --output ../techqa-reader-v1
python experiments/techqa_reader.py run --output ../techqa-reader-v1 --phase fit --model modern
python experiments/techqa_reader.py run --output ../techqa-reader-v1 --phase calibration --model modern
python experiments/train_techqa_policy.py --data ../techqa-prepared-v5 --runs ../techqa-reader-v1 --output ../techqa-policy-v1 --model modern
python experiments/techqa_reader.py run --output ../techqa-reader-v1 --phase development --model modern --trained ../techqa-policy-v1
python experiments/eval_techqa.py --data ../techqa-prepared-v5 --runs ../techqa-reader-v1 --trained ../techqa-policy-v1 --output ../techqa-analysis-modern-v1 --model modern
```

The registered readers are local Ollama `qwen3.5:4b` and `qwen2.5:1.5b`; no cloud
fallback is available. Obtain the matching tokenizers in the locations declared
in `experiments/cited_reader.py`. The Qwen3.5 tokenizer was downloaded from
`Qwen/Qwen3.5-4B` at revision `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`.
Registration captures actual model digests and tokenizer hashes before inference.
The revised interface's fictional canaries are retained separately and reveal
that the small reader may fail to supply quotations.

Run one inference workload at a time. The runner refuses to evict an existing
resident model. It stores a raw response before parsing and never automatically
retries an uncertain failed request. Do not remove a stale run lock until the
owned process and any pending request have been checked.

## Required report

Compare raw answers, exact-span direct answers, raw cited answers, checked cited
answers, the selected policy and abstain-all. Report positive answer F1 separately
from unanswerable false positives, useful answers lost, compiler/format failures,
truncations, coverage and the reader's actual tokens and request latency. Include
the costs of context compilation and checking. Keep all cases in the denominator.

Use the pinned native character-offset scorer. Its `Best_QA_F1` field chooses a
threshold using evaluation answers and must not be presented as an achieved
frozen-policy result. The local adapter excludes that oracle field from reported
metrics. Paired question bootstrap intervals are descriptive because documents
may overlap across questions.

Freeze policy selection before the development requests. Open development keys
for scoring only after all expected paired outputs and their hashes verify.
Publish failures as well as gains. Independent retention tasks and actual device
measurements are still required before a deployment candidate can be accepted.

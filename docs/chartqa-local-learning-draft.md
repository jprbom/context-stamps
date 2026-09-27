# Local visual memory and arithmetic — preparation draft

By Prashant Jagtap. No ChartQA model result or learned improvement is claimed yet.

The later [actual authored interface canary](../evidence/chartqa-interface-v1/README.md)
made four local calls and failed. Extraction supplies the three fixture values and
the arithmetic program returns 60, but direct/memory replies contain invalid answer
strings inside valid JSON. A 24-call comparison of four authored questions retains
all baseline, schema-grounded and JSON-only outcomes. Schema grounding improves
contract-correct answers from 1/4 to 3/4 for direct vision and 2/4 to 3/4 for memory,
but neither passes every case. Six JSON-only numeric answers are rejected by the
original string-only contract; a prospective typed-scalar protocol is needed.
These authored checks do not use ChartQA questions or train a model.

The intended application is repeated analysis of a chart on a local small model.
The experiment will compare reading the image for every question with reusing a
source-bound extraction, then test whether bounded numerical operations improve
answers over a reader working from that same extraction. Extraction can be wrong
and can cost more than rereading the image; both effects must remain in the result.

## Data that are ready

The publisher's [ChartQA release](https://github.com/vis-nlp/ChartQA) is pinned to
`044eabfc306abfe9340c5741f0093aefc5973d06`. Its code/data notices declare GPL-3.0;
the original paper also discusses the source-chart terms. Keep the source data,
images and answer keys outside this MIT-licensed code repository with their own
notices. The original authors are Ahmed Masry, Do Xuan Long, Jia Qing Tan, Shafiq
Joty and Enamul Hoque; cite their [ChartQA paper](https://aclanthology.org/2022.findings-acl.177/).

The prepared repeated-chart subset contains:

| Native split | Selected charts | Questions | Human / machine questions |
|---|---:|---:|---:|
| Train | 64 | 143 | 54 / 89 |
| Validation | 32 | 78 | 41 / 37 |
| Test | 64 | 145 | 101 / 44 |

Selection uses a fixed seed and input/source identities, with two to four distinct
questions per chart. Earlier splits exclude byte-identical images found in later
splits. A second check decodes every selected image and finds zero identical pixel
groups across these cohorts. Near-duplicates and base-model pretraining exposure
are not ruled out. The differing human/machine mix requires separate subgroup
reports. This is a workflow subset, not a full ChartQA leaderboard evaluation.

All 172 prepared files pass their recorded hashes. The selected images occupy
7,018,856 bytes. Reader inputs contain questions and image provenance; supplied
reference answers are copied separately. No ground-truth data table or chart
annotation enters the proposed reader context.

## Planned comparison

1. **Direct vision control:** the same local model receives the complete image
   and each question.
2. **Text-memory control:** extract a chart once, retain its cells, labels, series,
   colours and units as unverified model output, then answer from that memory.
3. **Numerical treatment:** use the same extraction, ask for a bounded expression
   over explicit cell IDs and execute supported operations locally.
4. **Learned choice:** fit a small policy on training outcomes and select its
   settings on validation only. Freeze it before scoring the selected test cases.

The installed `qwen3.5:4b` is a possible common reader; the local server advertises
vision support. The implemented protocol uses the local chat endpoint, the same
model for all three answer methods, a 16,384-token context limit, temperature zero,
seed 101 and non-thinking output. Extraction allows 4,096 output tokens; each
answer/program allows 1,024. JSON schema field order is preserved. These settings
are development choices, not a qualified run registration. The actual vision
interface still needs the separate four-call authored canary and a frozen study.
No paid provider or model-based external judge is planned. The current TechQA
experiment keeps exclusive use of the GPU until it finishes.

The policy must use information available before choosing a query-time route.
Answers, correctness labels and competing-arm outputs cannot be routing features.
Compare it with the best fixed method and simple routing rules. If it does not
help, retain the fixed control. This study alone cannot activate a candidate;
domain retention, prospective local feedback and device qualification remain.

## Numerical component implemented

`experiments/chartqa_quant.py` uses rational arithmetic for sums, means,
differences, ratios, percentages, comparisons and extrema. It preserves selected
cell IDs, question-supplied numeric constants and units in the result. Units must
be compatible; missing values are not imputed; zero division and ambiguous extrema
return errors. Expression depth, cell counts, JSON size and rational size are bounded.
Model-provided Python, shell code and arbitrary answer literals are not executed.

A successful expression has `execution_verified=True` and
`source_semantics_verified=False`. Correct calculation does not establish that the
model extracted the right values or selected the right cells. Such outputs cannot
become local training labels merely because the arithmetic passes.

Thirty experiment tests pass: eight preparation, eight numerical, three scoring,
eight source-bound protocol and three canary-lifecycle tests. Inference in the
lifecycle tests is mocked; these tests make **zero actual model calls**. Early tests caught
sentence punctuation in question constants and JSON nesting beyond the intended
bound; both were corrected before any model measurement. These are component
checks, not visual task scores or evidence of model improvement.

## Scoring and cost requirements

The paper defines a 5% tolerance for numeric answers. The pinned original VL-T5
`evaluate_raw` implementation uses stripped exact strings instead. The commonly
used [Pix2Struct relaxed-correctness implementation](https://github.com/google-research/pix2struct/blob/main/pix2struct/metrics.py)
also specifies percentage conversion and a string fallback for zero-valued targets.
The reference is now pinned to `6fe25c1dc8151823ee3b479519d8d5948812fee4`, with source
SHA-256 `375d5970dd3c05f71b27934eabfa3e0a400c0eb459ba540a387e5c7ec6e8cecd`.
The local scorer agrees with its reviewed function on all 21 authored canaries,
including tolerance boundaries, percentages, zero, whitespace and nonfinite text.
Report stripped exact agreement alongside relaxed correctness. No grading rule
was changed to improve a score. The [preparation record](../evidence/chartqa-preparation-v1/README.md)
preserves source fingerprints and canary outcomes without redistributing the charts.

Charge extraction, state preparation, query calls, verification and any fallback
to each treatment that uses them. Report first-question and subsequent-question
costs separately, alongside whole-chart totals, correctness and abstentions.
Count actual server tokens rather than estimating savings from image bytes or a
32-byte stamp. Use chart-group uncertainty estimates, retain every failed call and
report CPU/GPU contention and missing energy measurements. Adapter success does
not establish model quality, autonomous learning or edge-device performance.

## Local preparation

Use a separate CPU environment with Pillow (12.3.0 in the measured preparation);
no Parquet module is required for the native source path. From a reviewed clone:

```powershell
python -m pip install -e . Pillow==12.3.0
python experiments/fetch_chartqa_native.py --output ../chartqa-native-source
python experiments/chartqa_native_prepare.py --source ../chartqa-native-source --output ../chartqa-native-prepared
python -m unittest discover -s experiments -p 'test_chartqa_*.py' -v
python experiments/verify_chartqa_preparation.py
```

After the existing GPU workload finishes, the Windows interface canary can run
against the separately installed local `qwen3.5:4b` model:

```powershell
python experiments/chartqa_canary.py --output ../chartqa-canary-v1 --font C:/Windows/Fonts/arial.ttf
```

It renders an authored three-bar chart and records four local requests: direct
vision, extraction, text-memory answer and expression generation. Each raw response
is saved before parsing. The expected sum is declared before generation. A failed
call is retained, never retried automatically; an existing resident workload blocks
the canary without eviction. The model digest, server, font, source files and exact
requests are recorded. This is an interface check, not training or a benchmark.
**No actual vision canary had run when the preparation record was frozen.** Passing it would
only permit registering the public-data comparison; it would not qualify a model.

The first Hub download command treated a wildcard as a literal filename and
failed; the corrected command retained all five pinned Parquet shards. Windows
Application Control then blocked that native reader before decoding. The blocked
module was not retried. The publisher's original JSON/PNG format provided the
working path. Its first preparation rejected legitimate punctuation in 17 source
filenames; the corrected version URL-encodes publisher names, rejects path
separators and checks each downloaded PNG against its pinned source identity.
Both failed preparation sources and logs remain retained locally.

This document is a preparation record. It does not register a complete model
experiment or announce a ChartQA result.

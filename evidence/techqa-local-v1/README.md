# Local TechQA learning and reader comparison

By Prashant Jagtap.

**This study does not qualify a useful learned update.** All three fitted policies
fail the calibration preservation rule. The development run reduces false positives
through source checks, but loses answer quality and costs more than the direct
interface. Every model candidate remains inactive.

The local `qwen3.5:4b` reader made **1,774 measured requests**: 800 fitting, 354
calibration and 620 development requests, plus three separately recorded warmups.
Only the modern reader was run; registering another tokenizer does not constitute
another model evaluation. No language-model weights changed.

## Complete native development results

There are 160 answerable and 150 unanswerable questions. F1 measures character-span
overlap in the correct document; it is not percentage answer accuracy.

| Method | Overall native F1, 0–100 | Answerable F1, 0–100 | Unanswerable false positives | Answerable abstentions |
|---|---:|---:|---:|---:|
| Direct raw answer | 8.65 | 13.64 | 145/150 | 4/160 |
| Direct + exact-span filter | 24.78 | 13.64 | 95/150 | 43/160 |
| Quoted raw answer | 9.98 | 11.21 | 137/150 | 10/160 |
| Quoted + source check | 42.05 | 7.10 | 31/150 | 112/160 |
| Selected policy: fixed source check | 42.05 | 7.10 | 31/150 | 112/160 |
| Always abstain | 48.39 | 0.00 | 0/150 | 160/160 |

The direct exact-span filter removes 50 false positives without reducing recorded
positive F1, but still leaves 95/150 false positives. The quoted check reduces that
count to 31, but drops all native overlap on 19 answerable questions previously
receiving credit and gains overlap on four. Across all questions there are 75 F1
gains and 23 losses, including 68 gains and four losses on unanswerable questions.
The combined F1 increase must not conceal reduced useful answering. Always
abstaining scores higher on that combined metric.

![Native TechQA quality and reader cost](../../docs/assets/techqa-local-v1.png)

[Vector figure](../../docs/assets/techqa-local-v1.svg)

## Cost and failures

| Measured quantity, 310 requests each | Direct interface | Quoted interface |
|---|---:|---:|
| Input tokens | 1,172,866 | 1,181,856 |
| Output tokens | 11,672 | 78,421 |
| Summed request wall time | 505.73 s | 1,177.99 s |
| Median request wall time | 1.53 s | 3.71 s |
| p95 request wall time | 2.38 s | 6.30 s |
| Post-response processing | 0.502 s | 0.538 s |

The quoted interface uses 6.39% more total model tokens and 2.33 times the summed
request time. Filters operate after generation and retain its cost. Shared context
compilation takes 33.34 seconds for development and is reported separately.
Warmups, prior reader-data collection and fitting costs are separate; the small
ridge solve is not the total cost of learning. CPU preparation/tests ran on the
host during inference. These measurements are neither isolated throughput nor
edge-device or energy qualification.

The quoted response on `DEV_Q129` reaches the output limit and contains incomplete
JSON. It remains in every denominator. No request was automatically retried or
removed to improve the score. `failure-review.json` retains the failure and quality
transitions. All raw replies and prompts remain outside the repo.

Only 81/160 answerable cases have a complete native reference in one selected
window; 100/160 have their correct document selected. The training-only audit
identifies a separate reader/interface problem: among 295 answerable fitting
cases, 147 have a complete reference in one window, but 124 of those have no
accepted overlapping answer. Of 193 invalid citation proposals, 168 include a
quote absent verbatim from every selected source and 32 include a quote assigned
to another source. These overlapping counts are diagnostics, not independent test
results. Better selection and an existing-span-ID answer interface are next text
experiments; this recorded run is unchanged.

## Training and evidence boundaries

The 12-feature ridge model has 13 coefficients including its intercept. It fits
on 129 source-bound outputs from 400 fitting questions. Its target is native span
F1 on answerable cases, or −1 for answering an unanswerable question. Three
regularisation strengths and six thresholds are examined only on 177 calibration
questions. None preserves positive F1 while reducing false positives and improving
combined F1. Int16 coefficients are a storage diagnostic; inference uses FP64.

All 310 native public development questions are retained. Normalized question
duplicates are excluded or grouped before inference. Candidate documents still
overlap across cohorts. The question-cluster bootstrap is descriptive; it does
not establish IID significance, retention or deployment safety. Eleven original
development answer strings differ from the native offset spans. Original keys
are preserved and native offset scoring is used. The native oracle threshold
statistic is excluded from achieved results.

Native reference projections retain the publisher's CDLA-Permissive-1.0 terms,
included as [the original licence](CDLA-Permissive-v1.0.pdf) and
[data README](DATA_README.txt). Original code is MIT, copyright Prashant Jagtap.
The corpus, complete questions, generated answer text and prompts are not bundled.
Records include target offsets, predictions, check metadata, observable features,
costs and raw-record fingerprints. Rechecking every quotation additionally requires
the separately downloaded corpus and retained raw requests. Fingerprints do not
prove semantic correctness.

```powershell
python -m pip install -e . numpy
python -m unittest discover -s experiments -p 'test_techqa_*.py' -v
python experiments/verify_techqa.py
```

The independent replay recomputes offset scores, reconstructs every control,
refits all three policies using augmented least squares, checks all 18 calibration
settings, and recomputes costs and paired descriptive intervals. It makes no model
or network call. Historical source bytes are retained as data and never executed
by the verifier. The [RTX runbook](../../docs/techqa-local-learning-draft.md) gives
the full data and model procedure.

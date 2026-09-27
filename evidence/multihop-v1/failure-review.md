# Failures and unresolved questions

All 1,024 predictions remain in the evidence. The candidate is inactive; lower token count and stronger support retention do not override answer failures.

| Gap | Observed evidence | Consequence and next test |
|---|---|---|
| Retrieval misses | Diffusion omits at least one required support in 26/64 answerable cases | Improve candidate/selection recall on new source families; keep exact six-passage and measured token-budget controls |
| Reader fails with complete evidence | The 1.5B reader answers only 9/38 diffusion cases with all native supports exactly | Test explicit source-grounded intermediate decisions and a general-purpose reader, counting all added calls and regressions |
| Unsupported answering | Diffusion answers 27/64 unanswerable cases for 1.5B; the 7B reference answers 4/64 | Fit/calibrate a separate abstention stage; a ranking distribution is not a sufficiency certificate |
| Uncertain answer gain | 1.5B diffusion minus BM25 F1: +0.0281, descriptive 95% interval [-0.0745, +0.1317] | No answer-quality superiority claim; reserve new independent task clusters |
| New regressions | Against BM25, 1.5B diffusion has seven exact-answer gains and five regressions; reference has two gains and three regressions | Keep per-case paired results and older-skill retention, not only an aggregate mean |
| Cost versus a cheap control | 1.5B diffusion uses 13.5% more tokens than BM25 | Selection must earn full pipeline cost; top-k equality is not equal token cost |
| Larger reader is not a remedy | 7B diffusion exact answers 3/64, full context 8/64; both below this small reader's corresponding results | Coding specialization and prompt sensitivity remain confounds; the post-hoc reference is not a general size comparison |
| Evaluation scope | One filtered set of 64 question pairs, repeated with two readers; known exact overlaps excluded | No new holdout from rerunning the same pairs; pretraining/paraphrase contamination unresolved |
| Metric interpretation | Native support metrics grade selected context; one small-reader BM25 output truncates | Not a standard full-system leaderboard submission; truncation remains a failure opportunity in the denominator |
| Device qualification | Serial laptop request timing and Torch training allocation only | Need whole-process RAM/VRAM, energy, concurrency, retained skills and target-device budgets before deployment |
| Local update lifecycle | Training/evaluation are explicit commands; no candidate activates | Unattended training needs verified prospective labels and qualified promotion/rollback integration |

The native complete-pair result is especially restrictive: the small reader gets 5/64 pairs correct with diffusion and 5/64 with BM25; the reference gets 3/64 and 4/64. A correctly retrieved positive answer without appropriate abstention on its paired negative is not robust task completion.

`small/summary.json` and `reference/summary.json` enumerate every exact-answer gain and regression. Case/group score files preserve failures beyond those exact-answer changes. No model-based judge rewrites the native scoring or excludes an unfavorable result.

## Preserved preparation failures

The first partition request asked for 1,024 training groups. Strict source exclusions left only 487; preparation failed before fitting or generation. The revised request uses 400 groups, retaining the same exclusion rules and frozen evaluation/calibration selection. `preparation-history.json` retains the failure and exact source archive.

A first reader preparation reserved a 16K context. Measuring the longest complete input showed 3,712 tokens; the runner was re-prepared at 4K before any model call. The initial source and supersession record remain archived. No outcome was observed or discarded in making that preparation correction.

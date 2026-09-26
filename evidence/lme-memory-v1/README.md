# LongMemEval-V2 local memory development result

**Status: research candidate, inactive.** Context memory helps this small reader, but the learned policy is not reliable enough for deployment.

The small public haystacks supply 200 benchmark-generated web/enterprise trajectories with 5,095 recorded states. There are 294 text-only questions with native deterministic scoring. A fixed hash split selects 72 training questions; 222 remain for evaluation. Both splits share the same source histories. This is a development comparison, not independent retention qualification.

| Held-out method | Native credit | Reader input + output tokens | Summed generation time |
|---|---:|---:|---:|
| No memory | 21/222 | 41,017 | 73.889 s |
| State retrieval, FTS5 BM25 | 50/222 | 1,250,468 | 141.776 s |
| Linked state/change/path views | 50/222 | 1,323,568 | 152.592 s |
| Learned choice of state or linked views | 54/222 | 1,302,469 | 149.877 s |

![Measured memory comparison](../../docs/assets/lme-memory-v1.png)

The learned policy gains credit on 15 questions and loses it on 11 relative to state retrieval. Its net gain is four questions, with **4.16% more model tokens**. It selects linked views on 174 questions and state retrieval on 48. Learned-choice results reuse those measured outputs; they are not another set of reader calls. Timings exclude index construction and retrieval/packing. They do not establish a whole-workflow speed advantage.

## Training and execution

A CPU ridge model fits ten pre-answer features and an intercept from 72 paired measured outcomes. Its target trades native credit against token cost. Features describe the question and retrieved context; no answer keys or benchmark category labels enter them. Regularization and the utility tradeoff were frozen before generation. Fitting took 0.1245 seconds in this process, including its first NumPy import. Collecting the two training alternatives consumed **830,696 model tokens**; fitting time alone is not the learning cost.

The policy and all held-out choices were saved before held-out reader calls and scoring. Language-model weights remain unchanged. The reader is the pinned local Qwen2.5 1.5B Q4_K_M model, temperature zero, seed 71, with a complete input ceiling of 6,144 tokens and output ceiling of 256. There were **882 measured reader calls plus one warmup**, using 3,458,863 measured model tokens. No paid provider or remote judge was used.

The adapter stores source-bound observations, adjacent-state differences and recorded action order. It does not infer causality from adjacency, treat reported success as verified truth, or import agent thoughts. Current authorization and source identity are checked through canonical state nodes. A valid binding does not certify context sufficiency. This experiment does not benchmark 256-bit capsule search or modify internal model attention.

## Failures and limits

- The learned policy misses credit on **168/222** questions and introduces **11 regressions**. All are retained in [failure-review.json](failure-review.json). Linked views alone gain 16 and lose 16 versus state retrieval, leaving aggregate credit unchanged.
- Five outputs are truncated: one training and four held-out measured responses. All score zero. There are no request errors or tokenizer/server count mismatches. Failed and truncated responses stay in the denominator.
- Native phrase scoring checks reference phrases and can accept extra or contradictory text. Choice scoring is stricter about format. These are the official deterministic semantics, not a factuality guarantee.
- In a post-run diagnostic on the learned arm's 174 phrase-scored questions, reference phrases appear in selected memory on 86, but 49 of those responses still miss credit. Presence can be incidental; this is not evidence recall or proof of causal support. Context selection is not the only unresolved issue.
- **128 judge-dependent text questions and 29 image questions are excluded**, with every ID and reason in the registration. No gotcha, premise-awareness or image success is inferred from the text score.
- The index occupies 361,992,192 bytes (about 345 MiB) and built in 16.292 seconds locally. RAM/VRAM peaks, energy, independent retention and target edge-device performance remain unqualified. No deployment promotion trial was attempted.

Next priorities are better source selection with UI/container relationships, answer-format handling, externally checked answer support, abstention calibration and independent workflow retention. These results do not establish universal superiority, autonomous weight improvement or domain AGI.

## Reproduction

### Search latency without changing returned context

The initial native-rank optimization preserved all 882 ordered query/channel results and reduced median lookup time, but worsened p95 latency. The retained [first profile](search/summary.json) identifies the regression on sparse workflow-path searches.

A fixed refinement keeps the original query plan for path views and uses native rank iteration plus complete boundary-tie sorting for state/change views. In a new, hash-ordered paired run over all 882 lookups, **all returned fragments remain identical**. Median lookup time falls **279.39 to 98.80 ms**, p95 **357.41 to 311.84 ms**, and summed lookup time **246.67 to 128.95 seconds**. See the [refined profile](search-selective/summary.json) and its preregistered [selection rule](search-selective/selection.json).

These are warmed local SQLite lookups on this index, with one measurement per query/channel and no whole-request or production scalability claim. The refinement was selected after observing the first profile's failure, so this is engineering remeasurement on the same workload, not independent workload generalization. No model answer or score was used to choose the query plan. `experiments/lme_memory_optimized.py` exposes the separately pinned variant; original generation results remain attached to the original runner.

Use the [local runbook](../../docs/local-trajectory-memory.md). Offline replay makes no model call:

```sh
python experiments/verify_lme_memory.py
python experiments/test_lme_memory.py -v
python experiments/verify_lme_search.py
```

Compressed records contain all 882 measured outputs, 222 learned selections, short native references and feature vectors. Replay recalculates every score and aggregate, refits the policy and checks its selections. Raw trajectories, prepared prompts, screenshots and the index remain outside Git; fetch the licensed source data to regenerate them.

Dataset revision: `f152293e235517d504809563c833d7190b8c713b`. Official code revision: `2cc8c540bdb87fe6761629b585e727e1c4704520`. The unchanged upstream scorer and Apache-2.0 license are in `upstream/`; only reviewed pure functions are executed. See [third-party notices](../../docs/THIRD_PARTY_NOTICES.md). Original project code is MIT-licensed, copyright Prashant Jagtap.

# Structural memory: two local readers

By Prashant Jagtap. Development experiment on LongMemEval-V2, 27 September 2026.

Preserving recorded structure gives small aggregate gains in this run, but introduces regressions and increases model-token use. It does not qualify an automatic local update. The evidence includes all 288 measured predictions, references, source bindings, timings and native scores.

## Paired results

Both arms answer the same 72 questions per reader, with full prompts capped at 6,144 tokens and outputs at 256. The raw prompts exactly match the earlier experiment's state prompts for these 72 IDs. Both arms are freshly generated with an 8,192-token context capacity. These scores should not be substituted for the earlier 222-question results.

| Reader | Memory | Native credit | Total model tokens | Median generation |
|---|---|---:|---:|---:|
| Qwen2.5 1.5B | Raw chunks | 13/72 | 404,534 | 0.733 s |
| Qwen2.5 1.5B | Structural views | 14/72 | 428,159 | 0.741 s |
| Qwen2.5-Coder 7B | Raw chunks | 10/72 | 403,834 | 2.018 s |
| Qwen2.5-Coder 7B | Structural views | 15/72 | 427,891 | 2.171 s |

The 1.5B treatment gains seven answers and loses six. The 7B treatment gains eight and loses three. Total model tokens increase **5.84%** and **5.96%**, respectively. Summed measured generation time increases **2.62%** and **10.57%**. There are no request/tokenizer-mismatch errors. One raw 1.5B answer reaches the output cap and receives zero credit.

![Paired native credit, tokens and generation time](../../docs/assets/lme-structure-v1.png)

The 7B reader emits exact `UNKNOWN` on 51 raw and 44 structural cases; the 1.5B reader does so on none. This is a behavioral difference, not proof that either reader is calibrated. The larger reader's training specialization also differs. Parameter count alone is not isolated by this comparison.

All six domain/ability cells are retained in [failure-review.json](failure-review.json). Aggregate scores hide regressions: for example, 1.5B web/static credit falls from 7/12 to 4/12. Every candidate remains inactive.

## What was changed

The new adapter uses indentation to retain recorded ancestors and contiguous subtrees or sibling blocks. It omits volatile instance handles and combines equal projected bodies at the same location within one episode, retaining every source occurrence. It does not interpret recorded actions or turn observations into verified facts.

The source corpus contains 200 episodes and 5,095 observations. The new index contains 21,899 views, representing 78,249 occurrences. An independent source-line reconstruction verifies every occurrence and coverage of all **3,072,962 nonblank source lines** across the full view collection. No episode needs the explicit raw fallback. The index is 153,948,160 bytes and took 11.32 seconds to build in this run. This is not an index-size comparison with a dedicated raw-only database: the earlier database also contains change/path views.

The complete collection's source coverage does not imply that the selected prompt contains enough evidence. A 32-byte identity still references external content; these views do not compress the entire environment into 32 bytes.

## Retained failures and next hypotheses

- Both readers regress on the Low Stock Report column question and the Order Count Report control-label question. Selected structural memory can mix several pages. Page ownership and question-specific scope need stronger handling. Multiple titles are an observation, not a demonstrated cause of the wrong answer.
- A 1.5B answer lists the right Bulk Actions Log columns but uses `boxed{...}` without a backslash. The native scorer rejects it. We retain that result rather than repair outputs after seeing references.
- A raw 1.5B answer gets native credit while adding an extra mandatory field. Native phrase matching can reward incorrect extra content, so native credit is not a factuality or safety guarantee.
- Static hierarchy does not supply a verified procedure or action-to-result relationship. Procedure and dynamic-environment cases remain weak.
- Removing repeated observations and preserving ancestry are combined in this treatment. Separate ablations are needed before attributing any gain to one mechanism.

The next experiments should test page-scoped retrieval, explicit control/label and table-header relations, evidence adequacy and action-transition bindings. They need fresh evaluation material and older-task retention. More adapter training on these inspected questions would not establish generalization.

## Method and limits

The six domain/ability cells each contribute twelve questions, chosen by a fixed SHA-256 order before the run. All 72 are already development material from the earlier benchmark study; histories are shared. This is neither an independent deployment test nor the full LongMemEval-V2 suite. Image and judge-required cases remain excluded.

Both arms use the same lexical query terms, reader prompt, token ceiling, 32-candidate inspection limit and 16-view limit. The structural index includes the recorded goal for search, but the reader sees the structural source view. This is a combined representation/retrieval treatment. Canonical source identity and public-tenant scope are checked; semantic sufficiency is not certified.

Each reader runs separately on the local RTX 5080 Laptop GPU. Model digests, tokenizer digest, source hashes, options and sample IDs are in [registration.json](registration.json). Arm order is shuffled within each question; model order is fixed. Two warmups are excluded from scored calls. No paid API, remote judge, remote model, weight update or recorded browser action is executed.

Generation timing excludes CPU retrieval/packing, loading and index construction. [failure-review.json](failure-review.json) gives separate stage medians and a per-question stage sum; that sum is not a live end-to-end timing. Some CPU source verification/tests overlapped the first reader's run, so these are development workstation timings, not isolated microbenchmarks. Peak RAM, deployment concurrency, energy, target-edge behavior and update amortization remain unmeasured.

## Reproduce and inspect

- [Adapter, complete example and local runbook](../../docs/structured-local-memory.md)
- [All native predictions and short references](records.json.gz)
- [Source-line reconstruction report](projection-check.json)
- [Aggregate native scores](summary.json)
- [Post-run failure review](failure-review.json)

```powershell
python experiments/verify_lme_structure.py
python experiments/test_lme_structure.py -v
python experiments/test_lme_structure_evidence.py -v
python examples/structured_local_memory.py
```

Offline replay recomputes all 288 native scores and aggregate results. Eight new core tests cover source/structure/bounds/ACL behavior; the full local core suite passes 369 tests without skips. Three experiment boundary checks and five evidence-tampering checks also pass. Unit tests do not establish model robustness.

Raw histories, prompts and the SQLite index stay outside Git. Short native references derive from the Apache-2.0 [LongMemEval-V2 dataset](https://github.com/xiaowu0162/LongMemEval-V2); the unchanged scorer and its license are in [the earlier evidence directory](../lme-memory-v1/upstream). See [third-party notices](../../docs/THIRD_PARTY_NOTICES.md). Original code and figures are copyright Prashant Jagtap under the repository MIT License.

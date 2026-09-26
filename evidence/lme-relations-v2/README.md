# Page ownership and recorded relations: two local readers

By Prashant Jagtap. LongMemEval-V2 development experiment, 27 September 2026.

**Page filtering gives a small mixed result; the relation-packet treatment fails this comparison. Neither qualifies for automatic activation.** All 432 predictions, native references, measured request times and regressions are retained. No model weights were trained in this experiment.

## Results

The same 72 previously inspected questions are answered in each arm. Native credit uses the unchanged upstream deterministic scorer. Model tokens include input and output. Request time includes live retrieval, context compilation and local generation.

| Reader | Context | Native credit | Total model tokens | Median request | p95 request |
|---|---|---:|---:|---:|---:|
| Qwen2.5 1.5B | Structural baseline | 14/72 | 428,137 | 0.965 s | 1.132 s |
| Qwen2.5 1.5B | Page filter | 17/72 | 430,072 | 1.055 s | 1.357 s |
| Qwen2.5 1.5B | Page + relation packets | 13/72 | 424,991 | 1.701 s | 2.120 s |
| Qwen2.5-Coder 7B | Structural baseline | 16/72 | 427,901 | 2.336 s | 2.556 s |
| Qwen2.5-Coder 7B | Page filter | 16/72 | 429,838 | 2.457 s | 2.756 s |
| Qwen2.5-Coder 7B | Page + relation packets | 9/72 | 424,299 | 3.083 s | 3.711 s |

![Quality, model tokens and actual request latency](../../docs/assets/lme-relations-v2.png)

Page filtering gains five answers and loses two for 1.5B; it gains four and loses four for 7B. Relation packets gain four and lose five for 1.5B, and gain four and lose eleven for 7B. Compared with the structural baseline, relation packets reduce total model tokens by only **0.73% and 0.84%**, while summed request time rises **76.27% and 36.04%**. That is not an efficiency improvement.

There are no request/token-count errors. One 1.5B relation answer reaches the output cap and receives zero credit. The 7B reader returns exact `UNKNOWN` in 42, 43 and 48 cases respectively. The corresponding 1.5B counts are 0, 0 and 1; this behavior does not establish calibration. All six domain/ability cells and every paired gain/regression ID are in [failure-review.json](failure-review.json).

The 7B baseline received 15/72 in the preceding structural study and 16/72 in this fresh run, despite the same baseline prompts and configuration. Treat observed one-answer differences cautiously: a fixed seed and temperature do not establish bitwise repeatability across local GPU runs. All comparisons above use contemporaneous arms from this run.

## Check the metric as well as the model

Native phrase matching can accept an answer containing the required phrases plus incorrect extra content. A **post-hoc secondary diagnostic** requires exact normalized phrase sets or ordered lists within the native extracted answer. Choice questions retain native scoring. This does not alter official credit, inspect contradictions outside the final extracted box or prove factuality; valid paraphrases can fail it.

| Reader | Structural baseline | Page filter | Page + relation packets |
|---|---:|---:|---:|
| 1.5B: strict reference agreement | 11/72 | 15/72 | 12/72 |
| 7B: strict reference agreement | 15/72 | 15/72 | 9/72 |

Disagreement IDs are retained for inspection. Neither native credit nor this exact-match diagnostic is adequate as the sole verification signal for a production learner.

## What was tested

1. **Structural baseline:** the preceding study's source-bound ancestry views. All 72 prepared baseline prompts must match exactly.
2. **Page filter:** query/title affinity using inverse document frequency over authorized page paths; then the same structural representation and packing limits.
3. **Page + relation packets:** the same page filter, recorded ordered cells/options/tabs/menu items and named elements, local container ancestry, duplicate handling and grouping by page.

The third arm changes representation, duplicate handling and packing limits together. It does not isolate a single relation mechanism. Structural arms inspect at most 32 views and select 16; relation packing inspects 160 and selects 96. All arms share the 6,144-token complete-prompt limit, 256-token output cap and 8,192-token model context capacity. Page scores are heuristic affinities, not sufficiency probabilities.

The local source audit independently reconstructs retained text, source order, ancestor and page ownership for **1,065,198 occurrences in 406,532 packets across 200 episodes**. It finds no member crossing a nested page boundary. The representation is lossy; this audit does **not** certify complete source coverage, visual layout, causal effects, truth or sufficient selected evidence. The combined structural/relation index is **759,484,416 bytes**, built in **29.99 seconds**. It is not a relation-only size measurement or an edge-device memory result.

The [superseded preparation](../lme-relations-preparation-v1/README.md) omitted local container ownership. Its exact source bytes and metadata were archived before correction. It made zero reader calls; it is not a hidden failed model trial.

## Where this failed and what follows

- Both page-filtered treatments recover the Low Stock Report and Order Count Report questions that both readers missed in the preceding structural comparison. Other answers regress, so those repairs do not establish a general improvement.
- Source-level relation integrity is insufficient. Flattened packets can omit context needed to interpret a control, procedure or change. The 7B relation arm receives zero credit in web/procedure and enterprise/dynamic cells. This is a measured failure; the exact causal contribution of representation versus retrieval/packing remains unresolved.
- Many small packets increase CPU work. Median retrieval grows from about 33 ms to about 400 ms, and median packing from about 175 ms to about 585 ms. Fewer prompt tokens do not imply a faster workflow.
- Lexical title affinity cannot resolve every ambiguous page name, cross-page procedure or temporal transition. It cannot establish authorization or semantic sufficiency.
- These questions have already guided development. Further fitting on them cannot establish generalization. A next candidate needs distinct source/task clusters and older-skill retention, along with a frozen quality/resource plan.

The next implementation priorities are selective page-index access with exact returned-context parity, retention of larger source blocks around ambiguous relations, explicit action-to-observation bindings, and task-specific evidence-adequacy checks. Each needs a separate ablation. Do not enable relation packets as a default based on their structural tests.

## Local operation and reproduction

Use the [API example and RTX runbook](../../docs/page-relational-memory.md). Both readers are separately installed Q4_K_M models on the local RTX 5080 Laptop. Digests, tokenizer, source hashes, options and IDs are in [registration.json](registration.json). CPU selection runs locally; GPU inference runs one reader at a time. No recorded browser action, cloud model, paid provider or remote judge is invoked.

For every actual request the runner recompiles context, checks equality with the registered prompt and invokes the local reader. The enclosing timer includes those stages. Warm sequential timings exclude index building, loading and two warmups; they are not a production concurrency benchmark. Heavy source verification ran after inference. Peak RAM, energy, update amortization and physical target-edge behavior remain unmeasured.

```powershell
python experiments/verify_lme_relations.py
python experiments/test_lme_relations.py -v
python experiments/test_lme_relation_evidence.py -v
python examples/page_relational_memory.py
```

The [local learning registry](../../docs/local-domain-learning.md) can reject candidates, enforce separate adaptation/retention cohorts and support rollback. This study proposes no activation and no weight update. Locally self-improving models remain the objective; retaining the current version when a candidate regresses is part of that design.

The [compressed records](records.json.gz) contain outputs, short references, scope/source metadata and timings. Full histories, questions, prompts and the SQLite index remain outside Git. Offline replay recomputes all 432 native scores, secondary diagnostics and aggregates; hashes are integrity bindings, not independent attestation of measurements. Raw-data source auditing requires the pinned dataset and index.

[Engineering checks](engineering.json) record 379 passing local core tests without skips, five experiment boundary tests, seven evidence/metric tests and Linux replay. The source package replays both the measured comparison and retained preparation; the wheel contains the new adapter. Ruff, Bandit and secret scans pass, including expanded compressed records and archived sources. Secret-scanner exclusions are restricted to two exact public tokenizer checksums at their known paths; 135 positive/negative controls pass. These checks do not establish immunity from attacks or production qualification.

Short references derive from the Apache-2.0 [LongMemEval-V2 dataset](https://github.com/xiaowu0162/LongMemEval-V2); the unchanged scorer and license remain in [the earlier evidence](../lme-memory-v1/upstream). See [third-party notices](../../docs/THIRD_PARTY_NOTICES.md). Original code and figures are copyright Prashant Jagtap under the repository MIT License.

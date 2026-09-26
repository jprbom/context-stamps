# RULER-v1: local development result

By Prashant Jagtap · 27 September 2026

A small reader benefited from explicit context operations on this development set. The experiment does **not** establish an advantage over direct tools, reliable semantic sufficiency or autonomous model improvement.

The same installed Qwen2.5 1.5B Q4_K_M reader completed 208 local RTX calls: 52 inputs, all 13 task types, two nominal lengths (4K/16K), two examples per cell and four treatments. No weight update or paid call was made. Every generation is retained; zero transport/protocol errors and two output truncations occurred.

| Treatment | Mean native score | Complete native credit | Input tokens | Output tokens | Sum of request seconds |
|---|---:|---:|---:|---:|---:|
| Full context | 0.7500 | 34/52 | 507,356 | 1,353 | 59.93 |
| BM25 | 0.7689 | 36/52 | 199,359 | 1,336 | 38.70 |
| Direct operations + reader | 0.8654 | 45/52 | 102,888 | 1,111 | 26.23 |
| Runtime + reader | 0.8763 | 44/52 | 121,192 | 1,145 | 29.29 |

The zero-model-call deterministic control receives complete credit on **42/52**, abstaining on the other ten. It correctly handles every input for which its explicit parser/counting contract applies. It is the strongest cost control in that supported subset.

![Quality and input cost across the four reader treatments](../../docs/assets/ruler-development-v1.png)

The runtime uses 76.1% fewer input tokens than full context. It adds 18,304 input tokens relative to direct operations because it carries versioned source/provenance material. Its higher fractional native mean does not mean more complete answers: direct operations give 45 complete-credit outputs versus 44 for the runtime. Relative to full context, the runtime improves native score on 11 inputs and lowers it on none in this development sample. Relative to direct operations, it improves one and lowers two.

Request seconds include measured CPU preparation plus the local model call. Warm-up is separate. The server remains resident and prompt-cache effects are not disabled; repeated identical prompts can benefit later treatments. Treatment order is shuffled within each input, but a single small trial cannot remove this confound. These observed times are not a demonstrated production speedup. No streamed TTFT, energy, sustained throughput or target edge-device test is reported.

## What failed

- **QA remains weak.** The runtime receives credit on only two of eight QA inputs. Six failures are three repeated questions across the two lengths. Incorrect answers include a wrong century range, an incorrect yes/no answer and an invented government role. Full source context is retained because no semantic sufficiency certificate exists; more context alone does not repair these reader failures.
- **Generative copying can damage exact results.** The runtime reader changes `subconscious` to `sub-conscious` in one word-list answer, giving 9/10 native recall. It omits the most frequent coded word in another, giving 2/3 recall. Both deterministic operations had the correct answer before generation. Verified direct output should be a first-class route for these bounded tasks.
- **The conservative key parser costs tokens.** An unhandled multiword distractor key (`ad hoc-incidence`) triggers full-context fallback in two inputs. The failure is retained rather than silently dropping that statement. Broader key grammars require new tests and a fresh evaluation, not replacement of this result.
- **Native scoring is permissive.** A BM25 multivalue output repeats a number until the output cap; a full-context variable-tracking output repeats assignments until the cap. Both still receive complete native substring credit. They remain in the denominator. Complete credit is not a deployment-quality certificate.
- **Runtime overhead is real.** Direct operations plus the reader use fewer tokens and achieve more complete-credit answers. This experiment does not test policy changes, revocation, repeated-result reuse or local adaptation; it cannot establish that runtime overhead pays back in those workflows.

## Boundaries and next acceptance gates

This is a benchmark-aware development adapter, using explicit literal matching, ordered assignments and integer counts. It is not a learned context selector, a new attention architecture or a universal compression method. The 32-byte receipt resolves to external evidence; the model consumes the actual materialized token totals above.

The sample is small. QA questions repeat across lengths; public data may have appeared in model pretraining. No held-out significance or universal improvement claim is made. After this review, these inputs are regression cases. Further model or policy changes require new evaluation cases, a retention set and measurements on the intended device.

The next useful gates are stricter output validation; direct verified output for exact tasks; fresh projects and task families; explicit cold/warm cache latency controls; and prospective local policy updates with rollback. A candidate that only learns this benchmark grammar is insufficient for the domain-SLM objective.

## Evidence and reproduction

- [Protocol and local commands](../../docs/ruler-local-development.md)
- `registration.json`: frozen model, input, tokenizer, source and protocol identities.
- `data-plan.json` / `data-inventory.json`: all 26 task/length cells and native generator/corpus hashes.
- `records.json.gz`: 208 short predictions, scoring references, deterministic control outputs and per-input metadata.
- `summary.json`: all cell scores and aggregate outcomes, including errors and truncations.
- `manifest.json`: exported file identities and original local generation-log identity.

Run `python experiments/verify_ruler_development.py` for offline metric replay. It checks every paired treatment and recomputes native scores and resource totals from the exported predictions. Reproducing context selection requires the separately downloaded corpora and the protocol scripts; the public archive intentionally excludes long prompts and Ollama context-token IDs.

NVIDIA RULER/NeMo code retains Apache-2.0 attribution. QA reference snippets retain SQuAD/HotpotQA CC-BY-SA-4.0 attribution; see the [data notices](../../docs/ruler-local-development.md#data-and-rights). Long essay text is not redistributed. This evidence does not activate a local model or policy.

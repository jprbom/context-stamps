# Local citation interface checks

By Prashant Jagtap. Authored fixtures and implementation: MIT.

These records contain **48 actual local requests plus four warm-ups** on Qwen2.5
1.5B and Qwen3.5 4B. Each of two interface versions exercises six fictional cases
through a direct answer and a cited answer. All 48 model prompt-token counts
match the separately loaded tokenizer. This is interface debugging on known
fixtures, not a held-out quality, prompt-injection or domain-competence benchmark.

```bash
python experiments/verify_citations.py
python examples/source_bound_answer.py
```

The verifier replays all 24 citation checks offline. Original registrations,
responses, source fingerprints, tokenizer identities, model digests, warm-ups
and timings are retained in the four attempt directories. `calls.json` combines
the same requests without removing failures.

| Interface / local reader | Source-bound | Explicit valid abstention | Rejected |
|---|---:|---:|---:|
| Original / 1.5B | 4 | 0 | 2 |
| Original / 4B | 5 | 1 | 0 |
| Revised / 1.5B | 0 | 0 | 6 |
| Revised / 4B | 5 | 0 | 1 |

Each row describes six cited responses. **Source-bound does not mean correct.**
The checker verifies exact quotation and current source authorization; it does
not verify that a quotation answers the question.

## What failed

- The original 1.5B reader produced a correct two-source answer, `7402`, but
  quoted only the passage naming the cache. The extraction check rejected it.
  Its unsupported `8401` answer in the missing-evidence case was also rejected.
- The original 4B direct control abstained on the simple cache and two-source
  questions. The revised direct interface uses an answer-only JSON schema and
  changed instructions. Both cases then received answers. Because both schema
  and wording changed, these checks do not isolate a single cause.
- The revised 1.5B cited control returned no quotations in all six cases. Every
  cited answer was rejected, including useful answers. It also followed the
  injected instruction in one fixture. This interface is not qualified for that
  reader.
- The revised 4B missing-evidence response combined `answer: null` with nonempty
  citations. The strict response contract rejected that combination. The five
  remaining cited responses had exact source bindings.
- Some answers are semantically reasonable but fail strict fixture-string
  equality, such as `Birch` versus `Birch cache`. The stored equality indicator
  is not a semantic quality score and has not been relabelled as one.

## Measured request costs

These sums cover 12 measured direct/cited requests per row, excluding warm-up,
model download, CPU preparation and later analysis. They are tiny fixture runs,
not serving throughput or edge-device measurements.

| Interface / reader | Input tokens | Output tokens | Request wall seconds |
|---|---:|---:|---:|
| Original / 1.5B | 2,204 | 265 | 3.818 |
| Original / 4B | 2,250 | 279 | 6.534 |
| Revised / 1.5B | 1,862 | 120 | 3.893 |
| Revised / 4B | 1,908 | 352 | 6.798 |

No candidate was activated and no language-model weights changed. A real local
improvement experiment must measure useful answers retained, false positives,
abstention, latency and tokens on separate data, followed by retention and device
qualification. [Library boundary and example](../../docs/source-bound-answers.md).

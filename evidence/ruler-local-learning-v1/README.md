# Local routing-policy learning: retained development evidence

By Prashant Jagtap

The verified exact path removes model copying errors on the supported RULER task grammar. A four-cell CPU policy learns to use that path for lookup, variable dependencies and counts, and retains the local reader for document QA. **It chooses the same actions as the fixed verified rule on all 208 evaluation inputs. Learning does not establish an advantage over that rule.** The candidate remains inactive.

## Evaluation results

All arms use the same public input, local Qwen2.5 1.5B Q4_K_M reader and runtime. The baseline already has contract-aware context selection; this is not a new full-context comparison.

| Measured quantity, 208 inputs | Runtime + reader | Learned routing | Fixed verified rule |
|---|---:|---:|---:|
| Native substring mean | 0.942788 | 0.947115 | 0.947115 |
| Complete native credit | 193/208 | 197/208 | 197/208 |
| Custom strict output check | 143/208 | 188/208 | 187/208 |
| Independently checked exact results | 0 | 176 | 176 |
| Reader calls | 208 | 32 | 32 |
| Model input tokens | 408,813 | 320,080 | 320,080 |
| Model output tokens | 4,691 | 343 | 368 |
| Summed pipeline wall time, seconds | 88.543 | 15.274 | 22.731 |
| Transport/protocol errors | 0 | 0 | 0 |
| Truncated outputs | 0 | 0 | 0 |

The learned route uses 22.5% fewer total model tokens and makes 84.6% fewer reader calls in this sample. Its 45 strict-output gains and zero strict-output regressions are all in the exact-task subset: 131/176 becomes 176/176. Native credit improves on four inputs, with no native regressions. Much of the stricter gain removes extra prose or repeated answer atoms, rather than demonstrating newly acquired reasoning.

Document QA remains unresolved: the learned route and baseline each have 21/32 native-complete and 12/32 strict outputs. The fixed rule has 21/32 and 11/32. Learned/fixed routes and reader prompts are identical, but three reader generations differ; one explains the strict-score difference. This is generation variation, not a learned-policy advantage. See [every candidate failure and differing output](failure-review.json).

Wall times include input parsing, runtime preparation, exact verification and local inference. They exclude server warmup, offline reference scoring and policy fitting. The reader is warm, native caching is enabled and arms run in shuffled order in one trial. The sizeable learned/fixed timing difference despite identical actions is a reason **not** to interpret these sums as a stable causal speedup, a cold-service SLA or an edge-device benchmark. Peak whole-pipeline RAM, energy and throughput remain unmeasured.

## Training and activation

- 312 fresh training inputs, 624 paired action outcomes, 312 actual reader calls.
- Training reader: 271/312 native-complete; 218/312 strict. Exact operation: 264/312, with 48 document-QA abstentions.
- CPU fitting: 0.000369 seconds in this run. Collecting the reader training outcomes takes 117.857 summed pipeline seconds; the exact-action control takes another 0.355 seconds. The tiny fit time is not the total training cost.
- Training consumes 292,396 model tokens. The activation plan charges them over a declared 10,000-task horizon. Model-token cost does not represent CPU energy, elapsed time or money.
- The data-only [policy artifact](policy-data.json) is 314 bytes as serialized; it contains four cell/action choices plus an environment binding. It is an inactive research artifact, not a universal routing model.
- Evaluation has 104 fresh nominal-4K retention-diagnostic inputs and 104 fresh nominal-16K adaptation inputs. These are the same task families, not broad retention across older domains.

The [registered trial](registered-trial.json) freezes cohorts and policy before holdout generation. The [promotion result](promotion.json) rejects missing whole-pipeline RAM measurements and keeps the baseline active. No fabricated resource value is substituted. Source-cluster independence is also unqualified: a [post-run audit](source-overlap.json) finds shared QA documents despite no repeated full inputs or QA questions. Even if IID sampling were justified, 104 zero-new-failure observations give a 5.274% upper bound under this round's allocation, above the default 1% limit.

New registries now require separate adaptation and retention checks. The protected-cohort tests include a pooled evaluation that would pass while its older-skill subset fails; the new default rejects that candidate. Explicit legacy mode preserves earlier pooled evidence and cannot downgrade an existing protected registry.

## Reproduction and scope

Run `python experiments/verify_ruler_learning.py` for offline replay. It recomputes the fitted policy, every retained score, token/training totals, task/length cells and protected rejection. The [RTX runbook](../../docs/ruler-local-learning.md) regenerates inputs and executes the local model comparison. Raw licensed corpus text, private traces, model weights and context-token IDs are excluded from publication.

There are 1,248 retained action records: 624 training and 624 evaluation. These involve 584 actual reader calls, plus one warmup. Exact actions use CPU and consume no model tokens; they still scan and verify the full source. The 32-byte receipt references external source state; it does not contain or reconstruct all context.

This is benchmark-specific local policy learning and verification engineering. There is no base-model weight update, internal-attention improvement, autonomous domain intelligence, universal superiority or target-edge qualification. The next test needs project/source holdouts, real workflow retention, complete device costs and a choice where a learned policy can improve on fixed verified routing.

RULER/NeMo code and benchmark conventions are attributed in [third-party notices](../../docs/THIRD_PARTY_NOTICES.md). SQuAD/HotpotQA-derived references retain their source terms; the project license does not replace dataset rights. Original project code is MIT-licensed, copyright Prashant Jagtap.

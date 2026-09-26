# Local policy learning with verified exact results

By Prashant Jagtap

This experiment tests a practical question: when the local runtime can independently check an exact result, should it return that result or ask a small language model to reproduce it? It fits a CPU routing policy from actual local model outcomes and evaluates that frozen policy on new inputs. The Qwen reader weights remain unchanged.

The [retained results](../evidence/ruler-local-learning-v1/README.md) include the learned policy, every short prediction, both scoring methods, token counts, timings, training cost and the rejected activation. The [earlier RULER trial](ruler-local-development.md) remains unchanged.

## What changes in this experiment

The benchmark adapter handles literal key lookup, variable dependencies and exact frequency counts. The result producer and checker use different implementations: regular-expression extraction versus literal sentence parsing, forward value propagation versus dependency-chain traversal, and sorted counts versus independent integer tallies. Both read the same full public input; neither reads answer keys. These are checks of the declared benchmark grammar, not independent semantic authorities.

The runtime binds the complete source, question, tool and verifier revisions to the computation receipt. The checker can reject omitted or added answers, malformed source statements, missing dependencies, repeated values and count-boundary ties. Unsupported document QA keeps the reader path; its answer is not marked as independently verified. Exact results are returned as bounded JSON lists. They are never sent through the reader merely to restate them.

The comparison has three arms:

1. **Runtime + reader:** existing contract-aware context compilation followed by the local Qwen reader.
2. **Learned routing:** the same runtime, with a four-cell policy choosing the verified exact path or reader. A rejected exact result falls back to the reader.
3. **Fixed verified rule:** use the exact path whenever its input contract is supported; otherwise use the reader. This is a strong low-complexity control.

Training has two matched observations per input: exact operation and reader. Unsupported exact operations count as failed/abstained observations. The existing Beta-smoothed CPU fitter uses an error penalty of 20, cost cap of 31,000 model tokens and minimum support of 20 tasks per cell. The four cells are `exact`, `dependency`, `statistic` and `full`, derived from the public input parser. No answer label defines a cell. Smoothing proposes a policy; it is not a calibrated correctness probability.

## Data and protocol

- Training: 24 inputs per each of 13 tasks at nominal 4K, seed 73, QA indices 200–223: 312 inputs.
- Retention diagnostic: eight new inputs per task at nominal 4K, seed 89, QA indices 1000–1007: 104 inputs.
- Adaptation diagnostic: eight new inputs per task at nominal 16K, seed 10062, QA indices 2000–2007: 104 inputs.
- Qwen2.5 1.5B Q4_K_M, the same pinned local model, tokenizer and raw ChatML settings as the previous trial; temperature 0, seed 71, 32K context, 256 generated-token cap.
- One GPU inference workload. Data preparation, exact operations, policy fitting and scoring run on CPU. Calls use loopback Ollama only.
- The complete code/model/data plan is frozen before generation. Training labels are opened after the complete training log closes. The fitted policy and two cohorts are registered before any holdout generation. Holdout labels are opened only after its generation log closes.
- Arm order is shuffled deterministically per input. Ollama is warm and native caching remains enabled. These timings do not isolate cache effects or establish cold-service/edge latency.

The generator's native reserve and task formatting are retained. The nominal 4K common-words setting follows the native difficulty branch after subtracting the token reserve. This is a custom development protocol, not a leaderboard submission or a comparison against frontier-model rankings.

Exact full-input hashes do not overlap across partitions; QA question text also does not overlap. A later source audit still finds shared QA document text: three documents across train/4K, 19 across train/16K and 24 across the two evaluation cohorts. Synthetic backgrounds and task templates are also shared. This is not independent source-family generalization or broad older-skill retention.

## Scoring and activation

Native RULER substring credit is reported unchanged. A second, stricter output diagnostic rejects truncation, extra answer atoms, repeated answers and unexplained prose on exact tasks; QA requires equality to a normalized reference alias. This custom score can penalize a valid paraphrase. Neither score certifies general factual truth or safe autonomous actions.

New learning registries require separate adaptation and retention cohorts. Each has its own new-failure, absolute-failure and deadline checks; one pooled cost check also charges measured training model tokens over the declared 10,000-task deployment horizon. The policy artifact itself runs on CPU without a neural runtime.

This experiment cannot activate a candidate: trustworthy whole-pipeline peak RAM and source-cluster independence are not established. Unknown RAM remains `None`; the registry rejects the trial and preserves the baseline digest. Even under hypothetical IID sampling, 104 observations with zero new failures cannot meet the default 1% limit at this round's allocated confidence. Do not weaken that gate or restart the registry merely to get a passing result.

## Reproduce on the local RTX setup

First follow the pinned generator, corpus and tokenizer setup in [the previous runbook](ruler-local-development.md). Use separate output directories outside the publishing repository. Reuse the downloaded licensed assets and local reader; no second model download is needed.

```powershell
python experiments/ruler_native.py prepare --source ..\ruler-generator-source-v1 --assets ..\ruler-assets-v1 --tokenizer ..\qwen25-15b-train-base --output ..\ruler-learning-train-v1 --count 24 --seed 73 --lengths 4096 --qa-offset 200

python experiments/ruler_native.py prepare --source ..\ruler-generator-source-v1 --assets ..\ruler-assets-v1 --tokenizer ..\qwen25-15b-train-base --output ..\ruler-learning-holdout-v1 --count 8 --seed 89 --lengths 4096 16384 --qa-offset 1000 --seed-stride 9973 --qa-offset-stride 1000

python experiments/ruler_local_learning.py --train ..\ruler-learning-train-v1 --holdout ..\ruler-learning-holdout-v1 --tokenizer ..\qwen25-15b-train-base --output ..\ruler-local-learning-v1
```

Use the experiment environment with the dependencies specified in `experiments/ruler-requirements.txt`. The recorded preparation source is archived; current code moves the optional YAML import inside preparation but leaves generator arguments unchanged. Exact artifact hashes may differ after source/environment changes. Keep partial runs and errors; output directories are never overwritten or silently resumed.

The public replay needs only the installed core and standard library:

```bash
python experiments/test_ruler_context.py -v
python experiments/test_ruler_local_learning.py -v
python experiments/test_ruler_learning_evidence.py -v
python experiments/verify_ruler_learning.py
python -m unittest discover -s tests -p test_learning_cohorts.py -v
```

It recomputes the policy, output scores, task/length totals, training cost and the protected rejection. It cannot reconstruct source verification from short predictions alone; that requires the locally regenerated public inputs. Long corpus text, private traces, model weights and Ollama context-token IDs are excluded from the evidence export.

## Next experiment

Evaluate real operational tasks with exact verifiers, conflicting versions, changing permissions and recovery after failed tools. Hold out source projects and older task families, not just new random seeds. Compare verified fixed routing with a learned policy only where input features create a meaningful choice. Add complete device resource instrumentation before activation, then test the same frozen policy on a target edge device. A learned table matching a simple rule is useful implementation evidence, but does not establish a new learning algorithm or autonomous domain intelligence.

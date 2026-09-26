# Local code adapter: candidate only

By Prashant Jagtap

This is an experimental rank-eight LoRA adapter for Qwen/Qwen2.5-1.5B-Instruct
revision `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`. It is **inactive** and is
not a standalone model. It has not qualified for domain deployment, autonomous
self-improvement, edge inference or a Context Stamps performance claim.

| Item | Recorded value |
|---|---:|
| Eligible MBPP training examples | 368 |
| Epochs / optimizer steps | 2 / 184 |
| Trainable parameters | 1,089,536 |
| Adapter size | 4,372,840 bytes (4.17 MiB) |
| Training-loop time | 65.42 seconds |
| Peak Torch CUDA allocation | 7,879,181,312 bytes (7.34 GiB) |
| Base parameter fingerprint | Unchanged |
| Model quality | Not established by this training record |
| Activation | Disabled |

The device is an RTX 5080 Laptop GPU. Loading, data preparation and evaluation
are separate costs. Peak allocation excludes driver/system memory; energy,
sustained edge throughput and CPU/NPU inference are unmeasured.

Five exact reserved-split overlaps and one undefined-`Node` reference are
excluded from MBPP's published training split. Retained references passed their
three native assertions, and empty controls failed. Exact docstring-normalized
full-AST comparison against HumanEval+ found no additional overlaps. This does
not detect semantic equivalents or unknown pretraining contamination. Passing
three assertions is not a proof of correctness.

The fixed recipe uses BF16, query/value adapters, rank 8, alpha 16, dropout 0,
AdamW learning rate 0.0001, batch size 4, two epochs and a 1,024-token limit.
Assistant code/end-marker tokens receive the loss; prompt and padding tokens
do not. No held-out result selects a checkpoint. Training starts from the base,
not the earlier authored-workflow adapter.

All training rows, token counts, step losses, gradient norms, source hashes,
base inventory and frozen-base fingerprint are retained. `trained.json` records
training only; `benchmark_results: null` is intentional. Model comparisons are
separate runs and must retain denominators, regressions and resource measures.

Follow [the local guide](../../docs/verified-code-learning.md). Replay evidence
without loading weights:

```console
python experiments/verify_code_training_data.py
python experiments/verify_code_adapter.py
```

The coding comparison uses the same BF16 base/backend with the adapter enabled
and disabled. A generic SFT gain, if found, still needs a separate ablation of
Context Stamps memory, context selection and learning policy. No universal
superiority follows from this checkpoint.

The adapter is distributed under Apache-2.0; retain `APACHE-2.0.txt` and
attribution to Prashant Jagtap. The Qwen base retains its Apache-2.0 license and
attribution. MBPP records retain CC BY 4.0 and attribution to Austin et al.
(2021); see the [data notice](../mbpp-training-v1/README.md). Experiment code
remains MIT. No private user data or paid hosted calls were used. No upstream
endorsement is implied.

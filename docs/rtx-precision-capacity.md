# RTX training capacity and precision follow-up

By Prashant Jagtap

This experiment checks whether the initial short GPU throughput probe is a reliable guide for longer training. It uses the existing public SciFact and NFCorpus training partitions, verified against the published data manifest. No test query is loaded, no checkpoint is retained and no accuracy improvement is claimed.

The RTX 5080 Laptop GPU executed **11,277 measured optimizer steps** across eight randomized 20-second windows. Each window has 25 warmup steps. There are two seeds for each recurrent/nonrecurrent and FP32/BF16 combination, all at batch 128 with equally sampled training domains. The complete run took 169.6 seconds, including numerical checks and cleanup.

## Throughput result

| Controller | FP32 query lists/s, median of two windows | BF16 query lists/s, median of two windows | BF16 / FP32 |
|---|---:|---:|---:|
| Nonrecurrent | 12,820 | 9,499 | 0.74× |
| Recurrent | 6,807 | 6,950 | 1.02× |

**BF16 did not provide a consistent speed advantage.** The two recurrent FP32 windows ranged from 5,650 to 7,964 query lists/s, and BF16 ranged from 4,909 to 8,991. That variability is much larger than their 2% median difference. The earlier 9.43-second sweep therefore does not establish sustained throughput or justify a blanket mixed-precision setting.

Telemetry collected 290 samples. GPU utilization was 44% at the median and 93% at the sampled peak; the highest sampled temperature was 68°C. Maximum PyTorch reserved memory was 480 MiB, and maximum sampled device memory use was 712 MiB. The small models and synchronous step/control work do not keep the GPU continuously saturated. These measurements do not establish the cause of every slowdown or represent production concurrency. No power limit, clock or operating-system application setting was changed.

## Numerical result

After each timing window, the same nonzero trained weights were evaluated in FP32 and BF16 on 128 training query lists. Across the eight checks, **936 of 1,024 lists retained the exact top-10 order**; two changed their top-ranked candidate. The lists overlap across windows, so 1,024 is a count of comparisons, not independent evaluation examples. No relevance gain or loss is inferred from these differences.

This is why training arithmetic and deployment quality must be qualified separately. The initial zero-correction weights would make a weak precision check; these comparisons use weights after optimizer updates. Still, they are training-input diagnostics, not held-out ranking certification. Existing serving precision and retrieval routes remain unchanged.

## Reproduce locally

Use the isolated Python 3.12 CUDA environment recorded in the [foundation run](../evidence/enterprise-context-v1/foundation-cuda/requirements.txt): Torch 2.11.0+cu128, NumPy 2.4.6 and the matching prepared arrays. The core library does not require Torch; this experiment does.

```powershell
# The prepared data root holds controller-v1-data and controller-v2-data.
# Unset an earlier CPU-only mask for this shell before starting the GPU process.
Remove-Item Env:CUDA_VISIBLE_DEVICES -ErrorAction SilentlyContinue
python experiments/profile_context_precision.py `
  --work C:\path\to\prepared-work `
  --out capacity-independent.json `
  --seconds 20

# CPU-only arithmetic and published source/data-manifest verification.
python experiments/verify_context_precision.py
```

Model parameters, losses and optimizer state stay FP32. BF16 is forward autocast only, with TF32 disabled. AdamW uses its fused CUDA implementation, learning rate 0.0003, weight decay 0.01 and gradient clipping at 1.0. Data stays on GPU; query sampling and candidate gathers are included. Throughput uses the entire timed window, including Python/control overhead. Data loading and the subsequent precision comparisons are excluded.

Each output path must be new. The runner records failures instead of overwriting earlier evidence. Windows have equal elapsed-time budgets and therefore different optimizer-step counts; their training losses must not be compared as equal-budget learning results.

Use batch 128 as a starting point for these existing controllers. Profile the actual new context model before choosing batch size and arithmetic. Keep preparation, deterministic state/compiler work and unit tests on CPU. The next training pipeline should measure data/launch/synchronization overhead and test queued execution or CUDA graphs with parameter-update parity before claiming better GPU utilization. Retain FP32 as the numerical reference and require held-out quality checks before promoting a trained or lower-precision candidate.

Raw windows, step times, source/data identities and telemetry: [capacity.json](../evidence/enterprise-gpu-v1/capacity.json). These are resource and numerical measurements, separate from the [temporal/compiler engineering results](enterprise-state.md) and existing public retrieval benchmarks.

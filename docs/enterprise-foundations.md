# Enterprise-runtime foundation record

By Prashant Jagtap · 26 September 2026

The revised objective is the [Enterprise Context Intelligence Runtime](enterprise-context-plan.md). The earlier 96-task general-learning programme is superseded. The new [register](../evidence/enterprise-context-v1/requirements.json) maps 20 requirement groups to current source and the evidence still needed. It does not mark the wider objective complete.

## Implemented in this increment

- [Experience contracts](experience-contracts.md): versioned, immutable, reference-only episode/proposal/outcome records; ten tests for typing, provenance bindings, uncertainty/cost bounds, failures and observation/prediction separation. No persistent ledger or model update is implemented by these records.
- Isolated CPU and CUDA environments, environment/package snapshots, source hashes, test logs and existing evidence replays. The initial new runner missed the repository import path; its failure and corrected CPU run remain under `evidence/general-learning-v1`. This was a runner setup failure, not an accuracy regression.
- A local GPU capacity probe on existing, fingerprinted public training partitions. It creates temporary optimizer states for timing and publishes no new trained checkpoint.

## Validation

| Profile | Python / Torch | Tests | Additional evidence |
|---|---|---|---|
| Existing isolated CPU environment | 3.13.13 / 2.14.0+cpu | 155 passed; no skips | Controller/runtime evidence replay and unified example |
| Newly isolated CUDA environment | 3.12.10 / 2.11.0+cu128 | 155 passed; no skips | CUDA calculation parity, controller/runtime evidence replay and unified example |
| Newly isolated lightweight CPU environment | 3.12.10 / no Torch | 129 passed; 16 intentional Torch skips before new schema tests | Dependency check and replay; new schema's ten tests also executed separately |

The full suite's tensor unit tests run on CPU. The CUDA smoke check confirms device operation; it is not a CUDA reproduction of every historical benchmark. Package versions are locked in each environment's `requirements.txt`; the GPU run needs the `https://download.pytorch.org/whl/cu128` wheel index for the recorded Torch build. The existing CPU Torch build uses the corresponding CPU wheel index. Install the repository separately from a reviewed checkout.

## GPU throughput probe

RTX 5080 Laptop GPU, 16,303 MiB reported capacity. The initial probe ran in the previously available Python 3.12 CUDA environment while the isolated replacement was downloading. It used only the registered SciFact/NFCorpus training partitions, no tuning/test queries. Model parameters/loss stayed FP32, forward autocast used BF16, TF32 was disabled and AdamW used its fused CUDA implementation.

| Controller | Batch | Query lists/second | Median step ms |
|---|---:|---:|---:|
| Nonrecurrent | 16 | 2,981 | 5.11 |
| Nonrecurrent | 128 | 18,460 | 6.77 |
| Nonrecurrent | 256 | 12,701 | 17.85 |
| Recurrent | 16 | 1,768 | 8.88 |
| Recurrent | 128 | 9,540 | 13.39 |
| Recurrent | 256 | 9,090 | 26.23 |

Raw observations for all ten cases are in [gpu-capacity.json](../evidence/enterprise-context-v1/gpu-capacity.json). Both domains are sampled equally, and candidate gathers happen on GPU. Each case has ten warm-up and thirty measured optimizer steps. Timing includes gather, forward, loss, backward and optimizer operations, but excludes data loading/installation. The whole sweep is approximately 9.43 seconds, so it does not characterize sustained thermal behavior. Coarse telemetry reached 95% GPU utilization and 58°C; this is a sampled peak, not sustained 95% use. Peak memory stayed below 1 GiB for these small controllers.

Use batch 128 as an initial setting for these two workloads. Do not infer that it is optimal for a future multi-head model or that increasing batch size preserves training quality. Re-profile the actual new architecture and validate its learning curve before adopting a changed batch. Filling unused VRAM or forcing 100% utilization is not itself an efficiency gain.

The [longer precision follow-up](rtx-precision-capacity.md) finds substantial window variability, no consistent BF16 speed advantage and some changed ranking orders. It preserves this initial probe while limiting how its peak throughput should be interpreted.

```powershell
# Run from a reviewed clone with the matching prepared training files outside Git.
python experiments/profile_context_training.py --work C:\path\to\prepared-work --out capacity-new.json

# Offline arithmetic/hash replay; no new training or downloaded source data.
python experiments/verify_enterprise_foundations.py
```

The subsequent [state/compiler/decision increment](enterprise-state.md) adds temporal and epistemic contracts, typed decisions and explicit evidence compilation. Durable audit and verified workflow data remain prerequisites for the larger context-model experiments. Mainline retrieval behavior and these historical foundation measurements remain unchanged.

"""Longer FP32/BF16 RTX capacity windows on hash-checked training data only.

All temporary optimizer states are discarded. This is not a trained release,
held-out accuracy test, foundation-model fine-tune or quantized deployment claim.
"""

import argparse
import gc
import json
import random
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from profile_context_training import sha, telemetry  # noqa: E402

from context_stamps.contractive_controller import ContractiveRanker, ndcg_pair_loss  # noqa: E402


def load_training(work):
    path = ROOT / "evidence/controller-v2/data-manifest.json"
    manifest = json.loads(path.read_text())
    training, provenance = [], []
    for part in manifest["partitions"]:
        if part["split"] != "train":
            continue
        name = part["dataset"]
        if name not in ("scifact", "nfcorpus"):
            raise ValueError("unexpected training domain")
        array = work / f"controller-v2-data/{name}-train.npz"
        documents = work / f"controller-v1-data/{name}-D.npy"
        metadata = array.with_suffix(".json")
        if (sha(array) != part["array_sha256"] or sha(documents) != part["document_sha256"]
                or sha(metadata) != part["metadata_sha256"]
                or json.loads(metadata.read_text())["query_ids"] != manifest["assignments"][name]["train"]):
            raise ValueError("training identity mismatch")
        with np.load(array, allow_pickle=False) as data:
            row = {key: torch.as_tensor(data[key], device="cuda") for key in ("Q", "F", "Y", "mask", "candidates")}
        row["D"] = torch.as_tensor(np.load(documents, allow_pickle=False), device="cuda")
        training.append(row)
        provenance.append(part)
    if len(training) != 2:
        raise ValueError("two registered domains required")
    return training, provenance, sha(path)


def gather(data, ix):
    return data["Q"][ix], data["D"][data["candidates"][ix].long()], data["F"][ix], data["mask"][ix]


def probe(training, *, recurrent, precision, seed, seconds):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    model = ContractiveRanker(recurrent=recurrent).cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=.01, fused=True)
    times, losses, sampled_losses = [], [], []
    started, count = None, 0
    torch.cuda.reset_peak_memory_stats()
    while started is None or time.perf_counter() - started < seconds:
        torch.cuda.synchronize()
        begin = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        total = torch.zeros((), device="cuda")
        for data in training:
            ix = torch.randint(len(data["Q"]), (64,), device="cuda")
            inputs = gather(data, ix)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=precision == "bf16"):
                scores = model(*inputs).float()
            loss = ndcg_pair_loss(scores, data["Y"][ix], inputs[3])
            (loss / 2).backward()
            total += loss.detach() / 2
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
        optimizer.step()
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - begin
        value = total.item()
        if not np.isfinite(value):
            raise FloatingPointError("nonfinite capacity-probe loss")
        count += 1
        if count == 25:
            started = time.perf_counter()
        elif count > 25:
            times.append(elapsed)
            losses.append(value)
            if len(times) == 1 or len(times) % 100 == 0:
                sampled_losses.append({"step": len(times), "loss": value})
    window_seconds = time.perf_counter() - started
    # Nonzero learned corrections: compare both precisions on the same post-window
    # weights and training inputs. This diagnoses arithmetic, not test quality.
    maximum, squares, values, same_top1, same_top10, queries = 0., 0., 0, 0, 0, 0
    model.eval()
    with torch.inference_mode():
        for data in training:
            inputs = gather(data, torch.arange(min(64, len(data["Q"])), device="cuda"))
            a = model(*inputs).float()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                b = model(*inputs).float()
            difference = (a - b)[inputs[3]]
            maximum = max(maximum, difference.abs().max().item())
            squares += difference.square().sum().item()
            values += difference.numel()
            order_a = a.argsort(dim=1, descending=True, stable=True)
            order_b = b.argsort(dim=1, descending=True, stable=True)
            same_top1 += (order_a[:, 0] == order_b[:, 0]).sum().item()
            same_top10 += (order_a[:, :10] == order_b[:, :10]).all(dim=1).sum().item()
            queries += a.shape[0]
    row = dict(status="measured", recurrent=recurrent, precision=precision, seed=seed,
        batch=128, measured_steps=len(times), window_seconds=window_seconds,
        queries_per_second=128 * len(times) / window_seconds,
        median_step_ms=statistics.median(times) * 1000,
        step_seconds=times, minimum_loss=min(losses), maximum_loss=max(losses), sampled_losses=sampled_losses,
        peak_allocated_bytes=torch.cuda.max_memory_allocated(), peak_reserved_bytes=torch.cuda.max_memory_reserved(),
        precision_comparison=dict(max_abs_score_error=maximum, rmse=(squares / values)**.5,
            candidate_values=values, queries=queries, unchanged_top1=same_top1, unchanged_top10_order=same_top10))
    del model, optimizer, inputs, scores, loss, total, a, b, difference
    gc.collect()
    torch.cuda.empty_cache()
    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seconds", type=int, default=20)
    args = parser.parse_args()
    if args.out.exists() or not 10 <= args.seconds <= 60:
        raise ValueError("new evidence file and 10..60 second windows required")
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("BF16-capable CUDA GPU required")
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    training, partitions, manifest_sha = load_training(args.work)
    cases = [dict(recurrent=r, precision=p, seed=s) for s in (7, 29) for r in (False, True) for p in ("fp32", "bf16")]
    random.Random(2718).shuffle(cases)
    samples, rows = [], []
    stop = threading.Event()
    monitor = threading.Thread(target=telemetry, args=(stop, samples), daemon=True)
    monitor.start()
    started = time.perf_counter()
    error = None
    try:
        for index, case in enumerate(cases):
            print(json.dumps({"starting_case": index, **case}), flush=True)
            row = probe(training, **case, seconds=args.seconds)
            rows.append(row)
            print(json.dumps({key: value for key, value in row.items() if key not in ("step_seconds", "sampled_losses")}), flush=True)
    except Exception as exc:
        error = type(exc).__name__
    finally:
        stop.set()
        monitor.join(timeout=12)
        report = dict(schema=1, status="failed" if error else "measured", failure_type=error,
            source_base=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
            source_hashes={p.relative_to(ROOT).as_posix(): sha(p) for p in (
                Path(__file__), ROOT / "experiments/profile_context_training.py", ROOT / "context_stamps/contractive_controller.py")},
            data_manifest_sha256=manifest_sha, training_partitions=partitions,
            python=sys.version, torch=torch.__version__, cuda=torch.version.cuda, numpy=np.__version__,
            device=torch.cuda.get_device_name(0), vram_bytes=torch.cuda.get_device_properties(0).total_memory,
            threads=4, window_seconds=args.seconds, case_order=cases, rows=rows, gpu_telemetry=samples,
            elapsed_seconds=time.perf_counter() - started,
            precision="FP32 weights/loss/optimizer; optional BF16 forward autocast; TF32 disabled",
            optimizer="fused AdamW lr0.0003 weight_decay0.01 clip1.0; 25 warmup steps; equal training-domain sampling",
            limits=["Timing/finite-training/numerical probe only. No checkpoint retained or held-out quality measured.",
                    "Window throughput includes Python/control overhead; data loading and precision checks excluded.",
                    "Randomized case order, two windows per variant/precision on one laptop; not a thermal stability guarantee.",
                    "FP32/BF16 parity uses post-training weights on training queries; not serving or task-quality certification.",
                    "Window duration fixes time rather than optimizer steps. Losses are not equal-training-budget quality comparisons."])
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("w", encoding="utf-8", newline="\n") as stream:
            json.dump(report, stream, indent=2)
            stream.write("\n")
    if error:
        raise SystemExit("capacity probe failed; failure type preserved")


if __name__ == "__main__":
    main()

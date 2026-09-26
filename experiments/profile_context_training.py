"""Measure GPU training capacity on existing, hash-checked training data only.

Temporary optimizer steps are throughput probes, not trained release candidates.
No test split is loaded and no checkpoint is promoted by this experiment.
"""

import argparse
import gc
import hashlib
import json
import os
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
from context_stamps.contractive_controller import ContractiveRanker, ndcg_pair_loss  # noqa: E402


def sha(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def telemetry(stop, rows):
    while not stop.is_set():
        proc = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,temperature.gpu,power.draw",
                               "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=10)
        if proc.returncode == 0:
            rows.append({"time": time.time(), "sample": proc.stdout.strip()})
        stop.wait(.5)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--work", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--steps", type=int, default=30)
    args = parser.parse_args()
    if not 10 <= args.steps <= 200 or args.out.exists():
        raise ValueError("10..200 measured steps and a new output path required")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    manifest_path = ROOT / "evidence/controller-v2/data-manifest.json"
    manifest = json.loads(manifest_path.read_text())
    training, provenance = [], []
    for part in manifest["partitions"]:
        if part["split"] != "train":
            continue
        name = part["dataset"]
        if name not in ("scifact", "nfcorpus"):
            raise ValueError("unexpected training dataset")
        array_path = args.work / f"controller-v2-data/{name}-train.npz"
        docs_path = args.work / f"controller-v1-data/{name}-D.npy"
        if sha(array_path) != part["array_sha256"] or sha(docs_path) != part["document_sha256"]:
            raise ValueError("prepared training data hash mismatch")
        meta = array_path.with_suffix(".json")
        if sha(meta) != part["metadata_sha256"]:
            raise ValueError("metadata hash mismatch")
        if json.loads(meta.read_text())["query_ids"] != manifest["assignments"][name]["train"]:
            raise ValueError("training split mismatch")
        with np.load(array_path, allow_pickle=False) as data:
            row = {key: torch.as_tensor(data[key], device="cuda") for key in ("Q", "F", "Y", "mask", "candidates")}
        row["D"] = torch.as_tensor(np.load(docs_path, allow_pickle=False), device="cuda")
        training.append(row)
        provenance.append(part)
    if len(training) != 2:
        raise ValueError("both training domains required")
    total_memory = torch.cuda.get_device_properties(0).total_memory
    rows, samples = [], []
    stop = threading.Event()
    monitor = threading.Thread(target=telemetry, args=(stop, samples), daemon=True)
    monitor.start()
    started = time.perf_counter()
    try:
        for recurrent in (False, True):
            for batch_size in (16, 32, 64, 128, 256):
                torch.manual_seed(7)
                torch.cuda.manual_seed_all(7)
                model = ContractiveRanker(recurrent=recurrent).cuda()
                optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=.01, fused=True)
                torch.cuda.reset_peak_memory_stats()
                elapsed, losses = [], []
                row = {"recurrent": recurrent, "batch_size": batch_size, "status": "running"}
                try:
                    for step in range(args.steps + 10):
                        torch.cuda.synchronize()
                        begin = time.perf_counter()
                        optimizer.zero_grad(set_to_none=True)
                        # Balanced domains; GPU-side gather eliminates repeated host materialization.
                        total = torch.zeros((), device="cuda")
                        for data in training:
                            ix = torch.randint(len(data["Q"]), (batch_size // 2,), device="cuda")
                            inputs = (data["Q"][ix], data["D"][data["candidates"][ix].long()],
                                      data["F"][ix], data["mask"][ix])
                            with torch.autocast("cuda", dtype=torch.bfloat16):
                                scores = model(*inputs).float()
                            loss = ndcg_pair_loss(scores, data["Y"][ix], inputs[3])
                            (loss / 2).backward()
                            total += loss.detach() / 2
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
                        optimizer.step()
                        torch.cuda.synchronize()
                        seconds = time.perf_counter() - begin
                        if not torch.isfinite(total).item():
                            raise FloatingPointError("non-finite training loss")
                        if step >= 10:
                            elapsed.append(seconds)
                            losses.append(total.item())
                    row.update(status="measured", median_step_ms=statistics.median(elapsed) * 1000,
                        p95_step_ms=float(np.percentile(elapsed, 95)) * 1000,
                        queries_per_second=batch_size * len(elapsed) / sum(elapsed),
                        peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                        peak_reserved_bytes=torch.cuda.max_memory_reserved(), step_seconds=elapsed,
                        training_losses=losses)
                    row["headroom_qualified"] = row["peak_reserved_bytes"] <= .85 * total_memory
                except torch.OutOfMemoryError:
                    row.update(status="out_of_memory", headroom_qualified=False)
                rows.append(row)
                print(json.dumps({k: v for k, v in row.items() if k not in ("step_seconds", "training_losses")}), flush=True)
                del model, optimizer
                if "inputs" in locals():
                    del inputs, scores, loss, total
                gc.collect()
                torch.cuda.empty_cache()
                if row["status"] == "out_of_memory":
                    break
    finally:
        stop.set()
        monitor.join(timeout=12)
    qualified = [r for r in rows if r.get("headroom_qualified")]
    selected = {str(recurrent): max((r for r in qualified if r["recurrent"] == recurrent),
                                    key=lambda r: r["queries_per_second"], default=None)
                for recurrent in (False, True)}
    report = {"schema": 1, "source_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
              "profile_source_sha256": sha(Path(__file__)),
              "ranker_source_sha256": sha(ROOT / "context_stamps/contractive_controller.py"),
              "data_manifest_sha256": sha(manifest_path), "training_partitions": provenance,
              "torch": torch.__version__, "cuda": torch.version.cuda, "numpy": np.__version__,
              "device": torch.cuda.get_device_name(0), "vram_bytes": total_memory,
              "seed": 7, "precision": "FP32 parameters/loss, BF16 autocast, TF32 disabled",
              "optimizer": "fused AdamW lr0.0003 weight_decay0.01 clip1.0",
              "threads": torch.get_num_threads(), "environment": {k: os.getenv(k) for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS")},
              "warmup_steps": 10, "measured_steps": args.steps, "rows": rows,
              "selected_batch_by_recurrence": {k: v["batch_size"] if v else None for k, v in selected.items()},
              "elapsed_seconds": time.perf_counter() - started, "gpu_telemetry": samples,
              "limits": ["Training throughput probe only; no held-out accuracy or promoted checkpoint.",
                         "Training arrays resident on GPU. Timing includes gather, forward, loss, backward and optimizer.",
                         "Different batch sizes change optimization; throughput choice needs task-quality revalidation.",
                         "One hardware run, fixed case order; thermal/repeat variability is not characterized."]}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

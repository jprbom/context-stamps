"""Post-evaluation numerical audit: preserve FP32 retrieval features in int8 input.

No fitting, model selection or gate changes. Existing evaluation sets are now
regression data. Preserve whole-layer int8 results as the failure control.
"""

import argparse
import copy
import gzip
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))
from run_hybrid_retrieval import read, save, sha  # noqa: E402
from train_controller import batch, scored_records  # noqa: E402
from train_controller_v2 import OUT, load_partitions, values  # noqa: E402

from context_stamps.contractive_controller import load_contractive  # noqa: E402
from context_stamps.controller_quantization import split_precision_controller  # noqa: E402


def main(work):
    torch.set_num_threads(4)
    torch.backends.quantized.engine = next(v for v in ("x86", "onednn", "fbgemm", "qnnpack")
                                          if v in torch.backends.quantized.supported_engines)
    chosen = read(OUT / "selection.json")["selected_student"]
    model = load_contractive(OUT / ("models/" + chosen + ".safetensors"))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        whole = torch.ao.quantization.quantize_dynamic(copy.deepcopy(model), {torch.nn.Linear}, dtype=torch.qint8)
    models = {"cpu-fp32": model, "whole-int8": whole, "split-int8": split_precision_controller(model)}
    partitions = load_partitions(work, "test")
    records, fidelity, timing = [], {}, {}
    with torch.inference_mode():
        for data in partitions:
            predictions = {key: [] for key in models}
            for offset in range(0, len(data["Q"]), 32):
                inputs = batch(data, np.arange(offset, min(offset + 32, len(data["Q"]))), "cpu")
                for key, candidate in models.items():
                    predictions[key].append(candidate(*inputs).numpy())
            predictions = {key: np.concatenate(value) for key, value in predictions.items()}
            for key, score in predictions.items():
                records.extend(scored_records(data, score, key))
            reference = predictions["cpu-fp32"]
            fidelity[data["name"]] = {}
            for key, score in predictions.items():
                error = np.abs(score - reference)[data["mask"]]
                orders, ref_orders = np.argsort(-score, axis=1, kind="stable")[:, :10], np.argsort(-reference, axis=1, kind="stable")[:, :10]
                fidelity[data["name"]][key] = dict(mean_abs_logit_error=float(error.mean()),
                    max_abs_logit_error=float(error.max()), exact_top10_order_fraction=float(np.mean(np.all(orders == ref_orders, axis=1))),
                    ndcg10=float(values(records, data["name"], key).mean()))
            samples = {key: [] for key in models}
            parity = {key: [] for key in models}
            for i in np.linspace(0, len(data["Q"]) - 1, 20, dtype=int):
                inputs = batch(data, [i], "cpu")
                for repeat in range(4):
                    for key, candidate in models.items():
                        start = time.perf_counter()
                        output = candidate.score_candidates(*inputs)[0].numpy()
                        elapsed = (time.perf_counter() - start) * 1000
                        if repeat:
                            samples[key].append(elapsed)
                            parity[key].append(bool(np.array_equal(np.argsort(-output, kind="stable")[:10],
                                np.argsort(-predictions[key][i], kind="stable")[:10])))
            timing[data["name"]] = {key: dict(p50_ms=float(np.median(sample)), p95_ms=float(np.percentile(sample, 95)),
                exact_batch1_vs_batch32_fraction=float(np.mean(parity[key]))) for key, sample in samples.items()}
            print(data["name"], fidelity[data["name"]], flush=True)
    (OUT / "split-quantization-rankings.json.gz").write_bytes(gzip.compress(json.dumps(records, separators=(",", ":")).encode(), mtime=0))
    source = ["experiments/audit_split_quantization.py", "context_stamps/controller_quantization.py"]
    save(OUT / "split-quantization.json", dict(selected_student=chosen, fidelity=fidelity, forward_timing=timing,
        engine=torch.backends.quantized.engine, torch=torch.__version__, cpu_threads=4,
        source_sha256={p: sha(ROOT / p) for p in source},
        method="Split original input matrix into embedding projection and six-feature FP32 projection; quantize only embedding projection; keep output/attention FP32. No fitting or threshold tuning.",
        limits="Post-evaluation numerical engineering regression, not a fresh accuracy confirmation. Batch32 quality and batch1 timing/parity; dynamic activation quantization remains batch-dependent. Forward-only timing includes input validation but excludes retrieval, encoding, reader and transfer preparation. No serving gate changes."))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work", required=True, type=Path)
    main(parser.parse_args().work)

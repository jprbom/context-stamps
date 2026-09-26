"""Measure real online reranking and actual CPU dynamic-int8 execution."""

import argparse
import copy
import gzip
import json
import math
import platform
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))
from prepare_controller_v2 import teacher_model, teacher_scores  # noqa: E402
from run_hybrid_retrieval import BM25, read, save, sha  # noqa: E402
from train_controller import evaluate  # noqa: E402
from train_controller_v2 import OUT, load_partitions, normalize_lists, paired_bound, values  # noqa: E402

from context_stamps.contractive_controller import load_contractive  # noqa: E402


def retrieval(data, query, vector, lexical, positions, method):
    ds = (data["D"] @ vector).astype(np.float64)
    eligible = np.ones(len(ds), dtype=bool)
    if query in positions:
        eligible[positions[query]] = False
    dense = np.argsort(-np.where(eligible, ds, -np.inf), kind="stable")
    if method == "dense":
        return dense[:10], None
    ls = lexical.score(data["queries"][query])
    dz = (ds - ds[eligible].mean()) / max(float(ds[eligible].std()), 1e-9)
    lz = (ls - ls[eligible].mean()) / max(float(ls[eligible].std()), 1e-9)
    hs = .75 * dz + .25 * lz
    hybrid = np.argsort(-np.where(eligible, hs, -np.inf), kind="stable")
    if method == "hybrid":
        return hybrid[:10], None
    lexical_order = np.argsort(-np.where(eligible, ls, -np.inf), kind="stable")
    chosen = np.asarray(sorted(set(dense[:96]) | set(hybrid[:96]) | set(lexical_order[:64])))
    dr, lr = np.empty(len(ds), dtype=np.int32), np.empty(len(ds), dtype=np.int32)
    dr[dense], lr[lexical_order] = np.arange(len(ds)), np.arange(len(ds))
    features = np.stack((ds[chosen], dz[chosen], hs[chosen], lz[chosen], 1 / (1 + dr[chosen]),
                         1 / (1 + lr[chosen])), axis=1).astype(np.float32)
    return chosen, features


def main(work):
    torch.set_num_threads(4)
    protocol, selection = read(OUT / "protocol.json"), read(OUT / "selection.json")
    model = load_contractive(OUT / ("models/" + selection["selected_student"] + ".safetensors"))
    test = load_partitions(work, "test")
    # This really dispatches quantized CPU Linear operators; no quantized pickle export.
    available = torch.backends.quantized.supported_engines
    engine = next((name for name in ("x86", "onednn", "fbgemm", "qnnpack") if name in available), None)
    if engine is None:
        raise RuntimeError("no supported dynamic-int8 CPU backend; record this hardware limitation")
    torch.backends.quantized.engine = engine
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        quantized = torch.ao.quantization.quantize_dynamic(copy.deepcopy(model), {torch.nn.Linear}, dtype=torch.qint8)
    int8_records, _ = evaluate(quantized, test, "dynamic-int8", device="cpu")
    previous = json.loads(gzip.decompress((OUT / "rankings.json.gz").read_bytes()))["test"]
    reference = {(r["dataset"], r["query_id"], r["method"]): r["ranked_ids"] for r in previous + int8_records}
    quantization = {d["name"]: dict(ndcg10=float(values(int8_records, d["name"], "dynamic-int8").mean()),
         delta_vs_fp32=paired_bound(values(int8_records, d["name"], "dynamic-int8"), values(previous, d["name"], "student"))) for d in test}
    (OUT / "quantized-rankings.json.gz").write_bytes(gzip.compress(json.dumps(int8_records, separators=(",", ":")).encode(), mtime=0))
    tokenizer, teacher = teacher_model(protocol)
    measurements, builds = [], []
    for data in test:
        name = data["name"]
        source = work / ("public-validation/" + name if name in ("scidocs", "fiqa") else "replication-data/" + name)
        for relative, expected in read(OUT / "data-manifest.json")["data_sha256"][name].items():
            if sha(source / relative) != expected:
                raise ValueError("raw benchmark data changed after preparation")
        data["queries"] = {r["_id"]: r["text"] for r in map(json.loads, (source / "queries.jsonl").read_text(encoding="utf-8").splitlines())}
        docs = [json.loads(line) for line in (source / "corpus.jsonl").read_text(encoding="utf-8").splitlines()]
        texts = [(d.get("title", "") + " " + d["text"]).strip() for d in docs]
        start = time.perf_counter()
        lexical = BM25(texts)
        builds.append(dict(dataset=name, lexical_index_build_seconds=time.perf_counter() - start))
        positions = {d["_id"]: i for i, d in enumerate(docs)}
        indices = np.linspace(0, len(data["Q"]) - 1, 20, dtype=int)
        methods = ["dense", "hybrid", "student", "dynamic-int8", "teacher", "fusion"]
        for repeat in range(4):
            for i in indices:
                qid = data["meta"]["query_ids"][i]
                rel = {data["meta"]["document_ids"][j]: float(y) for j, y, valid in
                       zip(data["candidates"][i], data["Y"][i], data["mask"][i]) if valid}
                # Rotate order to avoid assigning warm/thermal effects to a fixed method.
                rotated = methods[repeat:] + methods[:repeat]
                for method in rotated:
                    torch.cuda.synchronize()
                    start = time.perf_counter()
                    chosen, features = retrieval(data, qid, data["Q"][i], lexical, positions, method)
                    if method in ("student", "dynamic-int8"):
                        ranker = quantized if method == "dynamic-int8" else model
                        inputs = (torch.from_numpy(data["Q"][i:i + 1]), torch.from_numpy(data["D"][chosen][None]),
                                  torch.from_numpy(features[None]), torch.ones(1, len(chosen), dtype=torch.bool))
                        scores = ranker.score_candidates(*inputs)[0].numpy()
                        result = chosen[np.argsort(-scores, kind="stable")[:10]]
                    elif method in ("teacher", "fusion"):
                        scores = teacher_scores([data["queries"][qid]] * len(chosen), [texts[j] for j in chosen], tokenizer, teacher)
                        if method == "fusion":
                            mask = np.ones((1, len(chosen)), dtype=bool)
                            scores = (.5 * normalize_lists(scores[None], mask) + .5 * normalize_lists(features[None, :, 2], mask))[0]
                        result = chosen[np.argsort(-scores, kind="stable")[:10]]
                    else:
                        result = chosen
                    torch.cuda.synchronize()
                    elapsed = (time.perf_counter() - start) * 1000
                    if repeat > 0:
                        ids = [docs[j]["_id"] for j in result]
                        measurements.append(dict(dataset=name, query_id=qid, repeat=repeat, method=method,
                            milliseconds=elapsed, ranked_ids=ids,
                            ndcg10=sum(rel[key] / math.log2(j + 2) for j, key in enumerate(ids)) / max(data["meta"]["records"][i]["ideal"], 1e-12),
                            matches_batched_ranking=ids == reference[(name, qid, method)]))
            print("Timed", name, "round", repeat, flush=True)
    timings = {n: {m: {key: float(np.percentile([r["milliseconds"] for r in measurements
                    if r["dataset"] == n and r["method"] == m], percentile))
                    for key, percentile in (("p50_ms", 50), ("p95_ms", 95))} for m in methods} for n in quantization}
    save(OUT / "online-runtime.json", dict(quantization=quantization, timings=timings, lexical_builds=builds,
        methods=methods, observations=measurements, queries_per_dataset=20, measured_repeats=3, warmup_repeats=1,
        batch_shape_parity={n: {m: float(np.mean([r["matches_batched_ranking"] for r in measurements
            if r["dataset"] == n and r["method"] == m])) for m in methods} for n in quantization},
        cpu_threads=4, cpu=platform.processor(), gpu=torch.cuda.get_device_name(), torch=torch.__version__, quantized_engine=torch.backends.quantized.engine,
        source_sha256={"experiments/audit_controller_v2.py": sha(Path(__file__))},
        limits="Warm single-user in-process CPU exhaustive retrieval and student versus GPU teacher. Includes retrieval, feature construction, reranker tokenization/transfers/scoring. Excludes query encoding, document encoding, index construction, reader generation, networking and concurrent load. Twenty fixed evenly spaced test queries per dataset, not a production latency SLA. Dynamic int8 is experimental and does not enter frozen method selection."))
    print(json.dumps(dict(quantization=quantization, timings=timings), indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work", required=True, type=Path)
    main(parser.parse_args().work)

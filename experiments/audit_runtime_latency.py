"""Replay all five inspected collections and measure paired online stage latency."""

import argparse
import gzip
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from audit_controller_v2 import retrieval  # noqa: E402
from prepare_controller_v2 import teacher_model, teacher_scores  # noqa: E402
from run_hybrid_retrieval import BM25, read, save, sha  # noqa: E402
from train_controller import scored_records  # noqa: E402
from train_controller_v2 import load_partitions, normalize_lists, paired_bound  # noqa: E402
from train_expert_router import route_features  # noqa: E402

from context_stamps.efficient_reranker import EfficientReranker  # noqa: E402
from context_stamps.expert_router import LinearExpertRouter  # noqa: E402

OUT = ROOT / "evidence/runtime-v1"


def main(work):
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    protocol = read(ROOT / "evidence/controller-v2/protocol.json")
    tokenizer, teacher = teacher_model(protocol)
    fast = EfficientReranker(tokenizer, teacher)
    router = LinearExpertRouter(**read(OUT / "router.json"))
    old = json.loads(gzip.decompress((ROOT / "evidence/controller-v2/rankings.json.gz").read_bytes()))["test"]
    previous = {(r["dataset"], r["query_id"], r["method"]): r for r in old}
    selection = read(ROOT / "evidence/controller-v2/selection.json")["decisions"]
    manifest = read(ROOT / "evidence/controller-v2/data-manifest.json")
    records, observations, summary = [], [], {}
    for data in load_partitions(work, "test"):
        name = data["name"]
        source = work / (("public-validation/" if name in ("fiqa", "scidocs") else "replication-data/") + name)
        for p, digest in manifest["data_sha256"][name].items():
            assert sha(source / p) == digest
        docs = [json.loads(x) for x in (source / "corpus.jsonl").read_text(encoding="utf-8").splitlines()]
        texts = [(d.get("title", "") + " " + d["text"]).strip() for d in docs]
        data["queries"] = {d["_id"]: d["text"] for d in map(json.loads, (source / "queries.jsonl").read_text(encoding="utf-8").splitlines())}
        fast.clear_cache()
        scores = np.zeros_like(data["teacher"])
        max_error, absolute_error, pairs = 0., 0., 0
        start = time.perf_counter()
        for i, qid in enumerate(data["meta"]["query_ids"]):
            valid = data["mask"][i]
            values = fast.score(data["queries"][qid], [texts[j] for j in data["candidates"][i][valid]])
            scores[i, valid] = values
            error = np.abs(values - data["teacher"][i][valid])
            max_error = max(max_error, float(error.max()))
            absolute_error += float(error.sum())
            pairs += len(error)
            if i % 100 == 0:
                print(name, "regression", i, "/", len(scores), flush=True)
        fusion = .5 * normalize_lists(scores, data["mask"]) + .5 * normalize_lists(data["F"][..., 2], data["mask"])
        current = scored_records(data, np.where(data["mask"], fusion, -np.inf), "accelerated_fusion")
        mismatches = sum(r["ranked_ids"] != previous[name, r["query_id"], "fusion"]["ranked_ids"] for r in current)
        reference_values = [previous[name, r["query_id"], "fusion"]["ndcg10"] for r in current]
        approved = mismatches == 0 and all(r["ndcg10"] == v for r, v in zip(current, reference_values))
        routed, cheap_count = [], 0
        for i, r in enumerate(current):
            qid = r["query_id"]
            decision, reason = router.choose(route_features(data["F"][i][data["mask"][i]]), scope=name)
            # No scope loses the earlier dense fallback. Failed acceleration keeps reference.
            if selection[name]["deployed"] != "fusion":
                decision = "dense"
            selected = r if decision == "fusion" and approved else previous[name, qid, decision]
            cheap_count += decision == "hybrid"
            routed.append(dict(selected, method="runtime_policy", route=decision, route_reason=reason))
        records.extend(current + routed)
        summary[name] = dict(queries=len(current), pairs=pairs, accelerated_approved=approved,
            top10_mismatches=mismatches, max_logit_error=max_error, mean_abs_logit_error=absolute_error / pairs,
            reference_fusion_ndcg=float(np.mean(reference_values)), accelerated_fusion_ndcg=float(np.mean([r["ndcg10"] for r in current])),
            runtime_ndcg=float(np.mean([r["ndcg10"] for r in routed])), cheap_queries=cheap_count,
            runtime_delta=paired_bound([r["ndcg10"] for r in routed], [previous[name, r["query_id"], "gated"]["ndcg10"] for r in routed]),
            regression_seconds=time.perf_counter() - start)
        lexical = BM25(texts)
        positions = {d["_id"]: i for i, d in enumerate(docs)}
        indices = np.linspace(0, len(data["Q"]) - 1, 20, dtype=int)
        methods = ["reference_fusion", "accelerated_cold", "accelerated_warm", "existing_policy", "runtime_policy"]
        for repeat in range(4):
            for i in indices:
                qid, query = data["meta"]["query_ids"][i], data["queries"][data["meta"]["query_ids"][i]]
                for method in methods[repeat:] + methods[:repeat]:
                    if method == "accelerated_cold":
                        fast.clear_cache()
                    elif method in ("accelerated_warm", "runtime_policy"):
                        valid = data["mask"][i]
                        fast.score(query, [texts[j] for j in data["candidates"][i][valid]])
                    torch.cuda.synchronize()
                    started = time.perf_counter()
                    original = selection[name]["deployed"]
                    retrieval_method = "dense" if method in ("existing_policy", "runtime_policy") and original == "dense" else "fusion"
                    chosen, features = retrieval(data, qid, data["Q"][i], lexical, positions, retrieval_method)
                    route = retrieval_method
                    if method == "runtime_policy" and route != "dense":
                        route, _ = router.choose(route_features(features), scope=name)
                    if route == "dense":
                        result = chosen
                    elif route == "hybrid":
                        result = chosen[np.argsort(-features[:, 2], kind="stable")[:10]]
                    else:
                        passages = [texts[j] for j in chosen]
                        use_fast = method in ("accelerated_cold", "accelerated_warm") or method == "runtime_policy" and approved
                        values = fast.score(query, passages) if use_fast else teacher_scores([query] * len(chosen), passages, tokenizer, teacher)
                        mask = np.ones((1, len(chosen)), dtype=bool)
                        combined = (.5 * normalize_lists(values[None], mask) + .5 * normalize_lists(features[None, :, 2], mask))[0]
                        result = chosen[np.argsort(-combined, kind="stable")[:10]]
                    torch.cuda.synchronize()
                    elapsed = (time.perf_counter() - started) * 1000
                    ids = [docs[j]["_id"] for j in result]
                    if method in ("reference_fusion", "existing_policy") or approved and route == "fusion":
                        assert ids == previous[name, qid, "fusion" if route == "fusion" else "dense"]["ranked_ids"], (name, qid, method)
                    if repeat:
                        observations.append(dict(dataset=name, query_id=qid, method=method, repeat=repeat, milliseconds=elapsed, route=route))
        summary[name]["latency"] = {m: {"p50_ms": float(np.percentile([r["milliseconds"] for r in observations if r["dataset"] == name and r["method"] == m], 50)),
            "p95_ms": float(np.percentile([r["milliseconds"] for r in observations if r["dataset"] == name and r["method"] == m], 95))} for m in methods}
        save(OUT / "latency-summary.json", summary)
        print(name, json.dumps(summary[name]), flush=True)
    (OUT / "latency-rankings.json.gz").write_bytes(gzip.compress(json.dumps(records, separators=(",", ":")).encode(), mtime=0))
    save(OUT / "latency-observations.json", observations)
    save(OUT / "latency-manifest.json", dict(protocol_sha256=sha(OUT / "protocol.json"),
        source_sha256={p: sha(ROOT / p) for p in ("experiments/audit_runtime_latency.py", "context_stamps/efficient_reranker.py",
            "context_stamps/expert_router.py", "experiments/train_expert_router.py")},
        python=platform.python_version(), torch=torch.__version__, gpu=torch.cuda.get_device_name(),
        precision="FP32 weights, BF16 autocast", query_count=sum(r["queries"] for r in summary.values())))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--work", type=Path, required=True)
    main(parser.parse_args().work)

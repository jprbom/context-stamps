"""Offline source, artifact, grading and aggregation checks for follow-up evidence."""

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.run_local_tasks import check

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def close(actual, expected):
    if not np.allclose(actual, expected, rtol=0, atol=1e-10):
        raise ValueError(f"aggregate mismatch: {actual} != {expected}")


def verify():
    for directory in ["replication-v1", "local-tasks-v1", "usability-smoke-v1"]:
        folder = ROOT / "evidence" / directory
        for name, expected in read(folder / "checksums.json").items():
            if hashlib.sha256((folder / name).read_bytes()).hexdigest() != expected:
                raise ValueError(f"artifact changed: {directory}/{name}")
        manifest = read(folder / "manifest.json")
        for name, expected in manifest.get("source_sha256", {}).items():
            if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != expected:
                raise ValueError(f"experiment source changed: {name}")
    folder = ROOT / "evidence/replication-v1"
    manifest = read(folder / "manifest.json")
    protocol = read(folder / "protocol.json")
    if manifest["protocol"] != protocol or protocol["retuning"]:
        raise ValueError("frozen protocol changed")
    weights = ROOT / "evidence/scifact-v1/selector-seed-7.json"
    if hashlib.sha256(weights.read_bytes()).hexdigest() != manifest["selector_weights_sha256"]:
        raise ValueError("frozen trained weights changed")
    rows = [json.loads(line) for line in (folder / "per-query.jsonl").read_text().splitlines()]
    expected_counts = {"nfcorpus": 323, "arguana": 1406}
    for dataset, count in expected_counts.items():
        for method in ["dense", "bm25", "coverage_diversity", "mmr", "frozen_linear"]:
            group = [r for r in rows if r["dataset"] == dataset and r["method"] == method]
            if len(group) != count or len({r["query_id"] for r in group}) != count:
                raise ValueError("incomplete or duplicate queries")
            for row in group:
                if row["query_id"] in row["ranked_ids"] or len(set(row["ranked_ids"])) != 10:
                    raise ValueError("self retrieval or duplicate/missing rank")
                if row["missing_positive_ids"] and (
                    row["low_lexical_overlap"] or row["max_positive_jaccard"] is not None
                ):
                    raise ValueError("missing positives misclassified")
        rng = np.random.default_rng(protocol["bootstrap_seed"])
        for stratum in ["all", "low_lexical_overlap"]:
            chosen = [
                r for r in rows if r["dataset"] == dataset and (stratum == "all" or r["low_lexical_overlap"])
            ]
            dense = np.array([r["ndcg10"] for r in chosen if r["method"] == "dense"])
            boot = rng.integers(0, len(dense), (protocol["bootstrap_samples"], len(dense)))
            for summary in read(folder / "summary.json"):
                if summary["dataset"] != dataset or summary["stratum"] != stratum:
                    continue
                group = [r for r in chosen if r["method"] == summary["method"]]
                close(len(group), summary["n"])
                for metric in ["ndcg10", "recall10", "retrieval_selection_ms"]:
                    close(np.mean([r[metric] for r in group]), summary[metric])
                delta = np.array([r["ndcg10"] for r in group]) - dense
                close(delta.mean(), summary["paired_delta_vs_dense"])
                close(np.quantile(delta[boot].mean(axis=1), [0.025, 0.975]), summary["paired_ci95"])
    folder = ROOT / "evidence/local-tasks-v1"
    tasks = {r["id"]: r for r in read(folder / "fixtures.json")}
    rows = [json.loads(line) for line in (folder / "per-call.jsonl").read_text().splitlines()]
    expected = {
        (case, repetition, mode)
        for case in tasks
        for repetition in range(2)
        for mode in ["full_current", "stamps", "stale_cache_control"]
    }
    if len(rows) != len(expected) or {(r["case"], r["repetition"], r["mode"]) for r in rows} != expected:
        raise ValueError("incomplete or duplicate model calls")
    for row in rows:
        for key, value in check(tasks[row["case"]], row["output"]).items():
            close(value, row[key])
        close(row["end_to_end_ms"], row["request_ms"] + row["selection_ms"] + row["ingestion_ms_shared"])
        if row["mode"] == "stamps" and row["packet"]["status"] != "current":
            raise ValueError("unexpected packet status")
    for summary in read(folder / "summary.json"):
        group = [r for r in rows if r["kind"] == summary["kind"] and r["mode"] == summary["mode"]]
        close(len(group), summary["calls"])
        for key in [
            "correct",
            "stale_answer",
            "citation_correct",
            "valid_output",
            "prompt_tokens",
            "output_tokens",
            "request_ms",
            "selection_ms",
            "end_to_end_ms",
        ]:
            close(np.mean([r[key] for r in group]), summary[key])
    print(
        "Follow-up evidence: artifact/source hashes, query counts, bootstrap intervals and all model grades verified"
    )


if __name__ == "__main__":
    verify()

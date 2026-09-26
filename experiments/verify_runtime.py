"""Offline replay of ranking, workflow and retained-failure evidence."""

import gzip
import hashlib
import json
import math
import statistics
from collections import Counter
from pathlib import Path

from source_evidence import verify_sources

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "evidence/runtime-v1"


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def close(a, b):
    # Recorded GPU/NumPy metrics use float32; replay arithmetic is Python float64.
    assert math.isclose(a, b, rel_tol=1e-6, abs_tol=1e-7), (a, b)


def main():
    for path, digest in read(OUT / "checksums.json").items():
        assert sha(OUT / path) == digest, path
    for filename in ("latency-manifest.json", "workflow-manifest.json", "router-training.json", "spherical-ablation.json", "exact-tool-control.json"):
        verify_sources(read(OUT / filename)["source_sha256"])
    assert read(OUT / "latency-manifest.json")["protocol_sha256"] == sha(OUT / "protocol.json")
    original = json.loads(gzip.decompress((ROOT / "evidence/controller-v2/rankings.json.gz").read_bytes()))["test"]
    lookup = {(r["dataset"], r["query_id"], r["method"]): r for r in original}
    total = 0
    for directory in (OUT, OUT / "initial-candidate"):
        summary = read(directory / "latency-summary.json")
        rows = json.loads(gzip.decompress((directory / "latency-rankings.json.gz").read_bytes()))
        keys = [(r["dataset"], r["query_id"], r["method"]) for r in rows]
        assert len(set(keys)) == len(keys) == 2 * 3677
        for row in rows:
            assert len(set(row["ranked_ids"])) == len(row["ranked_gains"]) == 10
            close(sum(g / math.log2(i + 2) for i, g in enumerate(row["ranked_gains"])) / max(row["ideal"], 1e-12), row["ndcg10"])
            if row["method"] == "runtime_policy":
                assert row["ranked_ids"] == lookup[row["dataset"], row["query_id"], "gated"]["ranked_ids"]
            total += 1
        for name, result in summary.items():
            selected = [r for r in rows if r["dataset"] == name and r["method"] == "accelerated_fusion"]
            mismatch = sum(r["ranked_ids"] != lookup[name, r["query_id"], "fusion"]["ranked_ids"] for r in selected)
            assert mismatch == result["top10_mismatches"]
            assert result["accelerated_approved"] == (mismatch == 0)
            close(statistics.mean(r["ndcg10"] for r in selected), result["accelerated_fusion_ndcg"])
        observations = read(directory / "latency-observations.json")
        counts = Counter((r["dataset"], r["method"]) for r in observations)
        assert len(counts) == 25 and set(counts.values()) == {60}
        assert all(math.isfinite(r["milliseconds"]) and r["milliseconds"] > 0 for r in observations)
        if directory.name == "initial-candidate":
            assert summary["arguana"]["top10_mismatches"] > 0
            initial = read(directory / "latency-manifest.json")
            assert initial["source_sha256"]["context_stamps/efficient_reranker.py"] == sha(directory / "efficient-reranker-source.txt")
    training = read(OUT / "router-training.json")
    router = read(OUT / "router.json")
    assert router["approved_scopes"] == [n for n, g in training["gates"].items() if g["approved"]]
    assert not router["approved_scopes"], "update the evidence verifier when a future router qualifies"
    trials = [t for t in training["trials"] if t["eligible"]]
    best = max(trials, key=lambda t: (t["cheap"], t["threshold"]))
    assert best["threshold"] == training["selected_threshold"] == router["threshold"]
    ablation = read(OUT / "spherical-ablation.json")
    for method, count in ablation["results"].items():
        assert count == sum(row["chosen"][method] == row["expected"] for row in ablation["rows"])
    workflow, aggregate = read(OUT / "workflow-observations.json"), read(OUT / "workflow-summary.json")
    tool = read(OUT / "exact-tool-control.json")
    assert tool["correct"] == 48 and tool["model_calls"] == tool["model_tokens"] == 0
    assert len(tool["observations"]) == 48 and tool["malformed_rejected"] == 3
    assert all(r["correct"] and r["actual"] == r["expected"] for r in tool["observations"])
    for row in workflow:
        assert row["correct"] == (row["exact_schema"] and row["actual"] == row["expected"])
        assert row["model_calls"] in (0, 1)
        assert not row["reused"] or row["model_calls"] == 0 and row["correct"]
    for model, modes in aggregate.items():
        for mode, metrics in modes.items():
            selected = [r for r in workflow if r["model"] == model and r["mode"] == mode]
            assert len(selected) == metrics["requests"] == 48
            for metric, field in (("verified", "correct"), ("calls", "model_calls"), ("prompt_tokens", "prompt_tokens"), ("output_tokens", "output_tokens")):
                assert metrics[metric] == sum(r[field] for r in selected)
            close(metrics["total_ms"], sum(r["elapsed_ms"] for r in selected))
    print(f"runtime-v1: {total:,} ranking records, {len(workflow)} reader observations and 80 facet fixtures replayed")
    print("Recorded-gain replay, not independent raw-qrels, service-scale or base-model replication.")


if __name__ == "__main__":
    main()

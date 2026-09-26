"""Offline integrity and recorded-ranking replay; no raw data or GPU required."""

import gzip
import hashlib
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "evidence/controller-v2"


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def close(actual, expected):
    if not math.isclose(actual, expected, rel_tol=1e-6, abs_tol=1e-7):
        raise AssertionError((actual, expected))


def main():
    checksums = read(OUT / "checksums.json")
    for relative, expected in checksums.items():
        if sha(OUT / relative) != expected:
            raise AssertionError("artifact changed: " + relative)
    manifest, config = read(OUT / "manifest.json"), read(OUT / "run-config.json")
    for relative, expected in manifest["source_sha256"].items():
        if sha(ROOT / relative) != expected:
            raise AssertionError("training source changed: " + relative)
    assert manifest["run_config_sha256"] == sha(OUT / "run-config.json")
    assert config["protocol_sha256"] == sha(OUT / "protocol.json")
    assert config["data_manifest_sha256"] == sha(OUT / "data-manifest.json")
    assert manifest["selection_sha256"] == sha(OUT / "selection.json")
    data = read(OUT / "data-manifest.json")
    for sources in (data["source_sha256"], read(OUT / "online-runtime.json")["source_sha256"],
                    read(OUT / "split-quantization.json")["source_sha256"]):
        for relative, expected in sources.items():
            assert sha(ROOT / relative) == expected, relative
    assert "train" not in data["assignments"]["fiqa"]
    for name, partitions in data["assignments"].items():
        all_ids = [qid for ids in partitions.values() for qid in ids]
        assert len(all_ids) == len(set(all_ids)), name
    trials = read(OUT / "training.json")
    assert len(trials) == 9
    assert len({r["trial"] for r in trials}) == 9
    for trial in trials:
        assert sha(OUT / ("models/" + trial["trial"] + ".safetensors")) == trial["checkpoint_sha256"]
        best = max(trial["history"], key=lambda r: r["tune_ndcg10"])
        close(best["tune_ndcg10"], trial["tune_ndcg10"])
        assert best["epoch"] == trial["selected_epoch"]
        assert [r["epoch"] for r in trial["history"]] == list(range(13))
    selected = max(trials, key=lambda r: r["tune_ndcg10"])["trial"]
    selection = read(OUT / "selection.json")
    assert selected == selection["selected_student"] == manifest["selected_student"]
    payload = json.loads(gzip.decompress((OUT / "rankings.json.gz").read_bytes()))
    quantized = json.loads(gzip.decompress((OUT / "quantized-rankings.json.gz").read_bytes()))
    split_quantized = json.loads(gzip.decompress((OUT / "split-quantization-rankings.json.gz").read_bytes()))
    grouped = defaultdict(list)
    seen = set()
    total = 0
    for split in ("test", "tune", "calibration", "int8", "split_int8"):
        records = split_quantized if split == "split_int8" else quantized if split == "int8" else payload[split]
        for row in records:
            key = (split, row["dataset"], row["method"], row["query_id"])
            assert key not in seen, key
            seen.add(key)
            gains = row["ranked_gains"]
            assert len(gains) == len(row["ranked_ids"]) == 10
            assert len(set(row["ranked_ids"])) == 10
            assert row["query_id"] not in row["ranked_ids"]
            assert all(math.isfinite(g) and g >= 0 for g in gains)
            close(sum(g / math.log2(i + 2) for i, g in enumerate(gains)) / max(row["ideal"], 1e-12), row["ndcg10"])
            close(sum(g > 0 for g in gains) / max(1, row["total_relevant"]), row["recall10"])
            close(next((1 / (i + 1) for i, g in enumerate(gains) if g > 0), 0), row["mrr10"])
            assert 0 <= row["ndcg10"] <= 1.000001
            grouped[(split, row["dataset"], row["method"])].append(row)
            total += 1
    summary = read(OUT / "summary.json")
    for name, methods in summary.items():
        for method, metrics in methods.items():
            records = grouped[("test", name, method)]
            assert len(records) == manifest["query_counts"][name]
            for metric in ("ndcg10", "recall10", "mrr10"):
                close(statistics.mean(r[metric] for r in records), metrics[metric])
            for baseline in ("dense", "hybrid", "teacher"):
                delta = metrics["delta_vs_" + baseline]
                close(metrics["ndcg10"] - methods[baseline]["ndcg10"], delta["mean"])
                assert delta["n"] == len(records)
        for method in manifest["methods"]:
            assert method in methods
    for name, decision in selection["decisions"].items():
        if "comparisons" in decision:
            candidate = decision["candidate"]
            means = {m: statistics.mean(r["ndcg10"] for r in grouped[("calibration", name, m)])
                     for m in ("dense", "hybrid", "teacher", candidate)}
            for baseline, bound in decision["comparisons"].items():
                close(means[candidate] - means[baseline], bound["mean"])
            passed = decision["queries"] >= 50 and all(m == candidate or b["lower"] > 0
                                                       for m, b in decision["comparisons"].items())
            assert passed == decision["qualified"]
            assert decision["deployed"] == (candidate if passed else "dense")
        else:
            assert decision["deployed"] == "dense"
        expected = sorted(grouped[("test", name, decision["deployed"])], key=lambda r: r["query_id"])
        actual = sorted(grouped[("test", name, "gated")], key=lambda r: r["query_id"])
        assert [r["ranked_ids"] for r in expected] == [r["ranked_ids"] for r in actual]
    coverage = read(OUT / "candidate-coverage.json")
    for row in payload["candidate_diagnostics"]:
        for metric in ("recall", "oracle_ndcg10"):
            assert 0 <= row["old_candidates"][metric] <= row["new_candidates"][metric] + 1e-8 <= 1.000001
    for name, result in coverage.items():
        records = [r for r in payload["candidate_diagnostics"] if r["dataset"] == name]
        assert len(records) == manifest["query_counts"][name]
        for which, metrics in result.items():
            for metric, expected in metrics.items():
                close(statistics.mean(r[which][metric] for r in records), expected)
    timing = read(OUT / "online-runtime.json")
    numerical = read(OUT / "split-quantization.json")
    for name, methods in numerical["fidelity"].items():
        for method, recorded in methods.items():
            rows = grouped[("split_int8", name, method)]
            assert len(rows) == manifest["query_counts"][name]
            close(statistics.mean(r["ndcg10"] for r in rows), recorded["ndcg10"])
        assert methods["split-int8"]["mean_abs_logit_error"] < methods["whole-int8"]["mean_abs_logit_error"]
    counts = Counter((r["dataset"], r["method"]) for r in timing["observations"])
    assert len(counts) == 5 * 6 and set(counts.values()) == {60}
    assert all(math.isfinite(r["milliseconds"]) and r["milliseconds"] > 0 for r in timing["observations"])
    print(f"controller-v2: {len(checksums)} artifact hashes; {total:,} ranking records replayed; frozen selection and gates verified")
    print("Offline replay checks recorded gains, not raw-corpus relevance or CUDA bitwise reproducibility.")


if __name__ == "__main__":
    main()

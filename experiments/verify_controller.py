"""Verify published controller and computation evidence without torch or datasets."""

import gzip
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    if path.name.endswith(".gz"):
        with gzip.open(path, "rt", encoding="utf-8") as stream:
            return json.load(stream)
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check(condition, message):
    if not condition:
        raise SystemExit(message)


def source(relative, expected):
    path = ROOT / relative
    snapshot = ROOT / "evidence/source-snapshots" / (path.stem + "-" + expected + ".py")
    check(sha(path) == expected or (snapshot.is_file() and sha(snapshot) == expected),
          "source differs without matching snapshot: " + relative)


def main():
    for name in ("controller-v1", "computation-v1"):
        folder = ROOT / "evidence" / name
        for key, expected in read(folder / "checksums.json").items():
            check(sha(folder / key) == expected, "artifact checksum: " + name + "/" + key)
        manifest = read(folder / "manifest.json")
        check(sha(folder / "protocol.json") == manifest["protocol_sha256"], "protocol changed")
        for relative, expected in manifest["source_sha256"].items():
            source(relative, expected)
    folder = ROOT / "evidence/controller-v1"
    manifest, summary = read(folder / "manifest.json"), read(folder / "summary.json")
    records = read(folder / "results.json.gz")
    groups, seen = defaultdict(list), set()
    for row in records:
        key = (row["dataset"], row["method"], row["query_id"])
        check(key not in seen, "duplicate public result")
        seen.add(key)
        check(len(row["ranked_ids"]) == len(set(row["ranked_ids"])) == 10, "invalid ranking")
        for metric in ("ndcg10", "recall10", "mrr10"):
            check(math.isfinite(row[metric]) and 0 <= row[metric] <= 1.000001, "invalid metric")
        if "ranked_gains" in row:
            dcg = math.fsum(float(v) / math.log2(i + 2) for i, v in enumerate(row["ranked_gains"]))
            ndcg = dcg / max(row["ideal"], 1e-12)
            check(math.isclose(ndcg, row["ndcg10"], abs_tol=3e-7), "ranking does not reproduce nDCG")
        groups[key[:2]].append(row)
    for name, count in manifest["query_counts"].items():
        for method in manifest["methods"]:
            rows = groups[(name, method)]
            check(len(rows) == count, "incomplete dataset/method")
            for metric in ("ndcg10", "recall10", "mrr10"):
                mean = math.fsum(r[metric] for r in rows) / count
                check(math.isclose(mean, summary[name][method][metric], abs_tol=1e-12), "aggregate differs")
            deltas = [a["ndcg10"] - b["ndcg10"] for a, b in zip(rows, groups[(name, "dense")])]
            check(math.isclose(math.fsum(deltas) / count, summary[name][method]["delta_vs_dense"]["mean"], abs_tol=1e-12),
                  "paired mean differs")
    training, selection = read(folder / "training.json"), read(folder / "selection.json")
    runtime = read(folder / "runtime.json")
    for kind, expected in runtime["checkpoint_sha256"].items():
        check(sha(folder / "checkpoints" / ("selected-" + kind + ".safetensors")) == expected, "checkpoint checksum differs")
    recurrent = next(r for r in training if r["trial"] == "recurrent-29")
    check(sha(folder / "checkpoints/recurrent-29.safetensors") == recurrent["checkpoint_sha256"], "recurrent checkpoint differs")
    winner = max(training, key=lambda r: r["tune_ndcg10"])
    check(winner["trial"] == selection["selected"] == manifest["selected"], "tune selection differs")
    for trial in training:
        best = max(trial["history"], key=lambda r: r["tune_ndcg10"])
        check(best["epoch"] == trial["selected_epoch"], "checkpoint selected outside tuning")
    for domain, decision in selection["per_scope"].items():
        route = "dense"
        if decision["neural_minus_dense"]["n"] >= 50:
            if decision["neural_minus_dense"]["lower"] > 0 and decision["neural_minus_hybrid"]["lower"] > 0:
                route = "selected"
            elif decision["hybrid_minus_dense"]["lower"] > 0:
                route = "hybrid"
        check(route == decision["route"], "calibration gate mismatch")
    for name in manifest["query_counts"]:
        route = selection["per_scope"].get(name, {"route": "dense"})["route"]
        method = selection["selected"] if route == "selected" else route
        for policy, expected in zip(groups[(name, "gated-policy")], groups[(name, method)]):
            check(policy["query_id"] == expected["query_id"] and policy["ranked_ids"] == expected["ranked_ids"], "policy bypassed its gate")
    supplemental = read(folder / "supplementary.json")
    parity = read(folder / "encoder-parity.json")
    source("experiments/audit_encoder_cache.py", parity["source_sha256"])
    check(len(parity["samples"]) == 4, "missing cache compatibility sample")
    for sample in parity["samples"]:
        check(sample["count"] == 16 and sample["minimum_cosine"] >= .9999
              and sample["maximum_coordinate_error"] <= .001, "encoder cache parity failed")
    source("experiments/audit_controller.py", supplemental["source_sha256"])
    supplement = read(folder / "supplementary-results.json.gz")
    for name, methods in supplemental["summary"].items():
        for method, value in methods.items():
            rows = [r for r in supplement if r["dataset"] == name and r["method"] == method]
            check(len(rows) == manifest["query_counts"][name], "incomplete supplementary evaluation")
            check(math.isclose(math.fsum(r["ndcg10"] for r in rows) / len(rows), value, abs_tol=1e-12), "supplement mean differs")
    folder = ROOT / "evidence/computation-v1"
    rows, summary = read(folder / "results.json"), read(folder / "summary.json")
    for mode, totals in summary.items():
        items = [r for r in rows if r["mode"] == mode]
        check(len(items) == totals["requests"], "workflow count differs")
        for r in items:
            value = json.loads(r["result"])
            correct = set(value) == {"value"} and type(value["value"]) is int and value["value"] == r["expected"]
            check(correct == r["correct"], "incorrect correctness label")
            if r["hit"]:
                check(r["event"] % 2 == 1 and r["prompt_tokens"] == r["output_tokens"] == 0,
                      "reuse on changed computation or model tokens attributed to hit")
        for field in ("correct", "prompt_tokens", "output_tokens"):
            check(sum(r[field] for r in items) == totals[field], "workflow total differs")
        check(sum(r["hit"] for r in items) == totals["hits"], "cache hits differ")
        check(math.isclose(math.fsum(r["milliseconds"] for r in items), totals["total_ms"], abs_tol=1e-8), "latency total differs")
    print(f"Verified {len(records)} public ranking records, {len(supplement)} precision/recurrence records and {len(rows)} local workflow observations.")


if __name__ == "__main__":
    main()

"""Offline integrity and arithmetic checks for hybrid-retrieval-v1 evidence."""

import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "evidence" / "hybrid-retrieval-v1"


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    checksums = json.loads((EVIDENCE / "checksums.json").read_text(encoding="utf-8"))
    for name, expected in checksums.items():
        actual = sha256(EVIDENCE / name)
        if actual != expected:
            raise SystemExit(f"checksum mismatch: {name}")

    protocol = json.loads((EVIDENCE / "protocol.json").read_text(encoding="utf-8"))
    manifest = json.loads((EVIDENCE / "manifest.json").read_text(encoding="utf-8"))
    if manifest["protocol_sha256"] != sha256(EVIDENCE / "protocol.json"):
        raise SystemExit("protocol digest mismatch")
    if protocol["semantic_weight"] != 0.75:
        raise SystemExit("unexpected frozen semantic weight")

    for relative, expected in manifest["source_sha256"].items():
        current = ROOT / relative
        snapshot = ROOT / "evidence" / "source-snapshots" / f"{Path(relative).stem}-{expected}.py"
        if sha256(current) != expected and (not snapshot.exists() or sha256(snapshot) != expected):
            raise SystemExit(f"missing matching source or snapshot: {relative}")

    records = json.loads((EVIDENCE / "results.json").read_text(encoding="utf-8"))
    summaries = json.loads((EVIDENCE / "summary.json").read_text(encoding="utf-8"))
    expected_queries = {item["dataset"]: item["queries"] for item in manifest["datasets"]}
    methods = {"dense_minilm", "bm25", "hybrid_z_075"}
    grouped = defaultdict(list)
    seen = set()
    for record in records:
        key = (record["dataset"], record["method"], record["query_id"])
        if key in seen or record["method"] not in methods:
            raise SystemExit("duplicate or unexpected result record")
        seen.add(key)
        if len(record["ranked_ids"]) != len(set(record["ranked_ids"])) or len(record["ranked_ids"]) > 10:
            raise SystemExit("invalid ranked IDs")
        grouped[(record["dataset"], record["method"])].append(record)

    if len(records) != 3 * sum(expected_queries.values()):
        raise SystemExit("unexpected result count")
    for dataset, query_count in expected_queries.items():
        for method in methods:
            if len(grouped[(dataset, method)]) != query_count:
                raise SystemExit(f"incomplete results: {dataset}/{method}")

    by_dataset = {item["dataset"]: item for item in summaries}
    for dataset, summary in by_dataset.items():
        for method in methods:
            rows = grouped[(dataset, method)]
            for metric in ("ndcg10", "recall10", "mrr10"):
                mean = math.fsum(row[metric] for row in rows) / len(rows)
                if not math.isclose(mean, summary["methods"][method][metric], abs_tol=1e-12):
                    raise SystemExit(f"aggregate mismatch: {dataset}/{method}/{metric}")
        difference = (
            summary["methods"]["hybrid_z_075"]["ndcg10"]
            - summary["methods"]["dense_minilm"]["ndcg10"]
        )
        if not math.isclose(difference, summary["hybrid_minus_dense_ndcg10"], abs_tol=1e-12):
            raise SystemExit(f"difference mismatch: {dataset}")

    if not all(by_dataset[name]["hybrid_minus_dense_bootstrap95"][0] > 0
               for name in ("scifact", "nfcorpus", "arguana")):
        raise SystemExit("expected positive intervals are absent")
    if not by_dataset["scidocs"]["hybrid_minus_dense_bootstrap95"][1] < 0:
        raise SystemExit("prospective SciDocs regression is not preserved")
    print(f"verified {len(records)} hybrid retrieval records, including the SciDocs failure")


if __name__ == "__main__":
    main()

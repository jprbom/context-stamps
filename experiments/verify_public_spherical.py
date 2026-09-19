"""Verify stored public retrieval aggregates and provenance without corpus downloads."""

import hashlib
import json
import math
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "evidence/spherical-public-v1"


def read(name):
    return json.loads((OUT / name).read_text(encoding="utf-8"))


def main():
    for name, digest in read("checksums.json").items():
        assert hashlib.sha256((OUT / name).read_bytes()).hexdigest() == digest, name
    manifest = read("manifest.json")
    for name, digest in manifest["source_sha256"].items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == digest, name
    selected = max(read("validation.json"), key=lambda row: row["ndcg10"])
    assert selected["semantic_weight"] == manifest["semantic_weight"]
    records = read("results.json")
    assert len(records) == (300 + 323 + 1406) * 3
    for row in records:
        assert math.isfinite(row["ndcg10"]) and 0 <= row["ndcg10"] <= 1.0000000001
        assert len(set(row["ranked_ids"])) == len(row["ranked_ids"]) == 10
        assert row["query_id"] not in row["ranked_ids"]
    for summary in read("summary.json"):
        group = [r for r in records if (r["dataset"], r["method"]) == (summary["dataset"], summary["method"])]
        assert len(group) == summary["queries"]
        assert len({r["query_id"] for r in group}) == len(group)
        assert abs(statistics.mean(r["ndcg10"] for r in group) - summary["ndcg10"]) < 1e-12
    print("Verified 6087 public retrieval records, aggregates, self-exclusion and provenance.")
    print("Full ranking/metric replay requires the external corpora and pinned embedding caches.")


if __name__ == "__main__":
    main()

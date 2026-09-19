"""Offline integrity and aggregation checks for the committed evidence."""

import hashlib
import json
from pathlib import Path


def verify():
    root = Path(__file__).resolve().parents[1]
    for directory, record_name in [("synthetic-v1", "per-case.jsonl"), ("scifact-v1", "per-query.jsonl")]:
        folder = root / "evidence" / directory
        manifest = json.loads((folder / "manifest.json").read_text())
        for name, expected in manifest["source_sha256"].items():
            actual = hashlib.sha256((root / name).read_bytes()).hexdigest()
            if actual != expected:
                raise ValueError(f"source differs from recorded experiment: {name}")
        records = [json.loads(line) for line in (folder / record_name).read_text().splitlines()]
        summary = json.loads((folder / "summary.json").read_text())
        for row in summary:
            group = [
                r for r in records if r["method"] == row["method"] and r.get("split") == row.get("split")
            ]
            metric = "success" if directory.startswith("synthetic") else "ndcg10"
            if (
                len(group) != row["n"]
                or abs(sum(r[metric] for r in group) / len(group) - row[metric]) > 1e-10
            ):
                raise ValueError("summary does not reproduce per-case measurements")
        if directory.startswith("synthetic"):
            actual = hashlib.sha256((folder / "fixtures.jsonl").read_bytes()).hexdigest()
            if actual != manifest["data_sha256"]:
                raise ValueError("fixture hash mismatch")
        else:
            groups = [set(manifest[k]) for k in ["training_queries", "validation_queries", "test_queries"]]
            if any(groups[i] & groups[j] for i in range(3) for j in range(i)):
                raise ValueError("query split overlap")
        print(f"{directory}: source hashes, splits and recorded aggregates verified")


if __name__ == "__main__":
    verify()

"""Offline consistency replay for the public precision-residual evidence."""

import argparse
import json
from pathlib import Path


def verify(directory, cohort):
    frozen = json.loads((directory / "validation.json").read_text(encoding="utf-8"))
    result = json.loads((directory / "results.json").read_text(encoding="utf-8"))
    analysis = json.loads((directory / "analysis.json").read_text(encoding="utf-8"))
    manifest = json.loads(cohort.read_text(encoding="utf-8"))
    assert result["validation"] == frozen["validation"]
    assert result["validation_gate"] == frozen["validation_gate"]
    assert result["validation_gate"]["pass"]
    assert result["validation_gate"]["selected"] == "pq32_sq8_top20"
    assert [r["repo"] for r in result["test"]["rows"]] == manifest["repositories"]
    assert result["test_vectors_sha256"] == json.loads(
        (directory / "latency-profile-1thread.json").read_text(encoding="utf-8"))["test_vectors_sha256"]
    for split in ("validation", "test"):
        records = result[split]["rows"]
        assert len(records) == analysis["splits"][split]["repositories"]
        assert sum(r["candidates"] for r in records) == analysis["splits"][split]["candidates"]
        if split == "test":
            assert sum(len(r["ranks"]["dense"]) for r in records) == manifest["queries"]
            assert sum(r["candidates"] for r in records) == manifest["candidates"]
        for route, summary in result[split]["summary"].items():
            ranks = [rank for record in records for rank in record["ranks"][route]]
            assert len(ranks) == summary["n"]
            assert all(1 <= rank <= 21 for rank in ranks)
            assert sum(rank == 1 for rank in ranks) == summary["top1"]
            assert sum(rank <= 10 for rank in ranks) == summary["top10"]
            mrr = round(sum(1 / rank for rank in ranks if rank <= 20) / len(ranks), 6)
            assert mrr == summary["mrr_at_20"]
            expected_bytes = (analysis["splits"][split]["candidates"] * result["code_sizes"][route]
                              + result["shared_bytes"][route])
            assert expected_bytes == analysis["splits"][split]["routes"][route]["index_bytes"]
    print("Residual evidence replay passed: 30 validation and 40 test repositories")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--directory", type=Path, required=True)
    p.add_argument("--cohort", type=Path, required=True)
    a = p.parse_args()
    verify(a.directory, a.cohort)

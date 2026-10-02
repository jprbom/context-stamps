"""Freeze a second, uninspected CodeSearchNet Python repository cohort.

The raw candidate text stays in a private cache outside Git. Only hashes and
repository identities are published. Copyright (c) 2026 Prashant Jagtap.
"""

import argparse
import json
from pathlib import Path

from codesearchnet_quantization import MODEL, REVISION, SOURCE_SHA, cohort, digest, file_sha, rows


def freeze(source, previous_manifest, cache, out):
    previous = json.loads(previous_manifest.read_text(encoding="utf-8"))
    excluded = set(previous["splits"]["test"]["selected_repositories"])
    candidates = cohort(rows(source / "python-test.parquet"), "matching-test-v1", max_repos=1000)
    print(f"eligible={len(candidates)} previously_inspected_overlap={sum(item['repo'] in excluded for item in candidates)}", flush=True)
    selected = [item for item in candidates if item["repo"] not in excluded][:40]
    if len(selected) != 40:
        raise ValueError("not enough unseen eligible repositories")
    train_repos = {row["repo"] for row in json.loads((cache / "train.json").read_text(encoding="utf-8"))}
    val_repos = {item["repo"] for item in json.loads((cache / "validation.json").read_text(encoding="utf-8"))}
    if {item["repo"] for item in selected} & (train_repos | val_repos | excluded):
        raise ValueError("cohort overlaps an inspected or training repository")
    private = cache / "matching-test-v1.json"
    private.write_text(json.dumps(selected, ensure_ascii=False), encoding="utf-8")
    manifest = {
        "task": "derived repository-local exact-function localization",
        "dataset": "code-search-net/code_search_net Python test split",
        "source_sha256": SOURCE_SHA["test"],
        "prepared_sha256": file_sha(private),
        "selection": "SHA-256 sort with label matching-test-v1; exclude first-study test repositories; first 40 eligible repositories; three unique-description queries per repository",
        "selection_seed": 20260930,
        "excluded_first_study_test_repositories_sha256": digest("\n".join(sorted(excluded))),
        "model": f"{MODEL}@{REVISION}",
        "repositories": [item["repo"] for item in selected],
        "queries": sum(len(item["targets"]) for item in selected),
        "candidates": sum(len(item["items"]) for item in selected),
        "first_study_validation_source_sha256": previous["splits"]["validation"]["source_sha256"],
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: manifest[k] for k in ("source_sha256", "prepared_sha256", "queries", "candidates")}, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--previous-manifest", type=Path, required=True)
    p.add_argument("--cache", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    freeze(a.source, a.previous_manifest, a.cache, a.out)

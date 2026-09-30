"""Replay the published RepoQA-derived aggregate and split invariants.

Copyright (c) 2026 Prashant Jagtap. MIT License.
"""

import argparse
import json
from pathlib import Path

SOURCE_SHA256 = "c050a2ad90a7df89d9dc1f1c3b3b20683edd20a56293b35fcaae43dec115d681"
SPLITS = {
    "train": ("psf/black", "python-poetry/poetry", "locustio/locust", "pyg-team/pytorch_geometric"),
    "validation": ("openai/openai-python", "mlc-ai/mlc-llm", "reactive-python/reactpy"),
    "final": ("marshmallow-code/marshmallow", "ethereum/web3.py", "Ciphey/Ciphey"),
}
BITS = {"semantic": 128, "task": 64, "entity": 32, "relation": 32}


def verify(path):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    assert data["source"]["sha256"] == SOURCE_SHA256
    assert data["splits"] == {key: list(value) for key, value in SPLITS.items()}
    assert data["stamp_bits"] == BITS and sum(BITS.values()) == 256
    expected = {repo: split for split, repos in SPLITS.items() for repo in repos}
    assert len(expected) == 10
    assert {r["repo"] for r in data["index"]} == set(expected)
    assert sum(r["functions"] for r in data["index"]) == 11164
    assert all(r["stamp_payload_bytes"] == r["functions"] * 32 for r in data["index"])
    records = data["records"]
    assert len(records) == 100 and len({r["id"] for r in records}) == 100
    for row in records:
        assert row["split"] == expected[row["repo"]]
        assert row["id"].startswith(row["repo"] + ":")
        assert 0 <= row["target"] < row["candidates"]
        for method in ("dense", "tfidf", "stamp"):
            rank = row[f"{method}_rank"]
            assert isinstance(rank, int) and 1 <= rank <= row["candidates"]
            top = row[f"{method}_top10"]
            assert len(top) == min(10, row["candidates"])
            assert len(set(top)) == len(top)
            assert all(isinstance(i, int) and 0 <= i < row["candidates"] for i in top)
            assert (row["target"] in top) == (rank <= 10)
            if rank <= 10:
                assert top[rank - 1] == row["target"]
    for split in SPLITS:
        group = [r for r in records if r["split"] == split]
        assert len(group) == data["summary"][split]["queries"] == len(SPLITS[split]) * 10
        for method in ("dense", "tfidf", "stamp"):
            ranks = [r[f"{method}_rank"] for r in group]
            reported = data["summary"][split]["methods"][method]
            assert reported["top1"] == sum(x == 1 for x in ranks)
            assert reported["top5"] == sum(x <= 5 for x in ranks)
            assert reported["top10"] == sum(x <= 10 for x in ranks)
            assert reported["mrr"] == round(sum(1 / x for x in ranks) / len(ranks), 6)
    followup_path = Path(path).with_name("quantizer-followup.json")
    if followup_path.exists():
        followup = json.loads(followup_path.read_text(encoding="utf-8"))
        assert followup["source_sha256"] == SOURCE_SHA256
        assert followup["train_repositories"] == list(SPLITS["train"])
        assert followup["validation_repositories"] == list(SPLITS["validation"])
        assert followup["training_rows"] == 4000
        assert len(followup["rows"]) == 30
        assert {r["id"] for r in followup["rows"]} == {
            r["id"] for r in records if r["split"] == "validation"
        }
        for method in ("dense", "gaussian256", "itq256"):
            ranks = [r["ranks"][method] for r in followup["rows"]]
            assert all(isinstance(rank, int) and rank > 0 for rank in ranks)
            metrics = followup["summary"][method]
            assert metrics["top1"] == sum(rank == 1 for rank in ranks)
            assert metrics["top10"] == sum(rank <= 10 for rank in ranks)
            assert metrics["mrr"] == round(sum(1 / rank for rank in ranks) / len(ranks), 6)
        assert {r["id"]: r["ranks"]["dense"] for r in followup["rows"]} == {
            r["id"]: r["dense_rank"] for r in records if r["split"] == "validation"
        }
    ablation_path = Path(path).with_name("quantization-ablation.json")
    if ablation_path.exists():
        ablation = json.loads(ablation_path.read_text(encoding="utf-8"))
        assert ablation["source_sha256"] == SOURCE_SHA256
        assert ablation["training_repositories"] == list(SPLITS["train"])
        assert ablation["validation_repositories"] == list(SPLITS["validation"])
        assert ablation["pq_config"]["code_bytes"] == 32
        assert len(ablation["rows"]) == 30
        assert len(ablation["warm_lookup_timings"]) == 3
        assert {r["id"] for r in ablation["rows"]} == {
            r["id"] for r in records if r["split"] == "validation"
        }
        for method in ("dense", "itq224_task32", "pq32"):
            ranks = [r["ranks"][method] for r in ablation["rows"]]
            assert all(isinstance(rank, int) and rank > 0 for rank in ranks)
            metrics = ablation["summary"][method]
            assert metrics["top1"] == sum(rank == 1 for rank in ranks)
            assert metrics["top10"] == sum(rank <= 10 for rank in ranks)
            assert metrics["mrr"] == round(sum(1 / rank for rank in ranks) / len(ranks), 6)
        assert {r["id"]: r["ranks"]["dense"] for r in ablation["rows"]} == {
            r["id"]: r["dense_rank"] for r in records if r["split"] == "validation"
        }
    return dict(repositories=10, tasks=len(records), functions=sum(r["functions"] for r in data["index"]),
                replay="passed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path, nargs="?", default=Path(__file__).resolve().parents[1] /
                        "evidence/repoqa-localization-v1/results.json")
    print(json.dumps(verify(parser.parse_args().path)))

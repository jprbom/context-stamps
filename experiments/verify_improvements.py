"""Offline verification of retrieval repair, fixed-bit training and workflow evidence."""

import hashlib
import json
import statistics
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from context_stamps.contracts import render_integer_assignments
from context_stamps.training import fit_pairwise
from experiments.run_workflow_efficiency import grade
from experiments.train_stamp256 import ENCODER, NAMES, codec, evaluate, features, prepare_codes

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def historical_source(filename, digest):
    filename = filename.replace("\\", "/")
    current = ROOT / filename
    if current.is_file() and hashlib.sha256(current.read_bytes()).hexdigest() == digest:
        return current
    return ROOT / "evidence/source-snapshots" / f"{Path(filename).stem}-{digest}{Path(filename).suffix}"


def verify():
    directories = ["residual-v1", "residual-v2", "stamp256-v1", "scale-comparison-v1", *[f"workflow-efficiency-v{i}" for i in range(1, 5)]]
    for directory in directories:
        folder = ROOT / "evidence" / directory
        for filename, digest in read(folder / "checksums.json").items():
            assert hashlib.sha256((folder / filename).read_bytes()).hexdigest() == digest, (directory, filename)
        manifest = read(folder / "manifest.json")
        for key in ("source_sha256", "input_sha256"):
            for filename, digest in manifest.get(key, {}).items():
                assert hashlib.sha256(historical_source(filename, digest).read_bytes()).hexdigest() == digest, filename
    for directory in ("residual-v1", "residual-v2"):
        folder = ROOT / "evidence" / directory
        records = read(folder / "results.json")
        assert len(records) == 2029 * 4
        gold = {(r["dataset"], r["query_id"]): r for r in records if r["method"] == "dense_float64_reference"}
        for row in records:
            assert row["query_id"] not in row["ranked_ids"]
            assert row["same_order_as_dense"] == (row["ranked_ids"] == gold[(row["dataset"], row["query_id"])]["ranked_ids"])
            if row["method"] == "residual_int8_exact":
                assert row["same_order_as_dense"]
                assert row["ndcg10"] == gold[(row["dataset"], row["query_id"])]["ndcg10"]
        for summary in read(folder / "summary.json"):
            group = [r for r in records if (r["dataset"], r["method"]) == (summary["dataset"], summary["method"])]
            assert len(group) == summary["queries"]
            assert abs(statistics.mean(r["ndcg10"] for r in group) - summary["ndcg10"]) < 1e-12
            assert abs(statistics.mean(r["refined_fraction"] for r in group) - summary["mean_refined_fraction"]) < 1e-12
        for row in read(folder / "scale.json"):
            assert row["all_orders_equal_dense"]
            assert row["index_bytes"] == row["rows"] * 432
            assert row["backing_bytes"] == row["rows"] * 1536
    folder = ROOT / "evidence/stamp256-v1"
    cases, trials = read(folder / "fixtures.json"), read(folder / "training.json")
    manifest = read(folder / "manifest.json")
    protocol, selected = manifest["protocol"], manifest["selected_widths"]
    def criterion(widths):
        rows = [r for r in trials if r["widths"] == widths]
        return statistics.mean(r["validation_top1"] for r in rows), statistics.mean(r["validation_mrr"] for r in rows)
    assert selected == max(protocol["allocations"], key=criterion)
    records = read(folder / "results.json")
    for seed in protocol["seeds"]:
        codes = prepare_codes(cases, seed)
        models = {}
        for widths in protocol["allocations"]:
            train = features(cases["train"], codes, widths)
            pairs = [values[row["positive"]] - value for row, values in zip(cases["train"], train)
                     for j, value in enumerate(values) if j != row["positive"]]
            template = codec(widths, seed).encode({n: ENCODER.encode(cases["train"][0]["query"][n]) for n in NAMES})
            model, history = fit_pairwise(template, pairs, penalty=.001)
            models[tuple(widths)] = model
            expected = next(r for r in trials if r["seed"] == seed and r["widths"] == widths)
            np.testing.assert_allclose(model.coefficients, expected["coefficients"], atol=1e-10, rtol=0)
            np.testing.assert_allclose([r["loss"] for r in history], [r["loss"] for r in expected["loss"]], atol=1e-10, rtol=0)
            validation = evaluate(cases["validation"], features(cases["validation"], codes, widths), model.coefficients)
            assert statistics.mean(r["top1"] for r in validation) == expected["validation_top1"]
        for split in ("fresh_test", "fresh_ood"):
            for method, widths, weights in (("equal_bits_uniform", [64] * 4, np.ones(4)),
                    ("equal_bits_pairwise", [64] * 4, models[(64, 64, 64, 64)].coefficients),
                    ("validation_selected_bits_pairwise", selected, models[tuple(selected)].coefficients)):
                actual = evaluate(cases[split], features(cases[split], codes, widths), weights)
                saved = [r for r in records if (r["seed"], r["split"], r["method"]) == (seed, split, method)]
                assert [r["order"] for r in actual] == [r["order"] for r in saved]
    for version in range(1, 5):
        folder = ROOT / f"evidence/workflow-efficiency-v{version}"
        records, setups = read(folder / "results.json"), read(folder / "setup.json")
        assert len(records) == 192
        assert len({(r["workflow"], r["repeat"], r["stage"], r["method"]) for r in records}) == 192
        for row in records:
            output = row["output"]
            if version >= 3 and row["stage"] == 1 and output:
                values = json.loads(output)
                code = render_integer_assignments({"SETTING": values["setting"], "LIMIT": values["limit"]},
                                                  allowed_names=("SETTING", "LIMIT"))
                assert code == row["rendered_code"]
                output = json.dumps({"code": code})
            correct = row["abstained"] if row["should_abstain"] else (not row["abstained"] and grade(output, row["expected"], row["stage"] == 1))
            assert correct == row["correct"]
            if version == 4:
                assert row["transport_bytes"] == (32 if row["method"] == "spherical_guarded" else 0)
        for summary in read(folder / "summary.json"):
            group = [r for r in records if r["method"] == summary["method"]]
            assert summary["input_tokens_total"] == sum(r["prompt_tokens"] for r in group)
            assert summary["output_tokens_total"] == sum(r["output_tokens"] for r in group)
            assert summary["correct_stages"] == sum(r["correct"] for r in group)
            assert summary["complete_workflow_success"] == sum(all(r["correct"] for r in group if (r["workflow"], r["repeat"]) == (case, repeat))
                for case in range(8) for repeat in range(2))
            total = sum(r["end_to_end_ms"] for r in group) + sum(s["setup_ms"] for s in setups if s["method"] == summary["method"])
            assert abs(total / 16 - summary["mean_workflow_ms_including_setup"]) < 1e-9
    scale = read(ROOT / "evidence/scale-comparison-v1/results.json")
    assert len(scale) == 540
    for summary in read(ROOT / "evidence/scale-comparison-v1/summary.json"):
        group = [r for r in scale if (r["rows"], r["method"], r["concurrency"]) == (summary["rows"], summary["method"], summary["concurrency"])]
        assert len(group) == 30
        assert statistics.mean(r["same_order"] for r in group) == summary["same_order_fraction"]
        assert abs(statistics.median(r["ms"] for r in group) - summary["median_ms"]) < 1e-12
    stats = read(ROOT / "evidence/workflow-efficiency-v4/paired-statistics.json")
    for name, digest in stats["source_sha256"].items():
        assert hashlib.sha256(historical_source(name, digest).read_bytes()).hexdigest() == digest
    print("Verified dense ranking parity,21 allocation training runs,3600 stamp rankings,768 workflow stages and540 scale calls.")
    print("Public corpus rank replay and generation require external data/runtime; timings are observations.")


if __name__ == "__main__":
    verify()

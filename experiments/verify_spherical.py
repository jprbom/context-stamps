"""Offline hash, replay, model-retraining and grading verification."""

import hashlib
import json
import statistics
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from context_stamps.facet_model import FacetModel
from context_stamps.guarded_activation import activate_constrained
from experiments.run_spherical import evaluate, prepare

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def verify():
    replay_count = 0
    guarded_rows = []
    for name in ("spherical-v1", "spherical-v2", "spherical-audit-v1", "spherical-guarded-v1"):
        folder = ROOT / "evidence" / name
        for filename, expected in read(folder / "checksums.json").items():
            assert hashlib.sha256((folder / filename).read_bytes()).hexdigest() == expected, filename
        manifest = read(folder / "manifest.json")
        for key in ("source_sha256", "input_sha256"):
            for filename, expected in manifest.get(key, {}).items():
                assert hashlib.sha256((ROOT / filename).read_bytes()).hexdigest() == expected, filename
        if name in {"spherical-v1", "spherical-v2"}:
            cases = read(folder / "fixtures.json")
            records = read(folder / "retrieval.json")
            for seed in (17, 41, 83):
                model = FacetModel.from_json((folder / f"facet-model-{seed}.json").read_text())
                train = prepare(cases["train"], seed, "views")
                features, labels = [], []
                for case, query, candidates in train:
                    features.extend(list(query.compare(c).values()) for c in candidates)
                    labels.extend(int(i == case["positive"]) for i in range(len(candidates)))
                trials = []
                for penalty in (.01, 1.0, 100.0):
                    fitted = FacetModel.fit(train[0][1], features, labels, penalty=penalty)
                    val = evaluate(prepare(cases["validation"], seed, "views"), fitted)
                    trials.append((statistics.mean(r["top1"] for r in val),
                                   statistics.mean(r["rr"] for r in val), fitted))
                restored = max(trials, key=lambda row: row[:2])[2]
                assert np.allclose(model.coefficients, restored.coefficients, rtol=0, atol=1e-10)
                assert abs(model.intercept - restored.intercept) < 1e-10
                for split in ("test", "ood"):
                    for method, mode in (("single_content_256", "single"), ("directions_4x64", "directions"),
                                         ("views_4x64_uniform", "views"), ("views_4x64_learned", "views")):
                        data = prepare(cases[split], seed, mode)
                        actual = evaluate(data, model if method.endswith("learned") else None)
                        saved = [r for r in records if (r["seed"], r["split"], r["method"]) == (seed, split, method)]
                        assert len(actual) == len(saved) == 80
                        for a, b in zip(actual, saved):
                            assert (a["case"], a["order"], a["top1"], a["rr"]) == (
                                b["case"], b["order"], b["top1"], b["rr"])
                            assert np.allclose(a["scores"], b["scores"], rtol=0, atol=1e-10)
                        replay_count += len(actual)
                    if name == "spherical-v2":
                        threshold = read(ROOT / "evidence/spherical-audit-v1/calibration.json")[str(seed)]["threshold"]
                        for case, query, candidates in prepare(cases[split], seed, "views"):
                            required = {k: case["query"][k] for k in ("entity", "intent", "task")}
                            hits = activate_constrained(query, candidates, metadata=case["candidates"],
                                                        required=required, threshold=threshold, limit=1, model=model)
                            # Regression: same cases previously exhibited wrong-entity activation.
                            assert [r["index"] for r in hits] == [case["positive"]]
                            missing = [c for i, c in enumerate(candidates) if i != case["positive"]]
                            metadata = [c for i, c in enumerate(case["candidates"]) if i != case["positive"]]
                            assert not activate_constrained(query, missing, metadata=metadata, required=required,
                                                            threshold=threshold, model=model)
                            guarded_rows.append((case["id"], seed))
        else:
            rows = read(folder / "slm-results.json")
            unique = {(r["case"], r["mode"], r["repetition"]) for r in rows}
            assert len(unique) == len(rows)
            for row in rows:
                try:
                    correct = json.loads(row["output"]) == row["expected"]
                except ValueError:
                    correct = False
                assert correct == row["correct"]
            for summary in read(folder / "slm-summary.json"):
                group = [r for r in rows if r["mode"] == summary["mode"]]
                assert len(group) == summary["calls"]
                assert abs(statistics.mean(r["correct"] for r in group) - summary["correct"]) < 1e-12
                for key in ("prompt_tokens", "end_to_end_ms", "request_ms", "assembly_ms"):
                    assert abs(statistics.mean(r[key] for r in group) - summary[key]) < 1e-10
            for row in read(folder / "activation.json"):
                assert row["recall"] == int(row["positive"] in row["activated"])
                assert row["false_activations"] == len(set(row["activated"]) - {row["positive"]})
                assert row["no_answer_false_activation"] == bool(row["false_activations"])
    print(f"Verified hashes, retrained six models, replayed {replay_count} query/seed/method records.")
    print(f"Exact guard passed {len(guarded_rows)} positive and {len(guarded_rows)} missing-answer regressions.")


if __name__ == "__main__":
    verify()

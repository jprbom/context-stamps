"""Additional candidate evidence verification; CPU-only and no downloads."""

import hashlib
import json
import statistics
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from context_stamps.facet_model import FacetModel
from context_stamps.training import fit_pairwise
from experiments.run_spherical import evaluate, prepare
from experiments.run_structured import prospective, regression

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def historical_source(filename, digest):
    current = ROOT / filename.replace("\\", "/")
    if current.is_file() and hashlib.sha256(current.read_bytes()).hexdigest() == digest:
        return current
    return ROOT / "evidence/source-snapshots" / f"{Path(filename).stem}-{digest}{Path(filename).suffix}"


def main():
    structured = ROOT / "evidence/structured-v1"
    manifest = read(structured / "manifest.json")
    for name, sha in manifest["source_sha256"].items():
        assert hashlib.sha256(historical_source(name, sha).read_bytes()).hexdigest() == sha
    assert regression() == read(structured / "regression.json")
    fixtures, fresh = prospective(manifest["protocol"]["seed"])
    # JSON encodes the generator's (source, document) tuples as arrays.
    assert json.loads(json.dumps(fixtures)) == read(structured / "fixtures.json")
    assert fresh == read(structured / "prospective.json")
    for directory in ("pairwise-v1", "multimodal-v1", "multimodal-v2"):
        folder = ROOT / "evidence" / directory
        for name, sha in read(folder / "checksums.json").items():
            assert hashlib.sha256((folder / name).read_bytes()).hexdigest() == sha, name
        for name, sha in read(folder / "manifest.json")["source_sha256"].items():
            assert hashlib.sha256(historical_source(name, sha).read_bytes()).hexdigest() == sha, name
    folder = ROOT / "evidence/pairwise-v1"
    cases, records = read(folder / "fixtures.json"), read(folder / "results.json")
    for seed in (17, 41, 83):
        train = prepare(cases["train"], seed, "views")
        pairs = []
        for case, query, candidates in train:
            features = [list(query.compare(c).values()) for c in candidates]
            pairs.extend([a - b for a, b in zip(features[case["positive"]], value)]
                         for i, value in enumerate(features) if i != case["positive"])
        trials = []
        for penalty in (0.0, .001, .01):
            model, history = fit_pairwise(train[0][1], pairs, penalty=penalty)
            saved = next(r for r in read(folder / "training.json") if r["seed"] == seed and r["penalty"] == penalty)
            np.testing.assert_allclose([r["loss"] for r in history], [r["loss"] for r in saved["loss"]], atol=1e-10)
            validation = evaluate(prepare(cases["validation"], seed, "views"), model)
            trials.append((statistics.mean(r["top1"] for r in validation),
                           statistics.mean(r["rr"] for r in validation), model))
        model = max(trials, key=lambda row: row[:2])[2]
        saved = FacetModel.from_json((folder / f"model-{seed}.json").read_text(encoding="utf-8"))
        np.testing.assert_allclose(model.coefficients, saved.coefficients, atol=1e-10)
        ridge = FacetModel.from_json((ROOT / f"evidence/spherical-v2/facet-model-{seed}.json").read_text())
        for split in ("fresh_test", "fresh_ood"):
            data = prepare(cases[split], seed, "views")
            for name, scorer in (("uniform", None), ("ridge", ridge), ("monotone_pairwise", model)):
                actual = evaluate(data, scorer)
                expected = [r for r in records if (r["seed"], r["split"], r["method"]) == (seed, split, name)]
                assert len(actual) == len(expected) == 200
                assert [(r["order"], r["top1"], r["rr"]) for r in actual] == [
                    (r["order"], r["top1"], r["rr"]) for r in expected]
    for row in read(ROOT / "evidence/multimodal-v1/results.json"):
        assert row["finite_nonempty"] and row["prompt_identity"]
        assert row["paired_max_absolute_error"] is not None
    multimodal = read(ROOT / "evidence/multimodal-v2/results.json")
    assert len(multimodal) == 36
    for modality in ("image", "speech", "video"):
        for case in range(3):
            for seed in (11, 29):
                pair = [r for r in multimodal if (r["modality"], r["case"], r["seed"]) == (modality, case, seed)]
                assert len(pair) == 2
                assert {r["condition"] for r in pair} == {"direct_oracle_prompt", "spherical_route_and_resolve"}
                assert pair[0]["sha256"] == pair[1]["sha256"]
                assert pair[0]["input_tokens"] == pair[1]["input_tokens"]
                for row in pair:
                    assert row["finite_nonempty"] and row["nonconstant"] and row["prompt_identity"]
                    assert row["paired_hash_equal"] and row["paired_max_absolute_error"] == 0
                    assert abs(row["generation_ms"] + row["routing_ms"] - row["end_to_end_ms"]) < .001
    scale = read(ROOT / "evidence/packed-scale-v1/results.json")
    for name, sha in scale["source_sha256"].items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == sha
    print("Pairwise training losses, selected weights and 3600 rankings replayed; multimodal/scale hashes checked.")
    print("Raw generated media remains local; perceptual quality is not verified by this script.")
    print("All 390 structured fixtures regenerated and their packets replayed exactly.")


if __name__ == "__main__":
    main()

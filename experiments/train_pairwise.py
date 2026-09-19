"""Validation-tuned monotone ranking and fresh procedural confirmation."""

import hashlib
import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from context_stamps.facet_model import FacetModel
from context_stamps.training import fit_pairwise
from experiments.run_spherical import aggregate, evaluate, prepare, save
from experiments.run_spherical_v2 import counterbalanced

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "evidence/pairwise-v1"


def main():
    cases = json.loads((ROOT / "evidence/spherical-v2/fixtures.json").read_text())
    cases["fresh_test"] = counterbalanced("test", 200, 719381)
    cases["fresh_ood"] = counterbalanced("ood", 200, 819381)
    save(OUT / "fixtures.json", cases)
    training, results, summary = [], [], []
    for seed in (17, 41, 83):
        train = prepare(cases["train"], seed, "views")
        differences = []
        for case, query, candidates in train:
            values = [list(query.compare(candidate).values()) for candidate in candidates]
            positive = values[case["positive"]]
            differences.extend([a - b for a, b in zip(positive, value)]
                               for i, value in enumerate(values) if i != case["positive"])
        trials = []
        validation = prepare(cases["validation"], seed, "views")
        for penalty in (0.0, .001, .01):
            start = time.perf_counter_ns()
            model, history = fit_pairwise(train[0][1], differences, penalty=penalty)
            fit_ms = (time.perf_counter_ns() - start) / 1e6
            scores = aggregate(evaluate(validation, model))
            trials.append((scores["top1"], scores["rr"], model, penalty))
            training.append({"seed": seed, "penalty": penalty, "epochs": 400, "pairs": len(differences),
                             "validation": scores, "loss": history, "fit_ms": fit_ms})
        _, _, selected, penalty = max(trials, key=lambda r: r[:2])
        (OUT / f"model-{seed}.json").write_text(selected.to_json() + "\n", encoding="utf-8", newline="\n")
        ridge = FacetModel.from_json((ROOT / f"evidence/spherical-v2/facet-model-{seed}.json").read_text())
        for split in ("fresh_test", "fresh_ood"):
            data = prepare(cases[split], seed, "views")
            for method, model in (("uniform", None), ("ridge", ridge), ("monotone_pairwise", selected)):
                rows = evaluate(data, model)
                results.extend(dict(r, split=split, seed=seed, method=method) for r in rows)
                summary.append({"split": split, "seed": seed, "method": method, "queries": len(rows),
                                "top1": statistics.mean(r["top1"] for r in rows),
                                "mrr": statistics.mean(r["rr"] for r in rows), "selected_penalty": penalty})
    save(OUT / "training.json", training)
    save(OUT / "results.json", results)
    save(OUT / "summary.json", summary)
    paths = ["context_stamps/training.py", "context_stamps/facet_model.py", "experiments/train_pairwise.py",
             "experiments/run_spherical_v2.py", "experiments/run_spherical.py"]
    save(OUT / "manifest.json", {"device": "CPU", "source_sha256": {
        p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in paths}})
    save(OUT / "checksums.json", {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                 for p in OUT.glob("*.json") if p.name != "checksums.json"})


if __name__ == "__main__":
    main()

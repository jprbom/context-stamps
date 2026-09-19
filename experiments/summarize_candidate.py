"""Exploratory paired bootstrap: resample tasks, never seeds or repeated calls."""

import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def interval(values, seed=99173):
    array = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    samples = array[rng.integers(0, len(array), size=(5000, len(array)))].mean(axis=1)
    return {"tasks": len(array), "mean_difference": float(array.mean()),
            "paired_bootstrap_95": [float(x) for x in np.quantile(samples, [.025, .975])],
            "resamples": 5000, "seed": seed}


def main():
    records = json.loads((ROOT / "evidence/pairwise-v1/results.json").read_text(encoding="utf-8"))
    output = {"status": "exploratory analysis after inspecting results; descriptive procedural-task intervals"}
    for split in ("fresh_test", "fresh_ood"):
        differences = []
        for case in sorted({r["case"] for r in records if r["split"] == split}):
            rows = [r for r in records if r["case"] == case and r["split"] == split]
            means = {method: np.mean([r["top1"] for r in rows if r["method"] == method])
                     for method in ("ridge", "monotone_pairwise")}
            differences.append(means["monotone_pairwise"] - means["ridge"])
        output[split] = interval(differences)
    source = "experiments/summarize_candidate.py"
    output["source_sha256"] = {source: hashlib.sha256((ROOT / source).read_bytes()).hexdigest()}
    (ROOT / "evidence/pairwise-v1/paired-statistics.json").write_text(json.dumps(output, indent=2) + "\n",
                                                                    encoding="utf-8", newline="\n")
    folder = ROOT / "evidence/pairwise-v1"
    (folder / "checksums.json").write_text(json.dumps({p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in folder.glob("*.json") if p.name != "checksums.json"}, indent=2) + "\n",
        encoding="utf-8", newline="\n")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()

"""Exploratory paired workflow intervals; no tuning or selective run removal."""

import hashlib
import json
import statistics
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "evidence/workflow-efficiency-v4"


def main():
    records = json.loads((OUT / "results.json").read_text())
    setup = json.loads((OUT / "setup.json").read_text())
    methods = ("full_current", "exact_graph", "spherical_guarded")
    case_latency = {}
    for method in methods:
        case_latency[method] = [statistics.mean(
            sum(r["end_to_end_ms"] for r in records if (r["workflow"], r["repeat"], r["method"]) == (case, repeat, method))
            + next(s["setup_ms"] for s in setup if (s["workflow"], s["repeat"], s["method"]) == (case, repeat, method))
            for repeat in range(2)) for case in range(8)]
    rng = np.random.default_rng(8299)
    indices = rng.integers(0, 8, size=(10000, 8))
    contrasts = []
    for baseline in ("full_current", "exact_graph"):
        difference = np.array(case_latency["spherical_guarded"]) - np.array(case_latency[baseline])
        means = difference[indices].mean(axis=1)
        contrasts.append({"contrast": "spherical_guarded minus " + baseline,
                          "mean_workflow_latency_difference_ms": float(difference.mean()),
                          "exploratory_95pct_interval_ms": np.percentile(means, [2.5, 97.5]).tolist()})
    output = {"status": "post-hoc exploratory paired bootstrap, not a preregistered significance test",
              "unit": "8 unique workflows, averaging2 repeats before resampling", "seed": 8299,
              "resamples": 10000, "contrasts": contrasts,
              "source_sha256": {"experiments/summarize_improvements.py": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}}
    (OUT / "paired-statistics.json").write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8", newline="\n")
    (OUT / "checksums.json").write_text(json.dumps({p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in OUT.glob("*.json") if p.name != "checksums.json"}, indent=2) + "\n", encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()

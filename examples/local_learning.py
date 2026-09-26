"""Offline local-learning mechanics with explicitly simulated task/resources.

Run from an installed clone: python examples/local_learning.py
No SLM call, downloaded data, device-speed measurement or benchmark qualification.
"""

import json
import tempfile
from dataclasses import asdict
from pathlib import Path

from context_stamps.local_learning import LearningLimits, LocalLearningRegistry, PairedOutcome, revision
from context_stamps.local_policy import CellPolicy, PolicyObservation, fit_cell_policy


def fixture_report(directory):
    binding = revision({"domain": "fictional-two-cell", "model": "simulator-v1", "policy": "fixture-v1",
                        "verifier": "fixture-truth-v1", "device": "simulated", "runtime": "fixture-v1"})
    baseline = CellPolicy(binding, "full", ())
    training = tuple(PolicyObservation(f"train-{cell}-{i}", cell, action,
                                       action == "full" or cell == "direct", 10. if action == "full" else 2.)
                     for cell in ("direct", "dependent") for i in range(30) for action in ("full", "compact"))
    learned = fit_cell_policy(training, binding=binding, baseline_action="full", cost_cap=10.)
    shortcut = CellPolicy(binding, "full", (("dependent", "compact"), ("direct", "compact")))
    limits = LearningLimits("fixture_units", 10., 0., 0., 1000)
    registry = LocalLearningRegistry(str(Path(directory) / "local-learning.sqlite"), scope="fictional-two-cell",
                                     binding=binding, baseline=baseline.revision)
    rounds = []
    policies = {p.revision: p for p in (baseline, learned, shortcut)}
    try:
        for round_index, candidate in enumerate((learned, shortcut), 1):
            ids = tuple(f"holdout-{round_index}-{i}" for i in range(600))
            current = policies[registry.active_revision(binding=binding)]
            plan = registry.register(candidate.revision, evaluation_clusters=ids,
                                     training_clusters=tuple(sorted({r.cluster_id for r in training})), limits=limits)
            rows = []
            for i, cluster_id in enumerate(ids):
                cell = "direct" if i % 2 == 0 else "dependent"
                old = current.choose(cell, binding=binding, allowed_actions=("full", "compact"))
                new = candidate.choose(cell, binding=binding, allowed_actions=("full", "compact"))
                rows.append(PairedOutcome(cluster_id, revision({"cell": cell, "old": old, "new": new}),
                                          old == "full" or cell == "direct", new == "full" or cell == "direct",
                                          10. if old == "full" else 2., 10. if new == "full" else 2.,
                                          10., 1024))
            result = registry.finish(plan, tuple(rows))
            rounds.append({"plan": plan, "outcomes": [asdict(r) for r in rows], "result": result})
        active_before = registry.active_revision(binding=binding)
        registry.rollback(reason="host_reported_drift_fixture")
        return {"schema": 1, "evidence_class": "deterministic_simulation_not_benchmark",
                "resource_values": "Simulated fixture units, wall time and memory; no measured hardware benefit.",
                "training": [asdict(r) for r in training], "baseline": asdict(baseline), "learned": asdict(learned),
                "shortcut": asdict(shortcut), "rounds": rounds,
                "active_before_rollback": active_before,
                "active_after_rollback": registry.active_revision(binding=binding)}
    finally:
        registry.close()


def main():
    with tempfile.TemporaryDirectory() as directory:
        report = fixture_report(directory)
    print(json.dumps({"evidence_class": report["evidence_class"], "learned": report["learned"]["choices"],
                      "rounds": [{"eligible": r["result"]["eligible"], "reasons": r["result"]["reasons"]}
                                 for r in report["rounds"]],
                      "rolled_back": report["active_after_rollback"] == revision(report["baseline"])}, indent=2))


if __name__ == "__main__":
    main()

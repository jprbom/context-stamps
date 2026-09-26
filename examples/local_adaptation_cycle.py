"""Local fit, protected evaluation, monitoring and rollback: simulated data.

No model calls or device measurements. The fixture demonstrates the control
flow; it does not qualify a model, deployment, edge device or statistical IID
assumption. All costs, correctness, RAM and wall-time labels are simulated.
"""

import json
import tempfile
from dataclasses import asdict
from pathlib import Path

from context_stamps.local_learning import (
    EvaluationCohort,
    LearningLimits,
    LocalLearningRegistry,
    PairedOutcome,
    revision,
)
from context_stamps.local_monitor import LocalRegressionMonitor, MonitoredOutcome, MonitorLimits
from context_stamps.local_policy import CellPolicy, PolicyObservation, fit_cell_policy


def fixture_report(directory):
    binding = revision({"domain": "two-cell-fixture", "runtime": "simulation-v1", "verifier": "fixture-v1"})
    baseline = CellPolicy(binding, "full", ())
    training = tuple(PolicyObservation(f"train-{cell}-{i}", cell, action,
                                       action == "full" or cell == "direct", 10. if action == "full" else 2.)
                     for cell in ("direct", "dependent") for i in range(30) for action in ("full", "compact"))
    candidate = fit_cell_policy(training, binding=binding, baseline_action="full", cost_cap=10.)
    registry = LocalLearningRegistry(str(Path(directory)/"learning.sqlite"), scope="fixture",
                                     binding=binding, baseline=baseline.revision)
    monitor_path = str(Path(directory)/"monitor.sqlite")
    monitor = LocalRegressionMonitor(monitor_path, scope="fixture", binding=binding, limits=MonitorLimits())
    try:
        cohorts = tuple(EvaluationCohort(name, tuple(f"{name}-{i}" for i in range(800)))
                        for name in ("adaptation", "retention"))
        plan = registry.register(candidate.revision, evaluation_clusters=tuple(k for c in cohorts for k in c.clusters),
                                 training_clusters=tuple(sorted({r.cluster_id for r in training})), cohorts=cohorts,
                                 limits=LearningLimits("fixture_units", 10., 0., 0., 1000))
        rows = []
        for cohort in cohorts:
            for i, key in enumerate(cohort.clusters):
                cell = "direct" if i % 2 == 0 else "dependent"
                action = candidate.choose(cell, binding=binding, allowed_actions=("full", "compact"))
                rows.append(PairedOutcome(key, revision({"key": key, "cell": cell, "action": action}),
                                          True, action == "full" or cell == "direct", 10.,
                                          10. if action == "full" else 2., 10., 1024))
        promotion = registry.finish(plan, tuple(rows))
        if not promotion["eligible"]:
            raise AssertionError("fixture setup must pass the demonstration gate")
        live_plan = monitor.start(registry.active_revision(binding=binding))
        events = []
        for i in range(60):
            key = f"live-{i}"
            # A changed dependency/tool contract causes verified failures after
            # twenty successes. No text is used as its own correctness label.
            result = MonitoredOutcome(key, binding, candidate.revision, revision({"fixture": i}), i < 20, 10., 1024)
            monitor.reserve(live_plan, cluster_id=key, cohort="retention")
            status = monitor.observe(live_plan, result)
            events.append(dict(outcome=asdict(result), status=status))
            if status["alarm"]:
                break
        # Simulate restarting after alarm persistence but before rollback.
        monitor.close()
        monitor = LocalRegressionMonitor(monitor_path, scope="fixture", binding=binding, limits=MonitorLimits())
        applied = monitor.enforce(live_plan, registry)
        return dict(evidence_class="deterministic_simulation_not_model_or_device_benchmark", model_calls=0,
                    training=[asdict(r) for r in training], candidate=asdict(candidate), baseline=asdict(baseline),
                    plan=plan, evaluation=[asdict(r) for r in rows], promotion=promotion,
                    live_plan=live_plan, events=events, alarm_survived_restart=monitor.status(live_plan)["alarm"],
                    rollback_applied=applied, restored_baseline=registry.active_revision(binding=binding) == baseline.revision)
    finally:
        monitor.close()
        registry.close()


def main():
    with tempfile.TemporaryDirectory() as directory:
        report = fixture_report(directory)
    print(json.dumps({k: report[k] for k in ("evidence_class", "model_calls", "alarm_survived_restart",
                                            "rollback_applied", "restored_baseline")}, indent=2))


if __name__ == "__main__":
    main()

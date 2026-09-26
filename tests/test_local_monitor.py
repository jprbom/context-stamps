"""Statistical oracles and restart/failure boundaries for local monitoring."""

import copy
import itertools
import math
import tempfile
import unittest
from dataclasses import replace
from fractions import Fraction
from pathlib import Path

from context_stamps.local_learning import (
    EvaluationCohort,
    LearningLimits,
    LocalLearningRegistry,
    PairedOutcome,
    revision,
)
from context_stamps.local_monitor import (
    LocalRegressionMonitor,
    MonitoredOutcome,
    MonitorLimits,
    bernoulli_log_e,
    update_restart_mixture,
)

BINDING, BASE, CANDIDATE = (revision({"name": n}) for n in ("environment", "base", "candidate"))


def direct_mixture(sequence):
    """Independent rational oracle: enumerate every possible start explicitly."""
    null, total = Fraction(1, 10), Fraction(1, len(sequence)+1)
    for start in range(1, len(sequence)+1):
        for fraction in (Fraction(1, 8), Fraction(1, 4), Fraction(1, 2), Fraction(3, 4)):
            alt = null + (1-null)*fraction
            product = Fraction(1, 4*start*(start+1))
            for failed in sequence[start-1:]:
                product *= alt/null if failed else (1-alt)/(1-null)
            total += product
    return float(total)


class MonitorMathTests(unittest.TestCase):
    def test_all_start_recurrence_matches_rational_oracle(self):
        for seq in itertools.product((False, True), repeat=8):
            sums = (None,)*4
            for i, failed in enumerate(seq, 1):
                sums, log_e = update_restart_mixture(sums, i, failed, .1)
            self.assertAlmostEqual(math.exp(log_e), direct_mixture(seq), delta=1e-8)

    def test_finite_horizon_optional_stopping_probability(self):
        # Exhaust every path under a dependent conditional null. Null failure
        # rate alternates with the previous label and is always <= .1.
        alpha = .05
        frontier = [((None,)*4, 1., False)]
        crossed = 0.
        for t in range(1, 15):
            next_frontier = []
            for sums, probability, previous in frontier:
                p = .05 if previous else .1
                for failed, chance in ((False, 1-p), (True, p)):
                    updated, log_e = update_restart_mixture(sums, t, failed, .1)
                    mass = probability*chance
                    if log_e >= -math.log(alpha):
                        crossed += mass
                    else:
                        next_frontier.append((updated, mass, failed))
            frontier = next_frontier
        self.assertGreater(crossed, 0)
        self.assertLessEqual(crossed, alpha)
        self.assertAlmostEqual(crossed + math.fsum(row[1] for row in frontier), 1.)

    def test_late_shift_detected_without_erasing_quiet_history(self):
        sums = (None,)*4
        alarm = None
        for t in range(1, 1031):
            sums, log_e = update_restart_mixture(sums, t, t > 1000, .1)
            if log_e >= math.log(800):
                alarm = t
                break
        self.assertIsNotNone(alarm)
        self.assertLessEqual(alarm, 1030)
        self.assertLess(bernoulli_log_e(alarm-1000, alarm, .1), math.log(800))

    def test_state_and_number_validation(self):
        for state, n, failed, limit in (([None]*4, 1, False, .1), ((None,)*4, True, False, .1),
                                      ((None,)*4, 1, 1, .1), ((None,)*4, 1, False, 0.),
                                      ((None,)*4, 1, False, float('nan')),
                                      ((0.,)*4, 1, False, .1), ((float('inf'),)*4, 2, False, .1)):
            with self.assertRaises(ValueError):
                update_restart_mixture(state, n, failed, limit)


class MonitorPersistenceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.path = str(Path(self.temp.name)/"monitor.sqlite")
        self.monitor = self.open()
        self.plan = self.monitor.start(CANDIDATE)

    def open(self, **kwargs):
        return LocalRegressionMonitor(self.path, scope="domain", binding=BINDING,
                                      limits=MonitorLimits(), **kwargs)

    def tearDown(self):
        self.monitor.close()
        self.temp.cleanup()

    def result(self, key, **kwargs):
        return replace(MonitoredOutcome(key, BINDING, CANDIDATE, revision({"task": key}), True, 10., 1024), **kwargs)

    def record(self, key, cohort="adaptation", **kwargs):
        self.monitor.reserve(self.plan, cluster_id=key, cohort=cohort)
        return self.monitor.observe(self.plan, self.result(key, **kwargs))

    def test_reserve_before_outcomes_and_do_not_skip_pending_after_restart(self):
        with self.assertRaises(ValueError):
            self.monitor.observe(self.plan, self.result("one"))
        self.monitor.reserve(self.plan, cluster_id="one", cohort="adaptation")
        self.monitor.close()
        self.monitor = self.open()
        self.assertEqual(self.monitor.current_plan(), self.plan)
        self.assertFalse(self.monitor.status(self.plan)["dispatch_allowed"])
        with self.assertRaises(ValueError):
            self.monitor.reserve(self.plan, cluster_id="two", cohort="retention")
        with self.assertRaises(ValueError):
            self.monitor.end(self.plan)
        self.assertTrue(self.monitor.observe(self.plan, self.result("one", correct=None))["alarm"])

    def test_repeated_outcomes_cannot_inflate_evidence(self):
        self.record("one")
        with self.assertRaises(ValueError):
            self.monitor.observe(self.plan, self.result("one"))
        with self.assertRaises(ValueError):
            self.record("one")

    def test_unknown_safety_and_memory_causes_immediate_persistent_alarm(self):
        for field in ({"correct": None}, {"wall_ms": None}, {"peak_ram_bytes": None},
                      {"safety_violation": True}, {"peak_ram_bytes": 2**40}):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as directory:
                path = str(Path(directory)/"monitor.sqlite")
                mon = LocalRegressionMonitor(path, scope="domain", binding=BINDING, limits=MonitorLimits())
                plan = mon.start(CANDIDATE)
                mon.reserve(plan, cluster_id="one", cohort="retention")
                self.assertTrue(mon.observe(plan, self.result("one", **field))["alarm"])
                mon.close()
                mon = LocalRegressionMonitor(path, scope="domain", binding=BINDING, limits=MonitorLimits())
                try:
                    self.assertTrue(mon.status(plan)["alarm"])
                    self.assertFalse(mon.status(plan)["dispatch_allowed"])
                finally:
                    mon.close()

    def test_retention_failures_cannot_be_diluted_by_adaptation(self):
        for i in range(100):
            self.record(f"new-{i}")
        for i in range(30):
            status = self.record(f"old-{i}", "retention", correct=False)
            if status["alarm"]:
                break
        self.assertIn("retention:failures_alarm", status["reasons"])
        self.assertEqual(status["counts"]["adaptation"]["trials"], 100)

    def test_late_answers_alarm_even_when_correct(self):
        for i in range(30):
            status = self.record(f"late-{i}", wall_ms=2000.)
            if status["alarm"]:
                break
        self.assertIn("adaptation:late_alarm", status["reasons"])
        self.assertEqual(status["counts"]["adaptation"]["failures"], 0)

    def test_binding_revision_and_plan_mutation_rejected(self):
        self.monitor.reserve(self.plan, cluster_id="one", cohort="adaptation")
        for value in (self.result("one", binding=BASE), self.result("one", candidate_revision=BASE)):
            with self.assertRaises(ValueError):
                self.monitor.observe(self.plan, value)
        changed = copy.deepcopy(self.plan)
        changed["per_check_alpha"] = .5
        with self.assertRaises(ValueError):
            self.monitor.observe(changed, self.result("one"))
        self.assertTrue(self.monitor.observe(self.plan, self.result("one"))["dispatch_allowed"])

    def test_epoch_budget_duplicate_and_configuration_survive_restart(self):
        self.monitor.end(self.plan)
        self.monitor.close()
        self.monitor = self.open()
        with self.assertRaises(ValueError):
            self.monitor.start(CANDIDATE)
        second = self.monitor.start(BASE)
        self.assertEqual(second["epoch"], 2)
        self.assertEqual(second["per_check_alpha"], .01/(2*3*4))
        with self.assertRaises(ValueError):
            self.open(alpha=.02)
        with self.assertRaises(ValueError):
            self.open(cohorts=("adaptation",))

    def test_competing_writers_cannot_reserve_different_pending_tasks(self):
        second = self.open()
        try:
            self.monitor.reserve(self.plan, cluster_id="first", cohort="retention")
            with self.assertRaises(ValueError):
                second.reserve(self.plan, cluster_id="second", cohort="retention")
            self.assertTrue(second.observe(self.plan, self.result("first"))["dispatch_allowed"])
        finally:
            second.close()

    def test_invalid_labels_bounds_and_identifiers(self):
        for changes in ({"correct": .99}, {"wall_ms": float("nan")}, {"peak_ram_bytes": True},
                        {"safety_violation": 1}, {"cluster_id": "private text"}):
            with self.assertRaises(ValueError):
                self.result("x", **changes)


class RollbackIntegrationTests(unittest.TestCase):
    def test_durable_alarm_retries_and_stale_alarm_cannot_revoke_new_candidate(self):
        with tempfile.TemporaryDirectory() as directory:
            registry = LocalLearningRegistry(str(Path(directory)/"registry.sqlite"), scope="domain",
                                             binding=BINDING, baseline=BASE)
            monitor = LocalRegressionMonitor(str(Path(directory)/"monitor.sqlite"), scope="domain",
                                             binding=BINDING, limits=MonitorLimits())
            try:
                def promote(candidate, prefix):
                    cohorts = tuple(EvaluationCohort(c, tuple(f"{prefix}-{c}-{i}" for i in range(800)))
                                    for c in ("adaptation", "retention"))
                    keys = tuple(k for c in cohorts for k in c.clusters)
                    plan = registry.register(candidate, evaluation_clusters=keys, training_clusters=(),
                                             limits=LearningLimits("fixture_units", 10., 0., 0., 1000), cohorts=cohorts)
                    rows = tuple(PairedOutcome(k, revision({"id": k}), True, True, 10., 2., 10., 1024) for k in keys)
                    self.assertTrue(registry.finish(plan, rows)["eligible"])

                promote(CANDIDATE, "first")
                plan = monitor.start(CANDIDATE)
                monitor.reserve(plan, cluster_id="live-one", cohort="retention")
                monitor.observe(plan, MonitoredOutcome("live-one", BINDING, CANDIDATE, BASE, None, None, None))
                self.assertTrue(monitor.enforce(plan, registry))
                self.assertEqual(registry.active_revision(binding=BINDING), BASE)
                self.assertFalse(monitor.enforce(plan, registry))
                newer = revision({"name": "newer"})
                promote(newer, "second")
                self.assertFalse(monitor.enforce(plan, registry))
                self.assertEqual(registry.active_revision(binding=BINDING), newer)
                with self.assertRaises(ValueError):
                    registry.register(CANDIDATE, evaluation_clusters=("new", "old"), training_clusters=(),
                                      limits=LearningLimits("fixture_units", 10., 0., 0., 1000),
                                      cohorts=(EvaluationCohort("adaptation", ("new",)), EvaluationCohort("retention", ("old",))))
            finally:
                monitor.close()
                registry.close()


if __name__ == "__main__":
    unittest.main()

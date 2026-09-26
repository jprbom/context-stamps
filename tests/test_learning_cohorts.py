"""Retention cannot be hidden by pooled success on newly learned tasks."""

import copy
import math
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from context_stamps.local_learning import (
    EvaluationCohort,
    LearningLimits,
    LocalLearningRegistry,
    PairedOutcome,
    assess_candidate,
    assess_cohorts,
    revision,
)

BINDING = revision({"scope": "cohort-test"})
BASELINE = revision({"policy": "base"})
CANDIDATE = revision({"policy": "new"})
LIMITS = LearningLimits("fixture_units", 10., 0., 0., 1000)


def rows(name, n, *, failures=0, cost=2.):
    return tuple(PairedOutcome(f"{name}-{i}", revision({"id": f"{name}-{i}"}), True,
                              i >= failures, 10., cost, 10., 1024) for i in range(n))


def partition(adaptation, retention):
    return (EvaluationCohort("adaptation", tuple(r.cluster_id for r in adaptation)),
            EvaluationCohort("retention", tuple(r.cluster_id for r in retention)))


class CohortStatisticsTests(unittest.TestCase):
    def test_pooled_success_cannot_hide_retention_harm(self):
        new, old = rows("new", 6000), rows("old", 700, failures=7)
        self.assertTrue(assess_candidate(new+old, LIMITS, alpha=.025)["eligible"])
        report = assess_cohorts(new+old, LIMITS, partition(new, old), alpha=.025)
        self.assertFalse(report["eligible"])
        self.assertIn("cohort:retention:new_failure_bound", report["reasons"])

    def test_retention_can_preserve_cost_while_adaptation_saves(self):
        new, old = rows("new", 800), rows("old", 800, cost=10.)
        report = assess_cohorts(new+old, LIMITS, partition(new, old), alpha=.025)
        self.assertTrue(report["eligible"])
        self.assertLess(report["cohorts"]["retention"]["net_saving_lower"], 0)
        self.assertNotIn("net_saving_bound", report["cohorts"]["retention"]["applied_statistical_checks"])

    def test_error_budget_is_shared_across_all_applied_tests(self):
        new, old = rows("new", 800), rows("old", 800)
        report = assess_cohorts(new+old, LIMITS, partition(new, old), alpha=.025)
        self.assertEqual(report["statistical_checks"], 7)
        self.assertAlmostEqual(report["per_check_alpha"]*7, .025)
        self.assertAlmostEqual(report["cohorts"]["retention"]["new_failure_upper"],
                               -math.expm1(math.log(.025/7)/800))

    def test_small_retention_set_cannot_borrow_new_task_sample_size(self):
        new, old = rows("new", 800), rows("old", 10)
        report = assess_cohorts(new+old, LIMITS, partition(new, old), alpha=.025)
        self.assertIn("cohort:retention:new_failure_bound", report["reasons"])

    def test_grouped_latency_memory_and_missing_data_still_fail(self):
        new, old = rows("new", 800), rows("old", 800)
        for replacement, reason in (({"candidate_wall_ms": 2000.}, "deadline_bound"),
                                    ({"candidate_peak_ram_bytes": 2**40}, "memory_ceiling"),
                                    ({"candidate_cost": None}, "missing_measurements"),
                                    ({"safety_violation": True}, "safety_violation")):
            with self.subTest(reason=reason):
                changed = tuple(replace(r, **replacement) for r in old)
                report = assess_cohorts(new+changed, LIMITS, partition(new, changed), alpha=.025)
                self.assertIn(f"cohort:retention:{reason}", report["reasons"])

    def test_partition_rejects_overlap_omission_and_unknown_clusters(self):
        new, old = rows("new", 2), rows("old", 2)
        for groups in (
            (EvaluationCohort("adaptation", (new[0].cluster_id,)), partition(new, old)[1]),
            (partition(new, old)[0], EvaluationCohort("retention", (old[0].cluster_id, new[0].cluster_id))),
            (partition(new, old)[0], EvaluationCohort("retention", (old[0].cluster_id, "unknown"))),
            (partition(new, old)[0], EvaluationCohort("adaptation", tuple(r.cluster_id for r in old))),
        ):
            with self.assertRaises(ValueError):
                assess_cohorts(new+old, LIMITS, groups, alpha=.025)

    def test_empty_and_mutable_cohorts_rejected(self):
        for clusters in ((), ["one"], ("one", "one")):
            with self.assertRaises(ValueError):
                EvaluationCohort("retention", clusters)


class CohortRegistryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.path = str(Path(self.temp.name)/"learning.sqlite")
        self.registry = self.open()
        self.new, self.old = rows("new", 800), rows("old", 800)

    def open(self, **kwargs):
        return LocalLearningRegistry(self.path, scope="cohort-test", binding=BINDING, baseline=BASELINE, **kwargs)

    def tearDown(self):
        self.registry.close()
        self.temp.cleanup()

    def register(self):
        return self.registry.register(CANDIDATE, evaluation_clusters=tuple(r.cluster_id for r in self.new+self.old),
                                      training_clusters=("train-1",), limits=LIMITS,
                                      cohorts=partition(self.new, self.old))

    def test_default_requires_adaptation_and_retention(self):
        with self.assertRaises(ValueError):
            self.registry.register(CANDIDATE, evaluation_clusters=("one",), training_clusters=(), limits=LIMITS)
        with self.assertRaises(ValueError):
            self.registry.register(CANDIDATE, evaluation_clusters=("one",), training_clusters=(), limits=LIMITS,
                                   cohorts=(EvaluationCohort("adaptation", ("one",)),))

    def test_reassignment_of_cohorts_after_registration_is_rejected(self):
        plan = self.register()
        changed = copy.deepcopy(plan)
        changed["cohorts"][0]["name"], changed["cohorts"][1]["name"] = changed["cohorts"][1]["name"], changed["cohorts"][0]["name"]
        with self.assertRaises(ValueError):
            self.registry.finish(changed, self.new+self.old)
        self.assertTrue(self.registry.finish(plan, self.new+self.old)["eligible"])

    def test_restart_keeps_required_groups_and_cannot_downgrade(self):
        plan = self.register()
        self.registry.close()
        self.registry = self.open()
        with self.assertRaises(ValueError):
            self.open(required_cohorts=())
        self.assertTrue(self.registry.finish(plan, self.new+self.old)["eligible"])
        self.assertEqual(self.registry.active_revision(binding=BINDING), CANDIDATE)
        self.registry.rollback(reason="verified_drift")
        self.assertEqual(self.registry.active_revision(binding=BINDING), BASELINE)

    def test_retention_failure_keeps_previous_revision(self):
        plan = self.register()
        bad = tuple(replace(r, candidate_correct=False) for r in self.old)
        self.assertFalse(self.registry.finish(plan, self.new+bad)["eligible"])
        self.assertEqual(self.registry.active_revision(binding=BINDING), BASELINE)

    def test_retention_ids_cannot_be_reused_as_fresh_evaluation(self):
        plan = self.register()
        self.registry.abandon(plan)
        with self.assertRaises(ValueError):
            self.register()

    def test_explicit_legacy_mode_preserves_existing_registry(self):
        path = str(Path(self.temp.name)/"legacy.sqlite")
        legacy = LocalLearningRegistry(path, scope="legacy", binding=BINDING, baseline=BASELINE, required_cohorts=())
        legacy.close()
        with self.assertRaises(ValueError):
            LocalLearningRegistry(path, scope="legacy", binding=BINDING, baseline=BASELINE)
        legacy = LocalLearningRegistry(path, scope="legacy", binding=BINDING, baseline=BASELINE, required_cohorts=())
        self.assertEqual(legacy.active_revision(binding=BINDING), BASELINE)
        legacy.close()


if __name__ == "__main__":
    unittest.main()

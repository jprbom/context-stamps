"""Adversarial and statistical boundaries of local policy adaptation."""

import math
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from context_stamps.local_learning import (
    LearningLimits,
    LocalLearningRegistry,
    PairedOutcome,
    assess_candidate,
    revision,
)
from context_stamps.local_policy import CellPolicy, PolicyObservation, fit_cell_policy

BINDING = revision({"scope": "fixture", "model": "small-v1", "verifier": "exact-v1"})
BASELINE = revision({"route": "full"})
CANDIDATE = revision({"route": "conditional"})
LIMITS = LearningLimits("fixture_units", 10., 0., 0., 1000)


def observations(n=600, prefix="eval"):
    return tuple(PairedOutcome(f"{prefix}-{i}", revision({"task": i}), True, True, 10., 2., 10., 1024)
                 for i in range(n))


class StatisticalGateTests(unittest.TestCase):
    def test_zero_error_exact_bound_and_cost_range(self):
        report = assess_candidate(observations(), LIMITS, alpha=.025)
        self.assertTrue(report["eligible"])
        self.assertAlmostEqual(report["new_failure_upper"], 1 - (.025 / 4)**(1 / 600))
        self.assertAlmostEqual(report["net_saving_lower"], 8 - 10 * math.sqrt(2 * math.log(160) / 600))

    def test_small_perfect_sample_is_not_qualification(self):
        report = assess_candidate(observations(10), LIMITS, alpha=.025)
        self.assertFalse(report["eligible"])
        self.assertIn("new_failure_bound", report["reasons"])

    def test_gains_elsewhere_do_not_cancel_new_harm(self):
        rows = observations()
        rows = tuple(replace(r, baseline_correct=(i >= 100), candidate_correct=(i < 100 or i >= 200))
                     for i, r in enumerate(rows))
        self.assertEqual(sum(r.baseline_correct for r in rows), sum(r.candidate_correct for r in rows))
        self.assertIn("new_failure_bound", assess_candidate(rows, LIMITS, alpha=.025)["reasons"])

    def test_equally_bad_baseline_does_not_qualify(self):
        rows = tuple(replace(r, baseline_correct=False, candidate_correct=False) for r in observations())
        self.assertIn("absolute_failure_bound", assess_candidate(rows, LIMITS, alpha=.025)["reasons"])

    def test_abstentions_remain_failures(self):
        rows = tuple(replace(r, candidate_correct=False) for r in observations())
        report = assess_candidate(rows, LIMITS, alpha=.025)
        self.assertEqual(report["candidate_failures"], 600)
        self.assertEqual(report["failure_upper"], 1.)

    def test_amortized_learning_cost_can_reject(self):
        limits = replace(LIMITS, update_cost=10000., amortization_tasks=1000)
        self.assertIn("net_saving_bound", assess_candidate(observations(), limits, alpha=.025)["reasons"])

    def test_tail_latency_is_separate_from_average_cost(self):
        rows = tuple(replace(r, candidate_wall_ms=2000.) if i < 60 else r
                     for i, r in enumerate(observations()))
        self.assertIn("deadline_bound", assess_candidate(rows, LIMITS, alpha=.025)["reasons"])

    def test_unknown_outlier_memory_and_safety_fail_closed(self):
        for changes, reason in (({"candidate_cost": None}, "missing_measurements"),
                                ({"candidate_wall_ms": None}, "missing_measurements"),
                                ({"candidate_peak_ram_bytes": None}, "missing_measurements"),
                                ({"candidate_cost": 11.}, "cost_out_of_registered_range"),
                                ({"baseline_cost": 11.}, "cost_out_of_registered_range"),
                                ({"candidate_peak_ram_bytes": 2**40}, "memory_ceiling"),
                                ({"safety_violation": True}, "safety_violation")):
            with self.subTest(changes=changes):
                rows = observations()
                report = assess_candidate((replace(rows[0], **changes), *rows[1:]), LIMITS, alpha=.025)
                self.assertFalse(report["eligible"])
                self.assertIn(reason, report["reasons"])

    def test_wall_cost_cannot_exclude_pipeline_overhead(self):
        report = assess_candidate(observations(), replace(LIMITS, metric="wall_ms"), alpha=.025)
        self.assertIn("inconsistent_wall_measurement", report["reasons"])

    def test_invalid_numbers_and_duplicate_tasks(self):
        for changes in ({"cost_cap": float("nan")}, {"maximum_failure": float("inf")},
                        {"amortization_tasks": True}, {"maximum_ram_bytes": -1}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                replace(LIMITS, **changes)
        with self.assertRaises(ValueError):
            replace(observations(1)[0], candidate_correct=1)
        with self.assertRaises(ValueError):
            replace(observations(1)[0], candidate_cost=float("nan"))
        with self.assertRaises(ValueError):
            assess_candidate(observations(1) * 2, LIMITS, alpha=.025)


class RegistryTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = str(Path(self.tmp.name) / "learning.sqlite")
        self.registry = self.open()
        self.rows = observations()

    def open(self):
        return LocalLearningRegistry(self.path, scope="fixture", binding=BINDING, baseline=BASELINE)

    def tearDown(self):
        self.registry.close()
        self.tmp.cleanup()

    def register(self, rows=None, candidate=CANDIDATE, train=()):
        rows = self.rows if rows is None else rows
        return self.registry.register(candidate, evaluation_clusters=tuple(r.cluster_id for r in rows),
                                      training_clusters=train, limits=LIMITS)

    def test_restart_promotion_and_rollback_revoke(self):
        plan = self.register()
        self.assertEqual(self.registry.active_revision(binding=BINDING), BASELINE)
        self.registry.close()
        self.registry = self.open()
        self.assertTrue(self.registry.finish(plan, self.rows)["eligible"])
        self.assertEqual(self.registry.active_revision(binding=BINDING), CANDIDATE)
        self.registry.rollback(reason="verified_regression")
        self.registry.close()
        self.registry = self.open()
        self.assertEqual(self.registry.active_revision(binding=BINDING), BASELINE)
        with self.assertRaises(ValueError):
            self.register(observations(prefix="fresh"))

    def test_changed_binding_returns_no_policy(self):
        self.assertIsNone(self.registry.active_revision(binding=revision({"policy": "changed"})))
        with self.assertRaises(ValueError):
            LocalLearningRegistry(self.path, scope="fixture", binding=revision({}), baseline=BASELINE)

    def test_reused_training_validation_and_duplicates_rejected(self):
        with self.assertRaises(ValueError):
            self.register(train=(self.rows[0].cluster_id,))
        with self.assertRaises(ValueError):
            self.register(self.rows + self.rows[:1])
        plan = self.register(train=("prior-train",))
        self.registry.abandon(plan)
        with self.assertRaises(ValueError):
            self.register()
        with self.assertRaises(ValueError):
            self.register((replace(self.rows[0], cluster_id="prior-train"),))

    def test_round_spending_survives_abandonment_and_restart(self):
        plan = self.register()
        self.registry.abandon(plan)
        self.registry.close()
        self.registry = self.open()
        plan2 = self.register(observations(prefix="fresh"))
        self.assertEqual(plan2["round"], 2)
        self.assertEqual(plan["round_alpha"], .05 / 2)
        self.assertEqual(plan2["round_alpha"], .05 / 6)
        self.assertLess(sum(.05 / (i * (i + 1)) for i in range(1, 129)), .05)

    def test_no_optional_stopping_or_second_result(self):
        plan = self.register()
        with self.assertRaises(ValueError):
            self.registry.finish(plan, self.rows[:50])
        self.registry.finish(plan, self.rows)
        with self.assertRaises(ValueError):
            self.registry.finish(plan, self.rows)

    def test_plan_mutation_and_duplicate_evaluation_rejected(self):
        plan = self.register()
        altered = {**plan, "limits": {**plan["limits"], "maximum_new_failure": .5}}
        with self.assertRaises(ValueError):
            self.registry.finish(altered, self.rows)
        with self.assertRaises(ValueError):
            self.registry.finish(plan, (*self.rows[1:], self.rows[-1]))
        self.assertTrue(self.registry.finish(plan, self.rows)["eligible"])

    def test_two_connections_cannot_register_pending_candidates(self):
        plan = self.register()
        second = self.open()
        try:
            with self.assertRaises(ValueError):
                second.register(revision({"another": True}), evaluation_clusters=("next",),
                                training_clusters=(), limits=LIMITS)
            second.abandon(plan)
            with self.assertRaises(ValueError):
                self.registry.finish(plan, self.rows)
        finally:
            second.close()

    def test_rollback_cancels_pending_promotion(self):
        plan = self.register()
        self.registry.rollback(reason="authority_changed")
        with self.assertRaises(ValueError):
            self.registry.finish(plan, self.rows)

    def test_rejection_keeps_baseline(self):
        rows = observations(10)
        plan = self.register(rows)
        self.assertFalse(self.registry.finish(plan, rows)["eligible"])
        self.assertEqual(self.registry.active_revision(binding=BINDING), BASELINE)


class LocalPolicyTests(unittest.TestCase):
    def rows(self):
        return tuple(PolicyObservation(f"train-{cell}-{i}", cell, action,
                                       action == "full" or cell == "simple", 10. if action == "full" else 2.)
                     for cell in ("simple", "dependent") for i in range(30) for action in ("full", "compact"))

    def fit(self, rows=None, **kwargs):
        return fit_cell_policy(self.rows() if rows is None else rows, binding=BINDING,
                               baseline_action="full", cost_cap=10., **kwargs)

    def test_local_fit_learns_different_context_actions(self):
        policy = self.fit()
        self.assertEqual(dict(policy.choices), {"dependent": "full", "simple": "compact"})
        self.assertEqual(policy.revision, self.fit(tuple(reversed(self.rows()))).revision)

    def test_under_supported_and_unseen_cells_use_baseline(self):
        policy = self.fit(minimum_tasks=40)
        self.assertTrue(all(action == "full" for _, action in policy.choices))
        self.assertEqual(policy.choose("unknown", binding=BINDING, allowed_actions=("full", "compact")), "full")

    def test_current_binding_and_host_authorization_required(self):
        policy = self.fit()
        self.assertEqual(policy.choose("simple", binding=BINDING, allowed_actions=("full",)), "full")
        self.assertIsNone(policy.choose("simple", binding=BINDING, allowed_actions=()))
        self.assertIsNone(policy.choose("simple", binding=revision({}), allowed_actions=("full", "compact")))

    def test_incomplete_counterfactuals_and_duplicate_tasks_rejected(self):
        for rows in (self.rows()[:-1], self.rows() + self.rows()[:1],
                     (replace(self.rows()[0], cell="wrong"), *self.rows()[1:])):
            with self.subTest(n=len(rows)), self.assertRaises(ValueError):
                self.fit(rows)

    def test_invalid_cells_outliers_and_confidence_labels_rejected(self):
        with self.assertRaises(ValueError):
            self.fit((replace(self.rows()[0], cost=11.), *self.rows()[1:]))
        with self.assertRaises(ValueError):
            replace(self.rows()[0], correct=.99)
        with self.assertRaises(ValueError):
            CellPolicy(BINDING, "full", (("cell", "full"), ("cell", "compact")))


if __name__ == "__main__":
    unittest.main()

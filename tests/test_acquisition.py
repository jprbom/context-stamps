"""Safety and statistical-unit tests for the optional acquisition planner."""

import json
import math
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from context_stamps.acquisition import (
    AcquisitionModel,
    AcquisitionRisk,
    StopPolicy,
    Trajectory,
    assess_policy,
    plan_prefix,
    prefix_features,
)
from context_stamps.decisions.calibration import error_upper_bound


def constant_model(score=.99):
    return AcquisitionModel((0.,) * 33, (1.,) * 33, (), (), ((0.,) * 33,) * 3,
                            (math.log(score / (1 - score)), 0., 0.))


ROWS = ((.2, 1., 2., 3., .5, 1.), (.7, 2., 4., 2., 1., .5), (.1, 0., 1., 0., .1, .1))


class AcquisitionTests(unittest.TestCase):
    def setUp(self):
        self.model = constant_model()
        self.policy = StopPolicy(self.model.revision, .9)
        self.family = (("fixture", self.policy.revision),)
        self.trajectories = tuple(Trajectory(str(i), (.99, .99, .99), (True,) * 3) for i in range(100))
        self.risk = assess_policy(self.policy, self.trajectories, scope="fixture", family=self.family)

    def test_prefix_alignment_and_tail_features(self):
        order, lengths, features = prefix_features(ROWS)
        self.assertEqual(order, (1, 0, 2))
        self.assertEqual(lengths, (1, 2, 3))
        self.assertTrue(all(len(v) == 33 for v in features))
        self.assertAlmostEqual(features[0][3], .7)
        self.assertAlmostEqual(features[0][7], .15)
        self.assertEqual(features[-1][7], 0)

    def test_tied_order_and_single_candidate(self):
        self.assertEqual(prefix_features((ROWS[0], ROWS[0]))[0], (0, 1))
        risk = self.risk
        result = plan_prefix((ROWS[0],), self.model, self.policy, scope="fixture", risk=risk)
        self.assertEqual(result.status, "pool_exhausted")
        self.assertEqual(result.indices, (0,))

    def test_malformed_candidates_rejected(self):
        for rows in ([], (), (ROWS[0],) * 257, ((0.,) * 5,), ((math.nan,) * 6,),
                     ((0., 0., 0., 0., -1., 1.),), (list(ROWS[0]),)):
            with self.subTest(rows=str(rows)[:50]), self.assertRaises(ValueError):
                prefix_features(rows)

    def test_model_shape_numeric_family_bounds(self):
        for change in ({"mean": (0.,)}, {"scale": (0.,) * 33}, {"output_bias": (math.nan,) * 3},
                       {"hidden_weights": ((0.,) * 33,)}, {"feature_revision": "unknown"},
                       {"output_weights": ((101.,) * 33,) * 3}):
            with self.subTest(change=next(iter(change))), self.assertRaises(ValueError):
                replace(self.model, **change)
        with self.assertRaises(ValueError):
            self.model.predict((math.inf,) * 33)

    def test_portable_model_roundtrip_and_tampering(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "model.json"
            self.model.save(path)
            loaded = AcquisitionModel.load(path)
            self.assertEqual(loaded, self.model)
            self.assertEqual(loaded.predict((0.,) * 33), self.model.predict((0.,) * 33))
            with self.assertRaises(FileExistsError):
                self.model.save(path)
            payload = json.loads(path.read_text())
            payload["model"]["output_bias"][0] = 0
            path.write_text(json.dumps(payload))
            with self.assertRaises(ValueError):
                AcquisitionModel.load(path)
            path.write_text('{"model":{},"model":{},"revision":"x"}')
            with self.assertRaises(ValueError):
                AcquisitionModel.load(path)

    def test_one_label_per_stopped_trajectory(self):
        rows = (Trajectory("a", (.1, .99, .99), (False, True, True)),
                Trajectory("b", (.99, .99, .99), (False, False, True)))
        report = assess_policy(self.policy, rows, scope="fixture", family=self.family)
        self.assertEqual((report.clusters, report.accepted, report.errors), (2, 2, 1))
        self.assertEqual(report.upper_error, error_upper_bound(1, 2, .05))

    def test_repeated_looks_not_independent_samples(self):
        rows = tuple(Trajectory(str(i), (.99,) * 9, (True,) * 9) for i in range(100))
        report = assess_policy(self.policy, rows, scope="fixture", family=self.family)
        self.assertEqual(report.accepted, 100)
        self.assertEqual(report.upper_error, self.risk.upper_error)
        with self.assertRaises(ValueError):
            assess_policy(self.policy, (rows[0], rows[0]), scope="fixture", family=self.family)

    def test_family_correction_covers_policy_and_scope_search(self):
        family = self.family + (("other", self.policy.revision),)
        report = assess_policy(self.policy, self.trajectories, scope="fixture", family=family)
        self.assertGreater(report.upper_error, self.risk.upper_error)
        self.assertEqual(report.upper_error, error_upper_bound(0, 100, .025))

    def test_no_acceptance_does_not_assert_zero_risk(self):
        rows = tuple(Trajectory(str(i), (0., 0.), (False, True)) for i in range(100))
        report = assess_policy(self.policy, rows, scope="fixture", family=self.family)
        self.assertIsNone(report.upper_error)
        self.assertFalse(report.permits(self.policy, scope="fixture", maximum_error=1))

    def test_wrong_scope_policy_and_strict_risk_fall_back(self):
        for scope, policy, limit in (("other", self.policy, .05), ("fixture", replace(self.policy, threshold=.8), .05),
                                     ("fixture", self.policy, .001)):
            result = plan_prefix(ROWS, self.model, policy, scope=scope, risk=self.risk, maximum_error=limit)
            self.assertEqual(result.status, "unqualified_full_pool")
            self.assertEqual(len(result.indices), 3)
            self.assertIsNone(result.score)
        with self.assertRaises(ValueError):
            plan_prefix(ROWS, constant_model(.8), self.policy, scope="fixture", risk=self.risk)

    def test_unqualified_path_skips_features_and_inference_but_validates_rows(self):
        with patch("context_stamps.acquisition.prefix_features", side_effect=AssertionError("unused")), \
                patch.object(AcquisitionModel, "predict", side_effect=AssertionError("unused")):
            result = plan_prefix(ROWS, self.model, self.policy, scope="fixture")
            self.assertEqual(result.status, "unqualified_full_pool")
            with self.assertRaises(ValueError):
                plan_prefix(((math.nan,) * 6,), self.model, self.policy, scope="fixture")

    def test_qualified_stop_and_budget_exhaustion_are_distinct(self):
        result = plan_prefix(ROWS, self.model, self.policy, scope="fixture", risk=self.risk)
        self.assertEqual((result.status, result.indices), ("early_stop", (1,)))
        for risk in (self.risk, None):
            result = plan_prefix(ROWS, self.model, self.policy, scope="fixture", risk=risk, max_items=0)
            self.assertEqual((result.status, result.indices), ("budget_exhausted", ()))
            self.assertIsNone(result.upper_error)
        result = plan_prefix(ROWS, self.model, self.policy, scope="fixture", max_items=2)
        self.assertEqual(result.status, "budget_exhausted")

    def test_risk_cannot_change_bound_without_counts(self):
        with self.assertRaises(ValueError):
            replace(self.risk, upper_error=0.)
        with self.assertRaises(ValueError):
            AcquisitionRisk(self.policy.revision, "fixture", "0" * 64, "0" * 64, 1, 2, 0, 1, .05, None)

    def test_policy_and_trajectory_validation(self):
        for scores in ([], (), (math.nan,), (True,), (0.,) * 10):
            with self.assertRaises(ValueError):
                self.policy.choose(scores)
        with self.assertRaises(ValueError):
            Trajectory("x", (.1, .2), (True, False))
        with self.assertRaises(ValueError):
            assess_policy(self.policy, self.trajectories, scope="fixture", family=self.family * 2)
        with self.assertRaises(ValueError):
            assess_policy(self.policy, self.trajectories, scope="fixture", family=(("other", self.policy.revision),))


if __name__ == "__main__":
    unittest.main()

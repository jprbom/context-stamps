import dataclasses
import unittest

from context_stamps.linear_policy import UtilityPair, fit_linear_policy


class LinearPolicyTests(unittest.TestCase):
    def rows(self):
        return tuple(UtilityPair(str(i), (float(i % 2),), i % 2 == 0, i % 2 == 1, 100, 100) for i in range(80))

    def fit(self, rows=None):
        return fit_linear_policy(self.rows() if rows is None else rows, binding="a"*64,
                                 baseline_action="state", candidate_action="linked", cost_cap=100)

    def test_learns_opposing_regions_and_preserves_binding(self):
        p = self.fit()
        self.assertEqual(p.choose((0.,), binding="a"*64, allowed_actions=("state", "linked")), "state")
        self.assertEqual(p.choose((1.,), binding="a"*64, allowed_actions=("state", "linked")), "linked")
        self.assertIsNone(p.choose((1.,), binding="b"*64, allowed_actions=("state", "linked")))
        self.assertEqual(p.choose((1.,), binding="a"*64, allowed_actions=("state",)), "state")

    def test_equal_quality_prefers_lower_cost(self):
        rows = tuple(UtilityPair(str(i), (0.,), True, True, 100, 50) for i in range(20))
        p = self.fit(rows)
        self.assertEqual(p.choose((0.,), binding="a"*64, allowed_actions=("state", "linked")), "linked")

    def test_singular_features_are_regularized(self):
        rows = tuple(UtilityPair(str(i), (1., 1.), True, False, 100, 100) for i in range(4))
        p = self.fit(rows)
        self.assertEqual(p.choose((1., 1.), binding="a"*64, allowed_actions=("state", "linked")), "state")

    def test_rejects_invalid_evidence_and_nonfinite_values(self):
        with self.assertRaises(ValueError):
            self.fit((self.rows()[0], self.rows()[0]))
        for bad in (float("nan"), float("inf"), 1.1, True):
            with self.assertRaises(ValueError):
                UtilityPair("x", (bad,), True, True, 1, 1)
        with self.assertRaises(ValueError):
            self.fit((self.rows()[0], dataclasses.replace(self.rows()[1], candidate_cost=101)))
        with self.assertRaises(ValueError):
            UtilityPair("x", (0.,), 1, True, 1, 1)


if __name__ == "__main__":
    unittest.main()

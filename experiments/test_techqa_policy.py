import unittest

import numpy as np
from techqa_policy import FEATURES, features, fit, predict


class TechQAPolicyTests(unittest.TestCase):
    def test_fixed_input_output_features(self):
        compiled = dict(question="Which port?", selected=[dict(score=3.0, query_coverage=.5)])
        row = dict(answer="7402", check=dict(quotes=[dict(start=0, end=20)]))
        x = features(compiled, row)
        self.assertEqual(len(x), len(FEATURES))
        compiled["ANSWER"] = "forbidden metadata"
        row["ANSWERABLE"] = "Y"
        self.assertEqual(x, features(compiled, row))
        self.assertEqual(x[-1], 1.0)

    def test_fit_numerically_matches_verified_signal(self):
        rng = np.random.default_rng(83)
        x = rng.normal(size=(200, len(FEATURES)))
        y = np.clip(.4*x[:, 0]-.2*x[:, 2], -1, 1)
        policy = fit(x, y, 1.0)
        predictions = predict(policy, x)
        self.assertLess(np.mean((predictions-y)**2), .01)
        self.assertLess(np.max(np.abs(predict(policy, x, quantized=True)-predictions)), 1e-3)
        self.assertFalse(policy["activated"])

    def test_sparse_constant_data_stays_finite(self):
        x = np.ones((30, len(FEATURES)))
        policy = fit(x, np.full(30, -.5), 10.0)
        self.assertTrue(np.allclose(predict(policy, x), -.5))
        self.assertTrue(np.isfinite(predict(policy, x*1e200)).all())

    def test_small_nonfinite_or_out_of_range_outcomes_fail(self):
        x = np.zeros((20, len(FEATURES)))
        for xx, yy in ((x[:19], np.zeros(19)), (x, np.ones(20)*2), (x*np.nan, np.zeros(20))):
            with self.assertRaises(ValueError):
                fit(xx, yy, 1.0)


if __name__ == "__main__":
    unittest.main()

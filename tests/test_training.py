import unittest

from context_stamps.spherical import SphericalStamp
from context_stamps.training import fit_pairwise
from stamps import Family


class PairwiseTests(unittest.TestCase):
    def test_loss_descent_and_nonnegative_weights(self):
        template = SphericalStamp.encode({"a": [1, 2], "b": [2, 1]},
                                         {"a": Family("a", 2, 64), "b": Family("b", 2, 64)})
        model, history = fit_pairwise(template, [[.5, -.1], [.4, -.2], [.3, .1]], epochs=100)
        self.assertLess(history[-1]["loss"], history[0]["loss"])
        self.assertTrue(all(w >= 0 for w in model.coefficients))
        for inputs in ([[float("nan"), 0]], [[2, 0]], []):
            with self.assertRaises(ValueError):
                fit_pairwise(template, inputs)

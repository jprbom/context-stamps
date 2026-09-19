import importlib.util
import unittest

from context_stamps import Family, stamp_vector


@unittest.skipUnless(importlib.util.find_spec("numpy"), "optional numpy not installed")
class LearningTests(unittest.TestCase):
    def test_fitted_codes_match_projection_and_roundtrip(self):
        import numpy as np

        from context_stamps.learning import fit_family

        x = np.random.default_rng(22).normal(size=(80, 12))
        family = fit_family(x, encoder="fixture", bits=8)
        restored = Family.from_json(family.to_json())
        heldout = np.random.default_rng(23).normal(size=(10, 12))
        direct = (heldout - np.asarray(family.mean)) @ np.asarray(family.planes).T
        for row, projection in zip(heldout, direct):
            expected = sum(1 << i for i, value in enumerate(projection) if value >= 0)
            self.assertEqual(stamp_vector(row, restored).value, expected)

    def test_fitting_is_reproducible_and_validates_shapes(self):
        import numpy as np

        from context_stamps.learning import fit_family

        x = np.random.default_rng(2).normal(size=(80, 12))
        self.assertEqual(
            fit_family(x, encoder="f", bits=8).identity, fit_family(x, encoder="f", bits=8).identity
        )
        for invalid in (x[:2], [[float("nan")] * 12] * 20):
            with self.assertRaises(ValueError):
                fit_family(invalid, encoder="f", bits=8)
        family = fit_family(x, encoder="f", bits=128, method="centered")
        self.assertEqual(family.bits, 128)


if __name__ == "__main__":
    unittest.main()

import math
import unittest

from context_stamps import (
    HybridScoreProfile,
    RelationEdge,
    RelationMap,
    certify_hybrid_scope,
    standardize_scores,
)
from context_stamps.learning import fit_product_families
from context_stamps.spherical import SphericalStamp


class HybridTests(unittest.TestCase):
    def test_standardization_and_fusion(self):
        values = standardize_scores([1, 2, 3])
        self.assertAlmostEqual(sum(values), 0)
        self.assertAlmostEqual(sum(value * value for value in values) / 3, 1)
        profile = HybridScoreProfile(.75)
        self.assertEqual(profile.fuse([1, 2], [2, 1])[1] > profile.fuse([1, 2], [2, 1])[0], True)
        self.assertEqual(standardize_scores([2, 2]), (0, 0))
        self.assertEqual(profile.identity, HybridScoreProfile(.75).identity)

    def test_fusion_validation(self):
        for value in (-.1, 1.1, float("nan"), True):
            with self.assertRaises(ValueError):
                HybridScoreProfile(value)
        with self.assertRaises(ValueError):
            HybridScoreProfile().fuse([1], [1, 2])
        with self.assertRaises(ValueError):
            standardize_scores([float("inf")])

    def test_scope_certificate_enables_gain_and_abstains_on_regression(self):
        profile = HybridScoreProfile(.75)
        dense = [0.2 + (index % 7) / 100 for index in range(120)]
        improved = [value + .04 for value in dense]
        regressed = [value - .01 for value in dense]
        enabled = certify_hybrid_scope("research-corpus-v1", profile, dense, improved, resamples=1000)
        abstained = certify_hybrid_scope("citation-corpus-v1", profile, dense, regressed, resamples=1000)
        self.assertTrue(enabled.enabled)
        self.assertEqual(enabled.selected_profile, profile.identity)
        self.assertFalse(abstained.enabled)
        self.assertIsNone(abstained.selected_profile)
        self.assertNotEqual(enabled.validation_digest, abstained.validation_digest)


class RelationTests(unittest.TestCase):
    def setUp(self):
        self.relations = RelationMap([
            RelationEdge("code", "requirement", "implements", 2),
            RelationEdge("code", "test", "verified_by", 1),
            RelationEdge("test", "fixture", "uses", 1),
        ])

    def test_diffusion_is_normalized_versioned_and_directional(self):
        code = self.relations.encode("code", dim=64, hops=3)
        requirement = self.relations.encode("requirement", dim=64, hops=3)
        self.assertAlmostEqual(math.sqrt(sum(value * value for value in code)), 1)
        self.assertNotEqual(code, requirement)
        self.assertEqual(code, self.relations.encode("code", dim=64, hops=3))
        changed = RelationMap([*self.relations.edges, RelationEdge("code", "policy", "governed_by")])
        self.assertNotEqual(self.relations.revision, changed.revision)
        self.assertNotEqual(code, changed.encode("code", dim=64, hops=3))

    def test_relation_bounds(self):
        with self.assertRaises(ValueError):
            RelationMap([])
        with self.assertRaises(ValueError):
            self.relations.encode("missing")
        with self.assertRaises(ValueError):
            RelationEdge("a", "b", "kind", 0)

    def test_product_itq_families_encode_256_bits(self):
        try:
            import numpy as np
        except ImportError:
            self.skipTest("optional numpy not installed")
        rows = 40
        matrices = {
            f"view{i}": np.array([[math.sin((i + 1) * row + column) for column in range(40)]
                                   for row in range(rows)])
            for i in range(8)
        }
        families = fit_product_families(
            matrices,
            encoders={name: name + "-encoder-v1" for name in matrices},
            allocation={name: 32 for name in matrices},
            iterations=2,
        )
        self.assertEqual(sum(family.bits for family in families.values()), 256)
        stamp = SphericalStamp.encode({name: values[0] for name, values in matrices.items()}, families)
        self.assertEqual(stamp.bits, 256)


if __name__ == "__main__":
    unittest.main()

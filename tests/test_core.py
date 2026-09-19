import math
import unittest

from context_stamps import Family, HashingEncoder, Stamp, content_digest, hamming, stamp_vector


class CoreTests(unittest.TestCase):
    def test_golden_serialized_stamp(self):
        value = stamp_vector([1, 2, 3], Family("fixture-v1", 3, bits=16))
        self.assertEqual(
            str(value), "cs1:39275745730fae6b79303c36e3e1a9c7c63d58912434d40820cb1dea21243fcd:16:6f5e"
        )

    def test_roundtrip_and_opposites(self):
        family = Family("fixture-v1", 3, bits=16)
        a = stamp_vector([1, 2, 3], family)
        b = stamp_vector([-1, -2, -3], family)
        self.assertEqual(hamming(a, b), 16)
        self.assertEqual(Stamp.parse(str(a)), a)
        self.assertEqual(Family.from_json(family.to_json()), family)

    def test_family_mismatch(self):
        a = stamp_vector([1, 2], Family("v1", 2))
        b = stamp_vector([1, 2], Family("v2", 2))
        with self.assertRaises(ValueError):
            hamming(a, b)

    def test_reject_invalid_vectors(self):
        for vector in ([0, 0], [1], [1, math.nan], [math.inf, 1]):
            with self.subTest(vector=vector), self.assertRaises(ValueError):
                stamp_vector(vector, Family("v1", 2))

    def test_exact_content_preserves_whitespace_and_edits(self):
        self.assertNotEqual(content_digest("a > b"), content_digest("a >= b"))
        self.assertNotEqual(content_digest("  return x"), content_digest("return x"))

    def test_invalid_serialization(self):
        for text in ("bad", "cs2:abc:128:ff", "cs1:" + "a" * 64 + ":8:fff"):
            with self.subTest(text=text), self.assertRaises(ValueError):
                Stamp.parse(text)

    def test_lexical_encoder_deterministic(self):
        a, b = HashingEncoder(), HashingEncoder()
        self.assertEqual(a.encode("Hello world!"), b.encode("Hello world!"))
        self.assertNotEqual(a.encode("A B"), a.encode("B A"))
        with self.assertRaises(ValueError):
            a.encode(" ")

    def test_projection_law_empirical(self):
        # Orthogonal directions should disagree in about half the Gaussian projections.
        family = Family("orthogonal-fixture", 2, bits=512)
        d = hamming(stamp_vector([1, 0], family), stamp_vector([0, 1], family))
        self.assertTrue(200 < d < 312, d)


if __name__ == "__main__":
    unittest.main()

import unittest

import numpy as np

from context_stamps.activation import StampSchema
from context_stamps.packed_index import PackedStampIndex
from context_stamps.spherical import SphericalStamp
from stamps import Family, Stamp


class IndexTests(unittest.TestCase):
    def test_scalar_parity_and_explicit_filter(self):
        families = {"a": Family("a", 2, 64), "b": Family("b", 2, 64)}
        query = SphericalStamp.encode({"a": [1, 2], "b": [2, 1]}, families)
        schema = StampSchema.for_stamp(query)
        rng = np.random.default_rng(912)
        codes = rng.integers(0, 256, (501, 16), dtype=np.uint8)
        index = PackedStampIndex(schema, [str(i) for i in range(501)], codes)
        eligible = list(range(0, 501, 3))
        scalar = []
        for i in eligible:
            candidate = SphericalStamp(tuple((name, Stamp(family, bits, int.from_bytes(
                codes[i, j * 8:j * 8 + 8].tobytes(), "big"))) for j, (name, family, bits) in enumerate(schema.views)))
            scalar.append((i, query.score(candidate, {"a": 2, "b": 1})))
        expected = sorted(scalar, key=lambda row: (-row[1], row[0]))[:10]
        result = index.search(query, eligible=eligible, weights={"a": 2, "b": 1}, chunk_size=17)
        self.assertEqual([r["row"] for r in result], [i for i, _ in expected])
        np.testing.assert_allclose([r["score"] for r in result], [s for _, s in expected])
        codes[:] = 0
        self.assertEqual(index.search(query, eligible=eligible, weights={"a": 2, "b": 1}, chunk_size=17), result)
        with self.assertRaises(ValueError):
            index.search(query, eligible=None)
        self.assertEqual(index.search(query, eligible=[]), [])


if __name__ == "__main__":
    unittest.main()

import unittest
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from context_stamps.residual_index import ResidualIndex


class ResidualTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(197)
        self.data = rng.normal(size=(301, 48)).astype(np.float32)
        self.data /= np.linalg.norm(self.data, axis=1, keepdims=True)
        self.index = ResidualIndex(self.data, encoder_id="test-v1")

    def search(self, q, **kwargs):
        args = dict(encoder_id="test-v1", eligible=np.arange(len(self.data)), fetch=lambda ids: self.data[ids])
        args.update(kwargs)
        return self.index.search(q, **args)

    def test_dense_parity_and_filters(self):
        for q in self.data[:20]:
            ids = np.arange(0, len(self.data), 2)
            scores = np.einsum("ij,j->i", self.data[ids], q.astype(np.float64), dtype=np.float64)
            expected = ids[np.lexsort((ids, -scores))[:10]].tolist()
            actual = self.search(q, eligible=ids)
            self.assertEqual(actual["rows"], expected)
            self.assertLessEqual(actual["refined_rows"], len(ids))

    def test_ties_refine_all_and_preserve_order(self):
        data = np.tile(self.data[0], (80, 1))
        index = ResidualIndex(data, encoder_id="ties")
        result = index.search(data[0], encoder_id="ties", eligible=np.arange(80), fetch=lambda ids: data[ids])
        self.assertEqual(result["rows"], list(range(10)))
        self.assertEqual(result["refined_rows"], 80)

    def test_corrupt_store_fails_closed(self):
        with self.assertRaises(ValueError):
            self.search(self.data[0], fetch=lambda ids: -self.data[ids])

    def test_bounds_and_identity(self):
        for kwargs in ({"encoder_id": "other"}, {"eligible": None}, {"eligible": [-1]},
                       {"eligible": [301]}, {"eligible": [.5]}, {"limit": 101}):
            with self.assertRaises(ValueError):
                self.search(self.data[0], **kwargs)
        for q in (np.zeros(48), np.full(48, np.nan), self.data[0] * 2):
            with self.assertRaises(ValueError):
                self.search(q)
        self.assertEqual(self.search(self.data[0], eligible=[])["rows"], [])
        self.assertEqual(self.search(self.data[0], limit=0)["rows"], [])

    def test_readonly_concurrent_queries(self):
        with ThreadPoolExecutor(max_workers=4) as pool:
            rows = list(pool.map(self.search, self.data[:12]))
        self.assertEqual([r["rows"][0] for r in rows], list(range(12)))
        with self.assertRaises(ValueError):
            self.index._codes[0, 0] = 0

    def test_close_scores_at_maximum_dimension(self):
        rng = np.random.default_rng(448)
        q = rng.normal(size=4096).astype(np.float32)
        q /= np.linalg.norm(q)
        values = q + rng.normal(scale=1e-6, size=(40, 4096)).astype(np.float32)
        values /= np.linalg.norm(values, axis=1, keepdims=True)
        index = ResidualIndex(values, encoder_id="near-ties")
        rows = np.arange(len(values))
        result = index.search(q, encoder_id="near-ties", eligible=rows, fetch=lambda ids: values[ids])
        scores = np.einsum("ij,j->i", values, q.astype(np.float64), dtype=np.float64)
        self.assertEqual(result["rows"], np.lexsort((rows, -scores))[:10].tolist())


if __name__ == "__main__":
    unittest.main()

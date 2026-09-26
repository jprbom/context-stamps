import hashlib
import json
import random
import unittest

from context_stamps.greedy_budget import pack_in_order


class GreedyBudgetTests(unittest.TestCase):
    def run_case(self, *, size=20, budget=60, limit=20, batch=8, count=None, identities=None):
        count = count or (lambda text: len(text))
        def render(indices):
            return json.dumps(indices)
        result = pack_in_order(size, render, lambda texts: [count(s) for s in texts], count,
            budget=budget, max_selected=limit, batch_size=batch,
            identity=None if identities is None else lambda i: identities[i])
        expected, seen = [], set()
        for i in range(size):
            key = i if identities is None else identities[i]
            if len(expected) == limit:
                break
            if key not in seen and count(render(tuple(expected+[i]))) <= budget:
                expected.append(i)
                seen.add(key)
        self.assertEqual(result.indices, tuple(expected))
        self.assertEqual(result.text, render(tuple(expected)))
        self.assertEqual(result.tokens, count(result.text))
        return result

    def test_nonmonotone_counts_match_sequential_oracle(self):
        for seed in range(100):
            rng = random.Random(seed)
            # Count can decrease after adding an item. No cumulative-weight
            # or prefix-monotonicity assumption is valid for this oracle.
            def count(text):
                return int(hashlib.sha256((str(seed)+text).encode()).hexdigest()[:8], 16) % 100 if text != "[]" else 0
            identities = [rng.randrange(12) for _ in range(30)]
            for batch in (1, 2, 8, 16):
                self.run_case(size=30, budget=50, limit=12, batch=batch, count=count, identities=identities)

    def test_rejected_identity_can_be_accepted_later(self):
        values = {"[]": 0, "[0]": 90, "[1]": 10, "[2]": 90, "[1, 2]": 20}
        result = self.run_case(size=3, budget=30, identities=("same", "other", "same"), count=lambda s: values[s])
        self.assertEqual(result.indices, (1, 2))

    def test_stale_speculative_counts_are_discarded_after_fit(self):
        values = {"[]": 0, "[0]": 90, "[1]": 10, "[2]": 10, "[3]": 10,
                  "[1, 2]": 90, "[1, 3]": 20}
        self.assertEqual(self.run_case(size=4, budget=30, count=lambda s: values[s]).indices, (1, 3))

    def test_rejection_batches_reduce_callback_rounds(self):
        result = self.run_case(size=40, budget=2, count=lambda s: 0 if s == "[]" else 100)
        self.assertEqual(result.indices, ())
        self.assertLess(result.batches, 10)
        self.assertEqual(result.counted_candidates, 40)

    def test_empty_candidates_and_zero_selection(self):
        self.assertEqual(self.run_case(size=0).indices, ())
        self.assertEqual(self.run_case(limit=0).indices, ())

    def test_prompt_overhead_is_counted_even_without_selected_items(self):
        with self.assertRaises(ValueError):
            self.run_case(size=0, budget=1)

    def test_partial_or_invalid_batches_fail_closed(self):
        for callback in (lambda texts: [], lambda texts: [True], lambda texts: [-1], lambda texts: [float("nan")]):
            with self.assertRaises(ValueError):
                pack_in_order(1, lambda _: "x", callback, lambda _: 1, budget=5, max_selected=1)

    def test_final_native_mismatch_fails_closed(self):
        with self.assertRaises(ValueError):
            pack_in_order(1, lambda _: "x", lambda texts: [1], lambda _: 2, budget=5, max_selected=1)

    def test_limits_and_large_rendering_are_rejected(self):
        for kwargs in (dict(size=True), dict(size=8193), dict(batch_size=0), dict(batch_size=17), dict(budget=-1)):
            params = dict(size=1, budget=10, max_selected=1) | kwargs
            with self.assertRaises(ValueError):
                pack_in_order(render=lambda _: "x", count_batch=lambda texts: [1], count_one=lambda _: 1, **params)
        with self.assertRaises(ValueError):
            pack_in_order(1, lambda _: "x"*(1024*1024+1), lambda texts: [1], lambda _: 1, budget=1, max_selected=1)


if __name__ == "__main__":
    unittest.main()

"""Numerical portability must not erase changed outcomes or missing records."""

import unittest

from verify_local_monitor import same_record


class ReplayTests(unittest.TestCase):
    def test_ulp_roundoff_only(self):
        self.assertTrue(same_record({"stat": -1.5686159179138455}, {"stat": -1.5686159179138452}))
        self.assertFalse(same_record({"stat": -1.5686159179138455}, {"stat": -1.568615}))
        self.assertFalse(same_record(float("nan"), float("nan")))
        self.assertFalse(same_record(float("inf"), float("inf")))

    def test_types_outcomes_lengths_and_keys_are_exact(self):
        for left, right in ((False, 0), (True, False), (1, 1.), ([1, 2], [1]),
                            ({"a": 1}, {"b": 1}), ("candidate", "baseline")):
            with self.subTest(left=left, right=right):
                self.assertFalse(same_record(left, right))


if __name__ == "__main__":
    unittest.main()

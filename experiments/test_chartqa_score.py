"""Authored scoring edge cases, independent of the reserved benchmark keys."""

import unittest

from chartqa_score import relaxed_correctness, score_answer

CANARIES = (
    ("100", "104.9", True), ("100", "105", True), ("100", "105.01", False),
    ("-100", "-95", True), ("-100", "-94", False),
    ("25%", "0.25", True), ("25%", "25", False), ("25", "25%", False),
    ("0", "0", True), ("0", "0.0", False), ("0%", "0", False),
    ("BLUE", "blue", True), (" blue ", "blue", False),
    ("1,000", "1000", False), ("1e3", "1000", True),
    (" 2 ", "2", True), ("two", "2", False), ("1", "NaN", False),
    ("NaN", "NaN", False), ("inf", "inf", False), ("Yes", "YES", True),
)


class ScorerTests(unittest.TestCase):
    def test_declared_reference_edge_cases(self):
        for target, prediction, expected in CANARIES:
            with self.subTest(target=target, prediction=prediction):
                self.assertEqual(relaxed_correctness(target, prediction), expected)

    def test_native_exact_metric_is_separate(self):
        value = score_answer([" Blue "], "blue")
        self.assertFalse(value["relaxed"])
        self.assertFalse(value["stripped_exact"])
        self.assertTrue(score_answer([" Blue "], "Blue")["stripped_exact"])
        self.assertFalse(score_answer(["100"], "105")["stripped_exact"])

    def test_abstention_and_reference_validation(self):
        self.assertEqual(score_answer(["0"], None), dict(relaxed=False, stripped_exact=False, abstained=True))
        self.assertTrue(score_answer(["two", "2"], "2")["relaxed"])
        for labels, answer in (([], "1"), ([""], None), ([1], "1"), (["1"], ""), (["1"], True)):
            with self.assertRaises(ValueError):
                score_answer(labels, answer)


if __name__ == "__main__":
    unittest.main()

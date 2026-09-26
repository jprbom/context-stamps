"""Protocol tests, independent of local weights and benchmark answer files."""

import unittest
from dataclasses import replace
from unittest.mock import patch

from ruler_context import Frame, select
from ruler_local_learning import exact_result, strict_score


class LearningProtocolTests(unittest.TestCase):
    def test_exact_result_binds_full_source_and_returns_no_model_call(self):
        view = Frame("", "One of the special magic numbers for a red-cat is: 123. "
                     "One of the special magic numbers for a red-cat is: 789.",
                     "What are all numbers for a red-cat mentioned in the provided text?", "needle")
        selected = select(view)
        with patch("ruler_local_learning.generate", side_effect=AssertionError("no model call")):
            result = exact_result(view, selected)
        self.assertEqual(result["status"], "verified")
        self.assertEqual(result["receipt_bytes"], 32)
        self.assertEqual(result["full_source_sha256"], selected.full_source_sha256)
        self.assertEqual(result["source_count"], 1)
        bad = exact_result(view, replace(selected, answer_values=("123",)))
        self.assertEqual(bad["status"], "unverified")
        self.assertEqual(bad["result"], "")

    def test_qa_abstains_without_fabricating_verification(self):
        view = Frame("", "some evidence", "why?", "qa")
        self.assertEqual(exact_result(view, select(view))["status"], "abstained")

    def test_custom_metric_rejects_native_substring_false_credit(self):
        for prediction in ("1234", "123 123", "Here is 123", "123 456", '["123", "123"]'):
            self.assertFalse(strict_score(prediction, ["123"], "all", False, None))
        for prediction in ('["123", "456"]', "123, 456", "456\n123"):
            self.assertTrue(strict_score(prediction, ["123", "456"], "all", False, None))
        self.assertFalse(strict_score("123", ["123"], "all", True, None))
        self.assertFalse(strict_score("123", ["123"], "all", False, "transport error"))

    def test_multiword_atoms_and_qa_alias_equality(self):
        self.assertTrue(strict_score("ad hoc, red cat", ["red cat", "ad hoc"], "all", False, None))
        self.assertFalse(strict_score("red cats ad hoc", ["red cat", "ad hoc"], "all", False, None))
        self.assertTrue(strict_score("  NEW   YORK ", ["NYC", "New York"], "part", False, None))
        self.assertFalse(strict_score("Maybe New York or Boston", ["New York"], "part", False, None))


if __name__ == "__main__":
    unittest.main()

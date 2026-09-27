"""Metric and failure-denominator checks independent of model inference."""

import unittest

from verify_techqa import ABSTAIN, INVALID, accepted, measure, point_score, reconstruct, replay_canaries


class EvidenceTests(unittest.TestCase):
    def setUp(self):
        self.keys = dict(yes=dict(ANSWERABLE="Y", DOCUMENT="doc", START_OFFSET="10", END_OFFSET="20"),
                         no=dict(ANSWERABLE="N", DOCUMENT="-", START_OFFSET="-", END_OFFSET="-"))
        self.exact = dict(doc_id="doc", score=1.0, start_offset=10, end_offset=20)

    def test_span_overlap_and_wrong_document(self):
        self.assertEqual(point_score(self.keys["yes"], self.exact), (1.0, 1))
        self.assertAlmostEqual(point_score(self.keys["yes"], dict(self.exact, end_offset=15))[0], 2/3)
        self.assertEqual(point_score(self.keys["yes"], dict(self.exact, doc_id="other")), (0.0, 0))
        self.assertEqual(point_score(self.keys["yes"], dict(self.exact, start_offset=30, end_offset=40)), (0.0, 1))

    def test_explicit_abstention_differs_from_missing_or_invalid(self):
        self.assertEqual(point_score(self.keys["no"], ABSTAIN), (1.0, 1))
        for prediction in (None, INVALID, self.exact):
            self.assertEqual(point_score(self.keys["no"], prediction), (0.0, 0))
        self.assertEqual(point_score(self.keys["yes"], ABSTAIN), (0.0, 0))

    def test_native_canaries_are_replayed_independently(self):
        scores = replay_canaries()
        self.assertEqual(scores["exact"], 100)
        self.assertAlmostEqual(scores["partial_span"], 100*5/6)
        self.assertEqual(scores["wrong_document"], 50)
        self.assertEqual(scores["abstain_all"], 50)
        self.assertEqual(scores["answer_negative"], 50)

    def test_quality_and_false_positives_are_separate(self):
        metrics, rows = measure(self.keys, dict(yes=ABSTAIN, no=ABSTAIN))
        self.assertEqual(metrics["QA_F1"], 50)
        self.assertEqual(metrics["positive_f1"], 0)
        self.assertEqual(metrics["false_positive_count"], 0)
        self.assertEqual(metrics["abstained_positive_count"], 1)
        self.assertEqual(rows, dict(yes=0, no=1))
        with self.assertRaises(ValueError):
            measure(self.keys, dict(yes=self.exact))

    def test_failed_raw_answer_cannot_be_credited_as_abstention(self):
        row = dict(error=None, truncated=False, prediction=None, answer_present=True,
                   final_scope_check=True, check=dict(status="rejected"), bound_prediction=None)
        controls = reconstruct(dict(direct=dict(no=row), cited=dict(no=row)), dict(no=dict(compile_error=None)))
        self.assertEqual(controls["direct_raw"]["no"], INVALID)
        self.assertEqual(controls["cited_raw"]["no"], INVALID)
        self.assertEqual(controls["direct_exact_span"]["no"], ABSTAIN)
        self.assertEqual(controls["cited_checked"]["no"], ABSTAIN)

    def test_compilation_failure_kept_and_acceptance_requires_complete_binding(self):
        row = dict(error=None, truncated=False, prediction=self.exact, answer_present=True,
                   final_scope_check=True, check=dict(status="source_bound"), bound_prediction=self.exact)
        self.assertTrue(accepted(row))
        for change in (dict(error="failed"), dict(truncated=True), dict(final_scope_check=False),
                       dict(check=dict(status="rejected")), dict(bound_prediction=None)):
            self.assertFalse(accepted(dict(row, **change)))
        controls = reconstruct(dict(direct=dict(yes=row), cited=dict(yes=row)), dict(yes=dict(compile_error="failed")))
        self.assertEqual(controls["cited_checked"]["yes"], INVALID)
        self.assertEqual(controls["learned_gate"]["yes"], INVALID)


if __name__ == "__main__":
    unittest.main()

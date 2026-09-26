"""Reject altered credit, timing and incomplete paired relation evidence."""

import copy
import gzip
import json
import unittest
from pathlib import Path

from lme_memory import native_scorer
from review_lme_relations import strict_reference_agreement
from verify_lme_relations import validate

ROOT = Path(__file__).resolve().parents[1]


class RelationEvidenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path = ROOT/"evidence/lme-relations-v2"
        cls.payload = json.loads(gzip.decompress((path/"records.json.gz").read_bytes()))
        cls.registration = json.loads((path/"registration.json").read_bytes())
        cls.summaries = json.loads((path/"summary.json").read_bytes())
        cls.review = json.loads((path/"failure-review.json").read_bytes())
        cls.scorer = native_scorer(ROOT/"evidence/lme-memory-v1/upstream/qa_eval_metrics.py")

    def check(self, payload):
        return validate(payload, self.registration, self.summaries, self.review, self.scorer)

    def test_complete_replay(self):
        self.assertEqual(self.check(self.payload)["native_scores_replayed"], 432)

    def test_duplicate_cannot_replace_missing_arm(self):
        payload = copy.deepcopy(self.payload)
        payload["scores"][0] = payload["scores"][1]
        with self.assertRaises(ValueError):
            self.check(payload)

    def test_credit_cannot_be_flipped(self):
        payload = copy.deepcopy(self.payload)
        payload["scores"][0]["correct"] = not payload["scores"][0]["correct"]
        with self.assertRaises(ValueError):
            self.check(payload)

    def test_request_time_cannot_omit_generation(self):
        payload = copy.deepcopy(self.payload)
        payload["scores"][0]["request_wall_seconds"] = 0
        with self.assertRaises(ValueError):
            self.check(payload)

    def test_sufficiency_claim_refused(self):
        payload = copy.deepcopy(self.payload)
        payload["scores"][0]["sufficient_context_certified"] = True
        with self.assertRaises(ValueError):
            self.check(payload)

    def test_secondary_diagnostic_rejects_extra_phrase(self):
        key = dict(answer="title, forum", eval_function="norm_phrase_set_match")
        self.assertTrue(strict_reference_agreement("Forum; Title", key, self.scorer))
        self.assertFalse(strict_reference_agreement("Title, Body, Forum", key, self.scorer))
        self.assertFalse(strict_reference_agreement("UNKNOWN", key, self.scorer))

    def test_secondary_order_and_native_extraction_preserved(self):
        key = dict(answer="first, last", eval_function="norm_phrase_set_match_ordered")
        self.assertTrue(strict_reference_agreement(r"\boxed{first, last}", key, self.scorer))
        self.assertFalse(strict_reference_agreement("last, first", key, self.scorer))
        self.assertFalse(strict_reference_agreement("first, middle, last", key, self.scorer))


if __name__ == "__main__":
    unittest.main()

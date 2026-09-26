"""Reject tampering with the published paired structural-memory results."""

import copy
import gzip
import json
import unittest
from pathlib import Path

from lme_memory import native_scorer
from verify_lme_structure import validate

ROOT = Path(__file__).resolve().parents[1]


class StructureEvidenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        root = ROOT/"evidence/lme-structure-v1"
        cls.payload = json.loads(gzip.decompress((root/"records.json.gz").read_bytes()))
        cls.registration = json.loads((root/"registration.json").read_bytes())
        cls.summaries = json.loads((root/"summary.json").read_bytes())
        cls.scorer = native_scorer(ROOT/"evidence/lme-memory-v1/upstream/qa_eval_metrics.py")

    def check(self, payload):
        return validate(payload, self.registration, self.summaries, self.scorer)

    def test_complete_replay(self):
        self.assertEqual(self.check(self.payload)["native_scores_replayed"], 288)

    def test_duplicate_arm_cannot_replace_missing_arm(self):
        payload = copy.deepcopy(self.payload)
        payload["scores"][0] = payload["scores"][1]
        with self.assertRaises(ValueError):
            self.check(payload)

    def test_claimed_credit_requires_native_score(self):
        payload = copy.deepcopy(self.payload)
        payload["scores"][0]["correct"] = not payload["scores"][0]["correct"]
        with self.assertRaises(ValueError):
            self.check(payload)

    def test_no_sufficiency_promotion(self):
        payload = copy.deepcopy(self.payload)
        payload["scores"][0]["sufficient_context_certified"] = True
        with self.assertRaises(ValueError):
            self.check(payload)

    def test_reference_duplication_refused(self):
        payload = copy.deepcopy(self.payload)
        payload["references"].append(payload["references"][0])
        with self.assertRaises(ValueError):
            self.check(payload)


if __name__ == "__main__":
    unittest.main()

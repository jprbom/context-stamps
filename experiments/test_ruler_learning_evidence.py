"""Evidence replay rejects changed scores, omitted treatments and fake activation."""

import copy
import gzip
import json
import unittest

from verify_ruler_learning import ROOT, validate, verify


class LearningEvidenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.path = ROOT/"evidence/ruler-local-learning-v1"
        cls.records = json.loads(gzip.decompress((cls.path/"records.json.gz").read_bytes()))
        cls.args = [json.loads((cls.path/f"{name}.json").read_bytes()) for name in
                    ("registration", "registered-trial", "policy", "promotion", "summary")]

    def test_exact_published_replay(self):
        self.assertTrue(verify(self.path)["candidate_inactive"])

    def test_missing_treatment_or_changed_score_rejected(self):
        for change in ("missing", "score"):
            records = copy.deepcopy(self.records)
            if change == "missing":
                records["holdout"].pop()
            else:
                records["holdout"][0]["strict_correct"] = not records["holdout"][0]["strict_correct"]
            with self.assertRaises(ValueError):
                validate(records, *self.args)

    def test_changed_policy_and_cost_are_rejected(self):
        for change in ("policy", "cost"):
            args = copy.deepcopy(self.args)
            if change == "policy":
                args[2]["policy"]["choices"][0][1] = "reader"
            else:
                args[-1]["update_tokens"] = 0
            with self.assertRaises(ValueError):
                validate(self.records, *args)

    def test_unqualified_activation_cannot_be_reported_as_success(self):
        args = copy.deepcopy(self.args)
        args[-1]["promoted"] = True
        with self.assertRaises(ValueError):
            validate(self.records, *args)


if __name__ == "__main__":
    unittest.main()

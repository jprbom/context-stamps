"""Tamper and omission checks for the published paired comparison."""

import copy
import gzip
import json
import unittest

from verify_ruler_development import ROOT, validate


class EvidenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path = ROOT/"evidence/ruler-development-v1"
        cls.records = json.loads(gzip.decompress((path/"records.json.gz").read_bytes()))
        cls.summary = json.loads((path/"summary.json").read_bytes())

    def test_actual_native_replay(self):
        self.assertTrue(validate(self.records, self.summary)["native_metrics_match"])

    def test_missing_or_duplicate_arm_is_rejected(self):
        records = copy.deepcopy(self.records)
        records["generations"][-1] = records["generations"][0]
        with self.assertRaises(ValueError):
            validate(records, self.summary)

    def test_omitting_truncation_or_inflating_success_is_rejected(self):
        for field in ("truncations", "calls"):
            summary = copy.deepcopy(self.summary)
            summary[field] -= 1
            with self.assertRaises(ValueError):
                validate(self.records, summary)
        summary = copy.deepcopy(self.summary)
        summary["overall"]["runtime"]["complete_credit"] += 1
        with self.assertRaises(ValueError):
            validate(self.records, summary)

    def test_reconstructable_prompt_tokens_must_not_be_exported(self):
        records = copy.deepcopy(self.records)
        records["generations"][0]["response"]["context"] = [123, 456]
        with self.assertRaises(ValueError):
            validate(records, self.summary)

    def test_wrong_prompt_token_count_is_rejected(self):
        records = copy.deepcopy(self.records)
        records["generations"][0]["response"]["prompt_eval_count"] += 1
        with self.assertRaises(ValueError):
            validate(records, self.summary)


if __name__ == "__main__":
    unittest.main()

"""Evidence tampering must fail native score and local fitting replay."""

import copy
import gzip
import json
import unittest
from pathlib import Path

from lme_memory import native_scorer
from verify_lme_memory import validate, verify

ROOT = Path(__file__).resolve().parents[1]/"evidence/lme-memory-v1"


class EvidenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.payload = json.loads(gzip.decompress((ROOT/"records.json.gz").read_bytes()))
        cls.registration = json.loads((ROOT/"registration.json").read_bytes())
        cls.policy = json.loads((ROOT/"policy.json").read_bytes())
        cls.choices = json.loads((ROOT/"holdout-choices.json").read_bytes())
        cls.summary = json.loads((ROOT/"summary.json").read_bytes())
        cls.scorer = native_scorer(ROOT/"upstream/qa_eval_metrics.py")

    def check(self, payload=None, registration=None, policy=None):
        return validate(self.payload if payload is None else payload,
                        self.registration if registration is None else registration,
                        self.policy if policy is None else policy,
                        self.choices, self.summary, self.scorer)

    def test_public_replay(self):
        self.assertEqual(verify(ROOT)["native_scores_replayed"], 1104)

    def test_changed_prediction_rejected(self):
        p = copy.deepcopy(self.payload)
        row = next(r for r in p["scores"] if r["correct"])
        row["prediction"] = "UNSUPPORTED_NULL_OUTPUT"
        with self.assertRaises(ValueError):
            self.check(payload=p)

    def test_missing_counterfactual_rejected(self):
        p = copy.deepcopy(self.payload)
        p["scores"].pop()
        with self.assertRaises(ValueError):
            self.check(payload=p)

    def test_holdout_in_training_rejected(self):
        p = copy.deepcopy(self.policy)
        p["training_observations"][0]["cluster_id"] = self.registration["holdout_ids"][0]
        with self.assertRaises(ValueError):
            self.check(policy=p)

    def test_invented_coefficient_or_activation_rejected(self):
        p = copy.deepcopy(self.policy)
        p["policy"]["coefficients"].append(0.)
        with self.assertRaises(ValueError):
            self.check(policy=p)
        p = copy.deepcopy(self.policy)
        p["active"] = True
        with self.assertRaises(ValueError):
            self.check(policy=p)


if __name__ == "__main__":
    unittest.main()

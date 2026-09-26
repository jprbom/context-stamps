import copy
import json
import unittest
from pathlib import Path

from verify_batched_packing import study_contexts, validate

ROOT = Path(__file__).resolve().parents[1]


class PackingEvidenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path = ROOT/"evidence/batched-packing-v1"
        cls.rows = [json.loads(s) for s in (path/"records.jsonl").read_bytes().splitlines()]
        cls.plan = json.loads((path/"registration.json").read_bytes())
        cls.summary = json.loads((path/"summary.json").read_bytes())
        cls.contexts = study_contexts()

    def check(self, rows):
        return validate(rows, self.plan, self.summary, self.contexts)

    def test_replay(self):
        self.assertEqual(self.check(self.rows)["comparison_pairs"], 432)

    def test_changed_prompt_is_rejected(self):
        rows = copy.deepcopy(self.rows)
        rows[0]["prompt_sha256"] = "0"*64
        with self.assertRaises(ValueError):
            self.check(rows)

    def test_duplicate_does_not_replace_a_repeat(self):
        rows = copy.deepcopy(self.rows)
        rows[0] = rows[1]
        with self.assertRaises(ValueError):
            self.check(rows)

    def test_incomplete_elapsed_time_is_rejected(self):
        rows = copy.deepcopy(self.rows)
        rows[0]["batched"]["wall_seconds"] = 0
        with self.assertRaises(ValueError):
            self.check(rows)

    def test_negative_cpu_time_is_rejected(self):
        rows = copy.deepcopy(self.rows)
        rows[0]["baseline"]["cpu_seconds"] = -1
        with self.assertRaises(ValueError):
            self.check(rows)

    def test_changed_source_is_rejected(self):
        rows = copy.deepcopy(self.rows)
        rows[0]["selected_sources_sha256"] = "0"*64
        with self.assertRaises(ValueError):
            self.check(rows)


if __name__ == "__main__":
    unittest.main()

import json
import tempfile
import unittest
from pathlib import Path

from public_record_invariants import check


class PublicRecordInvariantTests(unittest.TestCase):
    def setUp(self):
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq
        except ImportError:
            self.skipTest("optional pyarrow unavailable")
        self.pa, self.pq = pa, pq
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / "a.json").write_text(json.dumps([{"id": 101}, {"id": 102}]), encoding="utf-8")
        (self.root / "b.csv").write_text("user_id\n102\n103\n", encoding="utf-8")
        pq.write_table(pa.table({"userId": [103, 104]}), self.root / "c.parquet")
        self.sources = [(self.root / "a.json", "id"), (self.root / "b.csv", "user_id"),
                        (self.root / "c.parquet", "userId")]
        self.output = self.root / "out.parquet"
        self.report = self.root / "conflicts.json"
        self.report.write_text('{"total_conflicts":0,"conflicts":[]}', encoding="utf-8")

    def output_ids(self, values):
        self.pq.write_table(self.pa.table({"user_id": values, "name": ["x"] * len(values)}),
                            self.output)

    def test_complete_and_missing_coverage(self):
        self.output_ids([101, 102, 103, 104])
        result = check(self.sources, self.output, self.report,
                       required_columns=("user_id", "name"))
        self.assertEqual(result["status"], "complete")
        self.assertEqual(result["expected_count"], 4)
        self.output_ids([101, 102, 103, 103, 999])
        result = check(self.sources, self.output, self.report,
                       required_columns=("user_id", "name"))
        self.assertEqual(result["status"], "incomplete")
        self.assertEqual(result["missing_ids"], [104])
        self.assertEqual(result["unexpected_ids"], [999])
        self.assertEqual(result["duplicate_ids"], [103])

    def test_report_count_and_schema(self):
        self.output_ids([101, 102, 103, 104])
        self.report.write_text('{"total_conflicts":1,"conflicts":[]}', encoding="utf-8")
        result = check(self.sources, self.output, self.report,
                       required_columns=("user_id", "name", "email"))
        self.assertEqual(result["status"], "incomplete")
        self.assertFalse(result["conflict_count_matches_list"])
        self.assertEqual(result["missing_columns"], ["email"])


if __name__ == "__main__":
    unittest.main()

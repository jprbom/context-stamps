"""Validate experiment grading without calling a model or executing generated code."""

import unittest

from experiments.run_local_tasks import check, evaluate_expression, fixtures, prepare


class LocalTaskTests(unittest.TestCase):
    def test_boundary_grading(self):
        case = next(row for row in fixtures() if row["kind"] == "code")
        self.assertEqual(check(case, '{"expression":"value > threshold"}')["correct"], 1)
        stale = check(case, '{"expression":"value >= threshold"}')
        self.assertEqual(stale["correct"], 0)
        self.assertEqual(stale["stale_answer"], 1)

    def test_reject_executable_syntax(self):
        for expression in [
            "__import__('os')",
            "value.real",
            "[x for x in ()]",
            "(lambda: True)()",
            "value + 1",
            "unknown",
            "10001",
            "'True'",
            "True " * 100,
        ]:
            with self.subTest(expression=expression), self.assertRaises((ValueError, SyntaxError)):
                evaluate_expression(expression, 3, 2)

    def test_invalid_json_is_not_correct(self):
        for output in ["[]", "null", '{"value":true}', "not json"]:
            self.assertEqual(check(fixtures()[0], output)["correct"], 0)

    def test_current_source_and_stale_control(self):
        case = fixtures()[0]
        contexts, timings, ingestion, packet = prepare(case)
        self.assertIn(case["current"], contexts["stamps"])
        self.assertNotIn(case["cached"], contexts["stamps"])
        self.assertIn(case["cached"], contexts["stale_cache_control"])
        self.assertEqual(packet["status"], "current")
        self.assertTrue(all(value >= 0 for value in timings.values()))
        self.assertGreater(ingestion, 0)


if __name__ == "__main__":
    unittest.main()

"""Regression cases for typed numbers, zero semantics and unrepaired failures."""

import unittest

from chartqa_protocol import encoded
from chartqa_scalar_protocol import parse_response, request_body
from chartqa_score import score_answer
from test_chartqa_protocol import response


class ScalarTests(unittest.TestCase):
    def test_numbers_keep_type_and_zero_metric_semantics(self):
        for value in (60, 0, 0.0, -0.0, 3.25, -2, 1e20):
            result = parse_response(response(dict(answer=value)), "direct")
            self.assertEqual(type(result["answer_value"]), type(value))
            self.assertEqual(result["answer"], encoded(value).decode())
            self.assertEqual(result["answer_type"], "number")
        self.assertFalse(score_answer(["0"], parse_response(response(dict(answer=0.0)), "direct")["answer"])["relaxed"])
        self.assertTrue(score_answer(["60"], parse_response(response(dict(answer=60)), "direct")["answer"])["relaxed"])

    def test_strings_are_not_repaired(self):
        for value in ("}}60", '{"sum": 60}', "{20 + 40 = 60}", " 60 ", "Gamma"):
            result = parse_response(response(dict(answer=value)), "memory")
            self.assertEqual(result["answer"], value)
            self.assertEqual(result["answer_type"], "string")
        self.assertFalse(score_answer(["60"], "}}60")["relaxed"])

    def test_invalid_types_and_extra_fields_fail(self):
        for value in (True, False, {}, [60], "", " "*2):
            with self.assertRaises(ValueError):
                parse_response(response(dict(answer=value)), "direct")
        for raw in ('{"answer":NaN}', '{"answer":1e999}', '{"answer":1,"answer":2}', '{"answer":1,"x":2}'):
            with self.assertRaises(ValueError):
                parse_response(response({}, message=dict(role="assistant", content=raw)), "direct")

    def test_metadata_checks_survive_projection(self):
        for change in (dict(model="remote"), dict(done=False), dict(done_reason="length"),
                       dict(eval_count=True), dict(total_duration=None), dict(prompt_eval_count=16384)):
            with self.assertRaises(ValueError):
                parse_response(response(dict(answer=60), **change), "memory")
        with self.assertRaises(ValueError):
            parse_response(response({}, message=dict(role="assistant", content='{"answer":60}', thinking="hidden")), "direct")
        self.assertIsNone(parse_response(response(dict(answer=None)), "direct")["answer"])

    def test_same_contract_for_both_answer_arms(self):
        from test_chartqa_quant import table
        a = request_body("direct", question="How many?", image=b"fixture")
        b = request_body("memory", question="How many?", memory=dict(source_sha256="a"*64, kind="MODEL_OUTPUT", table=table()))
        self.assertEqual(a["messages"][0], b["messages"][0])
        self.assertEqual(a["format"], "json")
        self.assertEqual(a["options"], b["options"])

    def test_lookup_instruction_keeps_executor_strict(self):
        from chartqa_program_protocol import request_body as program_request
        from test_chartqa_quant import table
        memory = dict(source_sha256="a"*64, kind="MODEL_OUTPUT", table=table())
        body = program_request("program", question="Lookup?", memory=memory)
        self.assertIn('there is no operation named cell or lookup', body["messages"][0]["content"])
        self.assertEqual(parse_response(response(dict(program=dict(cell="c1"))), "program",
            question="Lookup?", table=table())["answer"], "0.2")
        with self.assertRaises(ValueError):
            parse_response(response(dict(program=dict(op="cell", cells=["c1"]))), "program",
                question="Lookup?", table=table())


if __name__ == "__main__":
    unittest.main()

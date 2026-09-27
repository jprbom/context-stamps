"""Authored local chart protocol fixtures; no model or benchmark keys opened."""

import copy
import unittest
from unittest.mock import patch

from chartqa_protocol import (
    ANSWER_SCHEMA,
    MODEL,
    TABLE_SCHEMA,
    bind_table,
    digest,
    encoded,
    local_chat,
    parse_response,
    request_body,
)
from test_chartqa_quant import table


def response(content, **changes):
    row = dict(model=MODEL, done=True, done_reason="stop", prompt_eval_count=100, eval_count=20,
               total_duration=1000, load_duration=0, eval_duration=400, prompt_eval_duration=600,
               message=dict(role="assistant", content=encoded(content).decode()))
    row.update(changes)
    return encoded(row)


class ProtocolTests(unittest.TestCase):
    def test_prompts_preserve_evidence_and_schema_order(self):
        body = request_body("direct", question="Which category?", image=b"authored bytes")
        self.assertEqual(body["messages"][1]["content"], "Which category?")
        self.assertEqual(body["format"], ANSWER_SCHEMA)
        self.assertEqual(request_body("extract", image=b"fixture")["format"], TABLE_SCHEMA)
        self.assertLess(encoded(TABLE_SCHEMA).index(b'"id"'), encoded(TABLE_SCHEMA).index(b'"value"'))
        self.assertFalse(body["think"])
        self.assertFalse(body["stream"])

    def test_source_binding_remains_unverified_and_revocation_invalidates(self):
        raw = b"authored image bytes; decode is tested in preparation"
        row = dict(image_sha256=digest(raw), byte_length=len(raw), width=20, height=10)
        state, snapshot, memory, bundle = bind_table(row, raw, table(), "a"*64, "b"*64)
        self.assertTrue(state.is_current(snapshot))
        self.assertEqual(memory["kind"], "MODEL_OUTPUT")
        self.assertIsNone(bundle.views[0].validation_revision)
        self.assertEqual(bundle.views[0].dependencies, (bundle.source.ref,))
        body = request_body("memory", question="Mean?", memory=memory)
        self.assertNotIn("images", body["messages"][1])
        state.set_roles(bundle.source.key, ("private",))
        self.assertFalse(state.is_current(snapshot))
        with self.assertRaisesRegex(ValueError, "identity"):
            bind_table(row, raw+b"altered", table(), "a"*64, "b"*64)

    def test_same_memory_used_for_text_and_arithmetic(self):
        memory = dict(kind="MODEL_OUTPUT", source_sha256="a"*64, table=table())
        a = request_body("memory", question="Sum?", memory=memory)
        b = request_body("program", question="Sum?", memory=memory)
        self.assertEqual(a["messages"][1], b["messages"][1])
        memory["kind"] = "FACT"
        with self.assertRaises(ValueError):
            request_body("memory", question="Sum?", memory=memory)

    def test_program_executes_only_bounded_expression(self):
        result = parse_response(response(dict(program=dict(op="sum", cells=["c0", "c1"]))),
                                "program", question="Sum?", table=table())
        self.assertEqual(result["answer"], "0.3")
        self.assertTrue(result["execution"]["execution_verified"])
        self.assertFalse(result["execution"]["source_semantics_verified"])
        for program in (dict(answer="fake"), dict(python="print(42)")):
            with self.assertRaises(ValueError):
                parse_response(response(dict(program=program)), "program", question="Sum?", table=table())

    def test_partial_or_foreign_responses_and_missing_usage_rejected(self):
        for change in (dict(done=False), dict(done_reason="length"), dict(model="foreign"),
                       dict(eval_count=True), dict(total_duration=None), dict(prompt_eval_count=16384),
                       dict(message=[]), dict(message=dict(role="assistant", content='{"answer":"1"}', thinking="hidden"))):
            with self.assertRaises(ValueError):
                parse_response(response(dict(answer="1"), **change), "direct")
        for raw in (b"[]", b'{}', b'{"message":{"content":"duplicate"},"message":{}}'):
            with self.assertRaises(ValueError):
                parse_response(raw, "direct")

    def test_answer_or_explicit_abstention_no_repair(self):
        self.assertEqual(parse_response(response(dict(answer=None)), "direct")["answer"], None)
        self.assertEqual(parse_response(response(dict(answer=" Blue ")), "direct")["answer"], " Blue ")
        for content in (dict(answer=1), dict(answer=""), dict(answer="1", explanation="extra"), ["1"]):
            with self.assertRaises(ValueError):
                parse_response(response(content), "direct")
        self.assertEqual(parse_response(response(dict(program=None)), "program")["answer"], None)

    def test_extraction_limits_and_complete_inputs(self):
        self.assertEqual(parse_response(response(table()), "extract")["table"], table())
        many = table()
        many["cells"] = [dict(many["cells"][0], id=f"c{i}") for i in range(65)]
        with self.assertRaises(ValueError):
            parse_response(response(many), "extract")
        for kwargs in (dict(question="", image=b"x"), dict(question="Q", image="http://external"),
                       dict(question="Q", image=b"x", memory=table())):
            with self.assertRaises(ValueError):
                request_body("direct", **kwargs)

    def test_transport_fixed_to_loopback_and_foreign_model_refused(self):
        body = request_body("direct", question="Q", image=b"x")
        with patch("chartqa_protocol.urllib.request.build_opener") as build:
            build.return_value.open.return_value.__enter__.return_value.read.return_value = b'{}'
            self.assertEqual(local_chat(body), b'{}')
            request = build.return_value.open.call_args.args[0]
            self.assertEqual(request.full_url, "http://127.0.0.1:11434/api/chat")
            self.assertEqual(build.call_args.args[0].proxies, {})
            with self.assertRaises(ValueError):
                build.call_args.args[1].redirect_request(None, None, None, None, None, "https://external")
            altered = copy.deepcopy(body)
            altered["model"] = "remote"
            with self.assertRaises(ValueError):
                local_chat(altered)
            self.assertEqual(build.return_value.open.call_count, 1)


if __name__ == "__main__":
    unittest.main()

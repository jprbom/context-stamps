"""CPU-only boundaries for the public long-context development pilot."""

import json
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import longbench_eval as runner


class CharacterTokenizer:
    def encode(self, text, add_special_tokens=False):
        return types.SimpleNamespace(ids=list(range(len(text))), offsets=[(i, i+1) for i in range(len(text))])


class LongBenchBoundaryTests(unittest.TestCase):
    def setUp(self):
        self.item = dict(context="reliable evidence " * 300, question="Which evidence is reliable?",
                         choice_A="reliable evidence", choice_B="other", choice_C="neither", choice_D="unknown",
                         answer="A", private_grader_text="NEVER INPUT")
        self.tokenizer = CharacterTokenizer()
        self.templates = runner.templates()

    def test_native_parser_is_not_lenient(self):
        cases = {"**The correct answer is (B)**": "B", "The correct answer is C": "C",
                 "B": None, "the correct answer is (B)": None, "The correct answer is (E)": None,
                 "The correct answer is (A). The correct answer is (D)": "A"}
        for text, expected in cases.items():
            self.assertEqual(runner.extract_answer(text), expected)

    def test_target_and_grader_metadata_do_not_change_any_prompt(self):
        changed = dict(self.item, answer="D", private_grader_text="changed secret")
        for method in runner.METHODS:
            first = runner.build_prompt(self.item, method, self.tokenizer, self.templates)
            second = runner.build_prompt(changed, method, self.tokenizer, self.templates)
            self.assertEqual(first, second)
            self.assertNotIn("NEVER INPUT", first[0])

    def test_no_context_has_no_source_and_bm25_has_complete_budget(self):
        for method in runner.METHODS:
            prompt, record = runner.build_prompt(self.item, method, self.tokenizer, self.templates)
            self.assertEqual(record["input_tokens"], len(prompt))
            if method == "no_context":
                self.assertNotIn(self.item["context"], prompt)
            if method == "bm25":
                self.assertLessEqual(len(prompt), runner.RETRIEVAL_INPUT)
                self.assertGreater(record["selected_chunks"], 0)
                self.assertEqual(record["selected_spans"], sorted(record["selected_spans"]))

    def test_no_silent_truncation_even_for_question_only_overflow(self):
        for method in runner.METHODS:
            item = dict(self.item, question="X" * (runner.MAX_INPUT+1))
            with self.assertRaises(ValueError):
                runner.build_prompt(item, method, self.tokenizer, self.templates)

    def test_evidence_placeholder_is_literal(self):
        prompt = runner.render(self.item, self.templates["full"], "Keep $Q$ and $C_A$ literal.")
        self.assertIn("Keep $Q$ and $C_A$ literal.", prompt)
        self.assertIn(self.item["question"], prompt)

    def test_chunk_coverage_unicode_and_stable_zero_scores(self):
        text = "\u0938\u0902\u0926\u0930\u094d\u092d\U0001f600 " * 200
        pieces = runner.chunks(self.tokenizer, text)
        covered = set()
        for left, right, value in pieces:
            self.assertEqual(value, text[left:right])
            covered.update(range(left, right))
        self.assertEqual(covered, set(range(len(text))))
        self.assertEqual(runner.bm25_order("unseen", pieces), list(range(len(pieces))))
        self.assertEqual(runner.chunks(self.tokenizer, ""), [])

    def test_failures_and_missing_attempts_remain_in_denominator(self):
        rows = [dict(method="full", status="error"), dict(method="full", status="ok", correct=True,
                prediction="A", usage=dict(prompt_eval_count=10, eval_count=3), whole_request_seconds=1.)]
        result = runner.summarize({"samples": [1, 2, 3]}, rows)
        self.assertEqual(result[0]["accuracy"], 1/3)
        self.assertEqual(result[0]["errors"], 1)
        self.assertEqual(result[0]["missing"], 1)
        self.assertEqual(result[1]["accuracy"], 0)

    def test_cached_lengths_require_exact_audit_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "audit.json"
            path.write_text('{}')
            with self.assertRaises(ValueError):
                runner.cached_lengths(path, {})

    def test_cached_lengths_bind_inputs_and_recompute_eligibility(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "audit.json"
            path.write_text(json.dumps(dict(input_hashes={"data": "pinned"}, tokenizers_version="fixture",
                eligibility=[dict(full_input_tokens=20000, eligible=False),
                             dict(full_input_tokens=30000, eligible=False)])))
            with (patch.object(runner, "LENGTH_AUDIT_SHA", runner.file_hash(path)),
                  patch.object(runner.importlib.metadata, "version", return_value="fixture")):
                result = runner.cached_lengths(path, {"data": "pinned"})
                self.assertEqual([r["eligible"] for r in result], [True, False])
                with self.assertRaises(ValueError):
                    runner.cached_lengths(path, {"data": "changed"})
            with (patch.object(runner, "LENGTH_AUDIT_SHA", runner.file_hash(path)),
                  patch.object(runner.importlib.metadata, "version", return_value="changed")):
                with self.assertRaises(ValueError):
                    runner.cached_lengths(path, {"data": "pinned"})

    def test_inference_is_fixed_loopback_raw_no_tools_and_no_proxy(self):
        class Response:
            def __enter__(self):
                return self
            def __exit__(self, *args):
                pass
            def read(self, limit):
                return b'{"done":true}'
        class Opener:
            def open(self, request, timeout):
                self.request = request
                return Response()
        opener = Opener()
        with patch.object(runner.urllib.request, "build_opener", return_value=opener) as build:
            self.assertEqual(runner.generate("fixture"), {"done": True})
        self.assertEqual(opener.request.full_url, "http://127.0.0.1:11434/api/generate")
        body = json.loads(opener.request.data)
        self.assertEqual(body["model"], runner.MODEL)
        self.assertTrue(body["raw"])
        self.assertNotIn("tools", body)
        self.assertEqual(build.call_args.args[0].proxies, {})
        with self.assertRaises(ValueError):
            build.call_args.args[1].redirect_request(None, None, 302, "", {}, "https://remote.invalid")


if __name__ == "__main__":
    unittest.main()

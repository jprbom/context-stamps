"""Adversarial boundaries for the benchmark input adapters (no model calls)."""

import json
import unittest

from ruler_context import Frame, frame, native_score, runtime_context, select, verify_direct


class ContractTests(unittest.TestCase):
    def test_multiword_keys_and_independent_value_verification(self):
        view = Frame("", "One of the special magic numbers for ad hoc-incidence is: 123. "
                     "One of the special magic numbers for ad hoc-incidences is: 999. "
                     "One of the special magic numbers for ad hoc-incidence is: 456.",
                     "What are all numbers for ad hoc-incidence mentioned in the provided text?", "needle")
        self.assertEqual(select(view).answer_values, ("123", "456"))
        self.assertTrue(verify_direct(view, '["456", "123"]'))
        for answer in ('["123"]', '["123", "456", "999"]', '["123", "123", "456"]',
                       '"123 456"', '[123,456]', '{"answer": ["123", "456"]}'):
            self.assertFalse(verify_direct(view, answer))
        changed = Frame("", view.context.replace("456", "457"), view.suffix, "needle")
        self.assertFalse(verify_direct(changed, '["123", "456"]'))

    def test_verifier_rejects_unparsed_statement_and_missing_key(self):
        context = "One of the special magic numbers for red-cat is: 123. "
        suffix = "What are numbers for red-cat mentioned in the provided text?"
        for text in (context + "One of the special magic numbers for blue-cat is: unknown value.",
                     context.replace(" is: ", " is ")):
            self.assertFalse(verify_direct(Frame("", text, suffix, "needle"), '["123"]'))
        self.assertFalse(verify_direct(Frame("", context, suffix.replace("red-cat", "blue-cat"), "needle"), '["123"]'))

    def test_graph_verifier_rejects_cycles_reassignment_missing_parent_and_extra_values(self):
        suffix = "assigned the value 12345 in the text above"
        source = "VAR ABCDE = 12345\nVAR FGHIJ = VAR ABCDE\nVAR KLMNO = VAR FGHIJ"
        view = Frame("", source, suffix, "variables")
        self.assertTrue(verify_direct(view, '["KLMNO", "ABCDE", "FGHIJ"]'))
        self.assertFalse(verify_direct(view, '["ABCDE", "FGHIJ"]'))
        for text in (source + "\nVAR ABCDE = 0", "VAR ABCDE = VAR FGHIJ\nVAR FGHIJ = VAR ABCDE",
                     "VAR ABCDE = VAR FGHIJ", "VAR ABCDE = 12345\nVAR FGHIJ = unknown"):
            self.assertFalse(verify_direct(Frame("", text, suffix, "variables"), '["ABCDE"]'))

    def test_tally_verifier_preserves_multiword_terms_and_rejects_boundary_ties(self):
        words = [f"word {chr(97+i)}" for i in range(10)]
        source = " ".join(f"{i+1}. {word}" for i, word in enumerate(words*2 + ["other"]))
        view = Frame("", source, "Question: top 10", "common")
        self.assertTrue(verify_direct(view, json.dumps(select(view).answer_values)))
        self.assertFalse(verify_direct(view, json.dumps(words[:-1] + ["other"])))
        self.assertFalse(verify_direct(Frame("", source.replace("2. ", "99. ", 1), view.suffix, "common"), json.dumps(words)))
        coded = Frame("", "aaaaaa aaaaaa aaaaaa bbbbbb bbbbbb cccccc ...", "", "frequency")
        answer = '["aaaaaa", "bbbbbb", "cccccc"]'
        self.assertTrue(verify_direct(coded, answer))
        self.assertFalse(verify_direct(Frame("", coded.context + " dddddd", "", "frequency"), answer))
        self.assertFalse(verify_direct(Frame("", coded.context + " bad!", "", "frequency"), answer))
        self.assertFalse(verify_direct(Frame("", "an assertion", "why?", "qa"), '["an assertion"]'))

    def test_needle_requires_all_keys_and_exact_key_identity(self):
        view = Frame("", "One of the special magic numbers for red-cat is: 123. "
                     "One of the special magic numbers for red-catfish is: 456. "
                     "One of the special magic numbers for red-cat is: 789.",
                     "What are all the special magic numbers for red-cat mentioned in the provided text?", "needle")
        result = select(view)
        self.assertEqual(result.direct_answer, "123 789")
        missing = select(Frame("", view.context, view.suffix.replace("red-cat", "red-cat, and blue-cat"), "needle"))
        self.assertEqual(missing.kind, "full")

    def test_dependency_closure_and_ambiguous_reassignment(self):
        view = Frame("", "VAR ABCDE = 12345\nVAR FGHIJ = VAR ABCDE\nVAR KLMNO = VAR FGHIJ",
                     "Question: Find all variables that are assigned the value 12345 in the text above.", "variables")
        selected = select(view)
        self.assertEqual(selected.dependencies, ((1, 0), (2, 1)))
        text, receipt = runtime_context(view, selected, lambda text: len(text.split()))
        self.assertEqual(receipt["receipt_bytes"], 32)
        self.assertEqual(len(receipt["sources"]), 3)
        self.assertIn("VAR ABCDE = 12345", text)
        reassigned = select(Frame("", view.context + "\nVAR ABCDE = 99999", view.suffix, "variables"))
        self.assertEqual(reassigned.kind, "full")

    def test_cycle_or_forward_reference_does_not_get_false_certificate(self):
        view = Frame("", "VAR ABCDE = VAR FGHIJ\nVAR FGHIJ = VAR ABCDE",
                     "assigned the value 12345 in the text above", "variables")
        self.assertEqual(select(view).kind, "full")

    def test_malformed_additional_statement_forces_full_context(self):
        view = Frame("", "One of the special magic numbers for red-cat is: 123. "
                     "One of the special magic numbers for red-cat is: unknown value.",
                     "What is the special magic number for red-cat mentioned in the provided text?", "needle")
        self.assertEqual(select(view).kind, "full")
        view = Frame("", "VAR ABCDE = 12345\nVAR FGHIJ = unknown",
                     "assigned the value 12345 in the text above", "variables")
        self.assertEqual(select(view).kind, "full")

    def test_count_contract_rejects_boundary_ties_and_unparsed_tokens(self):
        suffix = "Question: three words"
        context = "aaaaaa aaaaaa aaaaaa bbbbbb bbbbbb cccccc ..."
        result = select(Frame("", context, suffix, "frequency"))
        self.assertEqual(result.direct_answer, "aaaaaa bbbbbb cccccc")
        self.assertEqual(select(Frame("", context + " dddddd", suffix, "frequency")).kind, "full")
        self.assertEqual(select(Frame("", context + " unexpected!", suffix, "frequency")).kind, "full")

    def test_numbered_words_with_spaces_and_malformed_sequence(self):
        words = [f"word {chr(97+i)}" for i in range(10)]
        context = " ".join(f"{i+1}. {word}" for i, word in enumerate(words*2 + ["other"]))
        self.assertEqual(select(Frame("", context, "Question: top 10", "common")).kind, "statistic")
        self.assertEqual(select(Frame("", context.replace("2. ", "99. ", 1), "Question: top 10", "common")).kind, "full")

    def test_document_questions_keep_all_text_without_semantic_certificate(self):
        context = "evidence. " * 3000
        selected = select(Frame("", context, "why?", "qa"))
        self.assertEqual("".join(selected.texts), context)
        self.assertIsNone(selected.direct_answer)
        self.assertEqual(selected.contract, "no_semantic_sufficiency_certificate")

    def test_frame_excludes_worked_example_and_preserves_round_trip(self):
        start = "Below is a numbered list of words. In these words, some appear more often than others. Memorize the ones that appear most often.\n"
        question = start + "1. demonstration\nQuestion: old\nAnswer: example\n" + start + "1. real\nQuestion: new"
        result = frame(question)
        self.assertEqual(result.context, "1. real")
        self.assertEqual(result.render(result.context), question)
        with self.assertRaises(ValueError):
            frame({"question": question, "outputs": ["hidden answer"]})

    def test_native_score_retains_fractional_and_substring_semantics(self):
        self.assertEqual(native_score("foo", ["foo", "bar"], "all"), .5)
        self.assertEqual(native_score("FOOBAR", ["foo", "other"], "part"), 1.)
        self.assertEqual(native_score("", ["foo"], "all"), 0.)


if __name__ == "__main__":
    unittest.main()

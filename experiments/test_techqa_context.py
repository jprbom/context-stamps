import unittest

from techqa_context import grouped_split, locate_answer, public_input, rank_windows, windows


class TechQAContextTests(unittest.TestCase):
    def setUp(self):
        self.row = dict(QUESTION_ID="q1", QUESTION_TITLE="Engine", QUESTION_BODY="Which cache?",
                        DOC_IDS=["b", "a"], ANSWER="secret target", ANSWERABLE="Y", DOCUMENT="b")
        self.docs = {"a": dict(title="Engine", text="Engine uses café cache. café repeats."),
                     "b": dict(title="Other", text="Other unrelated information.")}

    def test_allowlist_ignores_targets(self):
        first = public_input(self.row)
        self.row.update(ANSWERABLE="N", ANSWER="different", DOCUMENT="a", START_OFFSET=12)
        self.assertEqual(first, public_input(self.row))
        self.assertEqual(set(first), {"id", "question", "doc_ids"})
        self.assertNotIn("secret", str(first))

    def test_duplicate_candidates_deduplicated_and_extra_fields_rejected(self):
        self.row["DOC_IDS"] = ["a", "a"]
        self.assertEqual(public_input(self.row)["doc_ids"], ["a"])
        self.row["DOC_IDS"] = ["a", 1]
        with self.assertRaises(ValueError):
            public_input(self.row)
        with self.assertRaises(ValueError):
            windows(dict(id="q", question="q", doc_ids=["a"], answer="label"), self.docs)

    def test_windows_cover_body_and_preserve_exact_offsets(self):
        rows = windows(public_input(self.row), self.docs, width=3, stride=2)
        for row in rows:
            self.assertEqual(self.docs[row["doc_id"]]["text"][row["start"]:row["end"]], row["text"])
        for doc_id, doc in self.docs.items():
            for index, char in enumerate(doc["text"]):
                if not char.isspace():
                    self.assertTrue(any(r["doc_id"] == doc_id and r["start"] <= index < r["end"] for r in rows))

    def test_bm25_ranking_tie_stable(self):
        rows = windows(public_input(self.row), self.docs)
        self.assertEqual(rank_windows("cache Engine", rows)[0]["doc_id"], "a")
        self.assertEqual(rank_windows("", rows), rank_windows("", list(reversed(rows))))

    def test_exact_unicode_and_quote_offsets(self):
        rows = [dict(doc_id="a", start=10, text="café cache. café repeats.")]
        self.assertEqual(locate_answer("café", rows)["start_offset"], 10)
        pred = locate_answer("café", rows, citations=[dict(source="s0", quote="café repeats.")])
        self.assertEqual(pred["start_offset"], 22)
        self.assertEqual(pred["end_offset"], 26)
        self.assertIsNone(locate_answer("CAFE", rows))
        self.assertIsNone(locate_answer("café", rows, citations=[dict(source="s9", quote="café")]))

    def test_question_title_body_deduplication(self):
        self.row.update(QUESTION_TITLE="same", QUESTION_BODY="same")
        self.assertEqual(public_input(self.row)["question"], "same")

    def test_native_text_alias_and_conflicting_bodies(self):
        expected = public_input(self.row)
        self.row["QUESTION_TEXT"] = self.row.pop("QUESTION_BODY")
        self.assertEqual(public_input(self.row), expected)
        self.row["QUESTION_BODY"] = "different"
        with self.assertRaises(ValueError):
            public_input(self.row)

    def test_grouped_split_withholds_dev_duplicates_and_never_splits_groups(self):
        train = [dict(id=str(i), question=q) for i, q in enumerate(("same", "Same!", "another", "third", "development"))]
        dev = [dict(id="dev", question="Development?")]
        split, audit = grouped_split(train, dev, fit_count=2)
        self.assertEqual(audit["withheld_training_questions"][0]["id"], "4")
        self.assertNotIn("4", split["fit"]+split["calibration"])
        self.assertEqual("0" in split["fit"], "1" in split["fit"])
        self.assertEqual(split["development"], ["dev"])
        self.assertEqual(grouped_split(list(reversed(train)), dev, fit_count=2), (split, audit))


if __name__ == "__main__":
    unittest.main()

"""Boundary checks for public trajectory ingestion and native local scoring."""

import json
import sqlite3
import unittest
from pathlib import Path

from lme_memory import episode, native_scorer, ranked, score, terms
from lme_rank_fast import ranked_fast
from lme_rank_selective import ranked_selective

from context_stamps.trajectory import trajectory_views

ROOT = Path(__file__).resolve().parents[1]


class MemoryBoundaryTests(unittest.TestCase):
    def test_ingestion_whitelist_excludes_thoughts_images_and_labels(self):
        source = dict(id="e1", goal="Set status", outcome="UNVERIFIED_SUCCESS", answer="SECRET_LABEL",
                      states=[dict(state_index=0, accessibility_tree="Status pending", action=None,
                                   url="local", thought="UNVERIFIED_THOUGHT", screenshot="PRIVATE_PATH")])
        views = list(trajectory_views(episode(source)))
        encoded = json.dumps([v.__dict__ for v in views])
        for marker in ("SECRET_LABEL", "UNVERIFIED_THOUGHT", "PRIVATE_PATH", "UNVERIFIED_SUCCESS"):
            self.assertNotIn(marker, encoded)

    def test_fts_query_cannot_use_expressions_or_cross_haystack(self):
        db = sqlite3.connect(":memory:")
        db.execute("CREATE VIRTUAL TABLE memory USING fts5(body,key UNINDEXED,episode UNINDEXED,source UNINDEXED,channel UNINDEXED,steps UNINDEXED,domain UNINDEXED)")
        for key, domain in (("authorized", "web"), ("outside", "web"), ("other-domain", "enterprise")):
            db.execute("INSERT INTO memory VALUES(?,?,?,?,?,?,?)", ("unique status", key, key, "a"*64, "state", "[0]", domain))
        found = ranked(db, 'unique" OR * ); DROP TABLE memory; --', "web", "state", {"authorized"})
        self.assertEqual([v.episode for v in found], ["authorized"])
        self.assertEqual(db.execute("SELECT count(*) FROM memory").fetchone()[0], 3)
        db.close()
        self.assertEqual(terms("the and is"), ())

    def test_native_scoring_and_remote_refusal(self):
        ns = native_scorer(ROOT/"evidence/lme-memory-v1/upstream/qa_eval_metrics.py")
        cases = (
            ("norm_phrase_set_match", r"\boxed{Alpha Beta}", "alpha;beta", True),
            ("norm_phrase_set_match_ordered", r"\boxed{beta then alpha}", "alpha;beta", False),
            ("mc_choice_match", r"\boxed{B}", "B", True),
            ("mc_choice_set_match", r"\boxed{B and C}", "C,B", True),
            ("mc_choice_match", r"\boxed{UNKNOWN}", "UNKNOWN", False),
        )
        for kind, prediction, answer, expected in cases:
            with self.subTest(kind=kind):
                self.assertEqual(score(ns, prediction, dict(eval_function=kind, answer=answer)), expected)
        with self.assertRaises(ValueError):
            score(ns, "anything", dict(eval_function="llm_abstention_checker", answer="x"))
        self.assertNotIn("llm_abstention_checker", ns)
        self.assertNotIn("_create_openai_client", ns)

    def test_native_phrase_credit_is_not_truth_verification(self):
        ns = native_scorer(ROOT/"evidence/lme-memory-v1/upstream/qa_eval_metrics.py")
        # Preserve, document and do not strengthen the upstream metric silently.
        self.assertTrue(score(ns, "The answer is not enabled", dict(eval_function="norm_phrase_set_match", answer="enabled")))

    def test_optimized_search_preserves_all_cutoff_ties_and_scope(self):
        db = sqlite3.connect(":memory:")
        db.execute("CREATE VIRTUAL TABLE memory USING fts5(body,key UNINDEXED,episode UNINDEXED,source UNINDEXED,channel UNINDEXED,steps UNINDEXED,domain UNINDEXED)")
        for i in reversed(range(600)):
            key = f"k{i:04d}"
            db.execute("INSERT INTO memory VALUES(?,?,?,?,?,?,?)", ("same observation", key, key, "a"*64, "state", "[0]", "web"))
        allowed = {f"k{i:04d}" for i in range(600)}
        for scope in (allowed, {"k0001", "k0400"}):
            a = ranked(db, "observation", "web", "state", scope)
            b = ranked_fast(db, "observation", "web", "state", scope)
            self.assertEqual(a, b)
        self.assertEqual(len(ranked_fast(db, "observation", "web", "state", allowed)), 240)
        self.assertEqual(ranked_fast(db, "absent", "web", "state", allowed), [])
        for channel in ("state", "path", "change"):
            self.assertEqual(ranked(db, "observation", "web", channel, allowed),
                             ranked_selective(db, "observation", "web", channel, allowed))
        db.close()


if __name__ == "__main__":
    unittest.main()

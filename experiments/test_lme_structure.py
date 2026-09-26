"""Boundary checks for structural-memory preparation; no reader is called."""

import json
import sqlite3
import unittest
from dataclasses import asdict

from lme_structure import deserialize, generate, ranked

from context_stamps.structured_memory import structured_views
from context_stamps.trajectory import ObservedEpisode, ObservedStep


class StructureExperimentTests(unittest.TestCase):
    def view(self, key):
        return next(structured_views(ObservedEpisode(key, "", (ObservedStep(0, "RootWebArea 'security settings'"),))))

    def test_roundtrip_preserves_exact_source_spans(self):
        v = self.view("allowed")
        self.assertEqual(v, deserialize(json.dumps(dict(structured=True, view=asdict(v)))))

    def test_scope_is_applied_before_topk(self):
        db = sqlite3.connect(":memory:")
        db.execute("CREATE VIRTUAL TABLE memory USING fts5(body,payload UNINDEXED,domain UNINDEXED,episode UNINDEXED,key UNINDEXED)")
        for i in range(300):
            v = self.view(f"episode-{i}")
            db.execute("INSERT INTO memory VALUES (?,?,?,?,?)", (v.body, json.dumps(dict(structured=True, view=asdict(v))), "web", v.episode, v.key))
        result = ranked(db, "security", "web", {"episode-299"})
        self.assertEqual([v.episode for v in result], ["episode-299"])
        self.assertEqual(ranked(db, 'security"; DROP TABLE memory; --', "web", set()), [])
        self.assertEqual(db.execute("SELECT count(*) FROM memory").fetchone()[0], 300)
        db.close()

    def test_unregistered_provider_refused_before_network(self):
        with self.assertRaises(ValueError):
            generate("remote:cloud", "anything")


if __name__ == "__main__":
    unittest.main()

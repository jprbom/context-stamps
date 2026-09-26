"""Boundary tests for scope filtering and relation serialization; no GPU."""

import json
import sqlite3
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

from lme_relations import Retrieval, pack_relations, packet
from verify_relation_sources import check_occurrence, parse

from context_stamps.observation_packets import PagePlan, observation_packets
from context_stamps.trajectory import ObservedEpisode, ObservedStep


class RelationExperimentTests(unittest.TestCase):
    def view(self, episode="local", page="Settings"):
        return next(observation_packets(ObservedEpisode(episode, "", (
            ObservedStep(0, f"RootWebArea '{page}'\n  button 'Security preferences'"),))))

    def test_roundtrip_preserves_source_binding(self):
        v = self.view()
        self.assertEqual(v, packet(json.dumps(asdict(v))))

    def test_source_replay_rejects_wrong_member_or_page(self):
        text = "RootWebArea 'Inventory'\n  table ''\n    row ''\n      columnheader 'Item'\n      columnheader 'Count'"
        ep = ObservedEpisode("x", "", (ObservedStep(0, text),))
        v = asdict(next(observation_packets(ep)))
        nodes = parse(text)
        self.assertEqual(check_occurrence(v, v["occurrences"][0], nodes), 0)
        with self.assertRaises(ValueError):
            check_occurrence(v | dict(body=v["body"].replace("'Count'", "'Price'")), v["occurrences"][0], nodes)
        with self.assertRaises(ValueError):
            check_occurrence(v | dict(page=("'Other'",)), v["occurrences"][0], nodes)

    def test_source_replay_rejects_invented_container(self):
        text = "RootWebArea 'A'\n  region 'First'\n    button 'Save'\n  region 'Second'"
        v = asdict(next(observation_packets(ObservedEpisode("x", "", (ObservedStep(0, text),)))))
        fake = v | dict(body=v["body"].replace("line 2: region 'First'", "line 4: region 'Second'"))
        occurrence = dict(step=0, lines=(1, 3, 4))
        with self.assertRaises(ValueError):
            check_occurrence(fake, occurrence, parse(text))

    def test_authorized_pages_are_filtered_before_cutoff(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)/"test.sqlite"
            db = sqlite3.connect(path)
            for table in ("structure", "relations"):
                db.execute(f"CREATE VIRTUAL TABLE {table} USING fts5(body,payload UNINDEXED,domain UNINDEXED,episode UNINDEXED,view_id UNINDEXED,page UNINDEXED)")
            for i in range(300):
                v = self.view(f"ep-{i}", "Allowed" if i == 299 else "Other")
                db.execute("INSERT INTO relations VALUES (?,?,?,?,?,?)", (v.text, json.dumps(asdict(v)), "web", v.episode, v.key, json.dumps(v.page)))
                db.execute("INSERT INTO structure VALUES (?,?,?,?,?,?)", ("", "", "web", v.episode, v.key, json.dumps(v.page)))
            db.commit()
            db.close()
            search = Retrieval(path)
            plan = PagePlan((("'Allowed'",),), ("allowed",), (1.,), False)
            result = search.search("Security preferences", "web", {"ep-299"}, "relations", plan)
            self.assertEqual([v.episode for v in result], ["ep-299"])
            self.assertEqual(search.search("Security preferences", "web", {"ep-1"}, "relations", plan), [])
            self.assertEqual(search.plan("Allowed", "web", {"ep-299"}).pages, (("'Allowed'",),))
            with self.assertRaises(ValueError):
                search.search("x", "web", {"ep-299"}, "relations; DROP TABLE structure", plan)
            search.db.close()

    def test_complete_prompt_budget_and_selected_source_counts(self):
        class Counter:
            def encode(self, text, **kwargs):
                return SimpleNamespace(ids=range((len(text.encode())+3)//4))
        views = [self.view(f"ep-{i}", f"Page {i}") for i in range(100)]
        rendered, meta = pack_relations("Find a setting", "web", views, Counter())
        self.assertLessEqual(meta["input_tokens"], 6144)
        self.assertLessEqual(len(meta["sources"]), 96)
        self.assertTrue(meta["scope_checked"])
        self.assertFalse(meta["sufficient_context_certified"])
        self.assertIn("Page 0", rendered)
        self.assertEqual(len(meta["sources"]), len({s["view_id"] for s in meta["sources"]}))


if __name__ == "__main__":
    unittest.main()

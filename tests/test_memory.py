import sqlite3
import tempfile
import unittest
from pathlib import Path

from context_stamps import ContextMemory, Family, HashingEncoder


class MemoryTests(unittest.TestCase):
    def test_update_failure_rolls_back_invalidation(self):
        with ContextMemory() as memory:
            old = memory.add("old", source="a")
            memory.add("derived", source="b", dependencies={"a": old["digest"]})
            memory.db.execute("""CREATE TRIGGER reject_update BEFORE UPDATE OF text ON items
                BEGIN SELECT RAISE(ABORT, 'test rejection'); END""")
            with self.assertRaises(sqlite3.IntegrityError):
                memory.add("new", source="a")
            self.assertEqual(memory.get("a")["text"], "old")
            self.assertFalse(memory.get("b")["stale"])

    def test_edit_remains_visible_even_with_identical_vectors(self):
        class Constant:
            identity = "constant-test"
            dim = 2

            def encode(self, text):
                return [1.0, 0.0]

        with ContextMemory(encoder=Constant()) as memory:
            memory.add("a > b", source="one")
            memory.add("a >= b", source="two")
            packet = memory.pack("a", token_budget=1000)
            self.assertEqual([d["reason"] for d in packet.decisions], ["selected", "selected"])

    def test_duplicate_does_not_remove_the_original(self):
        with ContextMemory() as memory:
            memory.add("same text", source="a")
            memory.add("same text", source="b")
            packet = memory.pack("text", token_budget=1000)
            self.assertEqual(packet.text.count("same text"), 1)
            self.assertEqual(packet.decisions[1]["duplicate_of"], "a")
            self.assertEqual(memory.get("b")["text"], "same text")

    def test_updates_and_dependency_invalidation(self):
        with ContextMemory() as memory:
            old = memory.add("old code", source="a")
            middle = memory.add("derived result", source="b", dependencies={"a": old["digest"]})
            memory.add("further result", source="c", dependencies={"b": middle["digest"]})
            memory.add("new code", source="a")
            self.assertTrue(memory.get("b")["stale"])
            self.assertTrue(memory.get("c")["stale"])
            self.assertEqual([r["source"] for r in memory.recall("code")], ["a"])

    def test_missing_and_mismatched_revisions_fail_closed(self):
        with ContextMemory() as memory:
            item = memory.add("manual filter", source="manual", dependencies={"config": "v2"})
            self.assertEqual(memory.recall("filter", revisions={}), [])
            self.assertEqual(
                memory.recall("filter", revisions={"manual": item["digest"], "config": "v1"}), []
            )
            result = memory.recall("filter", revisions={"manual": item["digest"], "config": "v2"})
            self.assertEqual(result[0]["freshness"], "current")

    def test_budget_includes_headers_separators_and_unicode(self):
        with ContextMemory() as memory:
            memory.add("नमस्ते " * 15, source="hindi")
            memory.add("short", source="en")
            for budget in (0, 100, 200, 1000):
                packet = memory.pack("short", token_budget=budget)
                self.assertEqual(packet.tokens, len(packet.text.encode("utf-8")))
                self.assertLessEqual(packet.tokens, budget)
            packet = memory.pack("short", token_budget=140, token_counter=len)
            self.assertLessEqual(len(packet.text), 140)

    def test_same_text_with_different_dependencies_not_deduplicated(self):
        with ContextMemory() as memory:
            memory.add("same output", source="a", dependencies={"env": "v1"})
            memory.add("same output", source="b", dependencies={"env": "v2"})
            self.assertEqual(memory.pack("output", token_budget=1000).text.count("same output"), 2)

    def test_persistence_family_and_delete(self):
        with tempfile.TemporaryDirectory() as folder:
            path = str(Path(folder) / "memory.sqlite")
            with ContextMemory(path) as memory:
                memory.add("keep", source="a")
            with ContextMemory(path) as memory:
                self.assertEqual(memory.get("a")["text"], "keep")
                self.assertTrue(memory.forget("a"))
                self.assertIsNone(memory.get("a"))
            encoder = HashingEncoder()
            with self.assertRaises(ValueError):
                ContextMemory(path, family=Family(encoder.identity, encoder.dim, seed=3))

    def test_refresh_avoids_reembedding(self):
        class Counter(HashingEncoder):
            calls = 0

            def encode(self, text):
                self.calls += 1
                return super().encode(text)

        encoder = Counter()
        with ContextMemory(encoder=encoder) as memory:
            memory.add("repeat", source="a")
            self.assertEqual(memory.add("repeat", source="a")["status"], "unchanged")
            memory.invalidate("a")
            self.assertEqual(memory.add("repeat", source="a")["status"], "refreshed")
            self.assertEqual(encoder.calls, 1)

    def test_explicit_empty_revisions_are_not_unchecked(self):
        with ContextMemory() as memory:
            memory.add("hello", source="a")
            packet = memory.pack("hello", revisions={})
            self.assertEqual(packet.text, "")
            self.assertEqual(packet.decisions[0]["reason"], "unknown_version")


if __name__ == "__main__":
    unittest.main()

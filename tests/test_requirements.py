import itertools
import json
import random
import unittest

from context_stamps import Claim, ContextMemory, Requirement, rank_candidates_safe, select_structured


class RequirementTests(unittest.TestCase):
    def setUp(self):
        self.memory = ContextMemory()
        self.addCleanup(self.memory.close)
        self.revisions = {}
        self.claims = []

    def add(self, source, text, subject, attribute, value):
        row = self.memory.add(text, source=source)
        self.revisions[source] = row["digest"]
        self.claims.append(Claim(source, row["digest"], subject, attribute, value))

    def select(self, requirements, budget=2048, **overrides):
        kwargs = {"claims": self.claims, "revisions": self.revisions, "budget_bytes": budget}
        return select_structured(self.memory, "settings", requirements=requirements, **(kwargs | overrides))

    def test_wrong_entity_and_required_coverage(self):
        self.add("wrong", "wrong device: timeout 4", "other_device", "timeout", "4")
        self.add("right", "device: timeout 5", "device", "timeout", "5")
        result = self.select([Requirement("device", "timeout")])
        self.assertEqual(result.selected, ["right"])
        self.assertNotIn("wrong device", result.text)

    def test_missing_stale_claim_and_version_map(self):
        self.add("a", "v1", "device", "timeout", "4")
        request = [Requirement("device", "timeout")]
        self.assertEqual(self.select(request).status, "current")
        self.assertEqual(self.select(request, revisions={}).status, "insufficient_evidence")
        row = self.memory.add("v2", source="a")
        self.revisions["a"] = row["digest"]
        self.assertEqual(self.select(request).status, "insufficient_evidence")
        with self.assertRaises(ValueError):
            self.select(request, revisions=None)

    def test_conflict_requires_resolution(self):
        self.add("a", "value 4", "device", "timeout", "4")
        self.add("b", "value 5", "device", "timeout", "5")
        result = self.select([Requirement("device", "timeout")])
        self.assertEqual(result.status, "insufficient_evidence")
        self.assertEqual(result.conflicts[0]["values"], ["4", "5"])
        self.assertFalse(result.text)

    def test_byte_budget_is_exact_including_unicode_and_headers(self):
        self.add("µ", "Δειγμα", "device", "timeout", "4")
        required = [Requirement("device", "timeout")]
        result = self.select(required)
        self.assertEqual(result.units, len(result.text.encode("utf-8")))
        self.assertEqual(self.select(required, result.units).status, "current")
        self.assertEqual(self.select(required, result.units - 1).status, "insufficient_evidence")

    def test_typed_and_bounded_inputs(self):
        with self.assertRaises(ValueError):
            Requirement("", "x")
        with self.assertRaises(ValueError):
            Claim("a", "not-a-digest", "x", "y", "z")
        with self.assertRaises(ValueError):
            self.select([Requirement("x", "y")] * 2)
        with self.assertRaises(ValueError):
            self.select([Requirement("x", str(i)) for i in range(13)])

    def test_optimizer_matches_exhaustive_subsets(self):
        rng = random.Random(743)
        requirements = [Requirement("device", str(i)) for i in range(4)]
        masks = {}
        for i in range(8):
            source = str(i)
            mask = rng.randrange(1, 16)
            masks[source] = mask
            row = self.memory.add("x" * rng.randrange(1, 80), source=source)
            self.revisions[source] = row["digest"]
            self.claims.extend(
                Claim(source, row["digest"], "device", str(j), "1") for j in range(4) if mask & (1 << j)
            )
        best = 100000
        for n in range(1, 9):
            for subset in itertools.combinations(masks, n):
                mask = 0
                for source in subset:
                    mask |= masks[source]
                if mask != 15:
                    continue
                chunks = []
                for source in subset:
                    row = self.memory.get(source)
                    chunks.append(
                        json.dumps({"source": source, "sha256": row["digest"]}) + "\n" + row["text"]
                    )
                best = min(best, len("\n\n".join(chunks).encode()))
        result = self.select(requirements)
        self.assertEqual(result.units, best)
        self.assertFalse(self.select(requirements, best - 1).text)

    def test_baseline_preserves_score_order_and_validates(self):
        self.assertEqual(rank_candidates_safe("q", ["a", "b", "c"], [0.4, 0.9, 0.9]), [1, 2, 0])
        with self.assertRaises(ValueError):
            rank_candidates_safe("q", ["a"], [float("nan")])
        with self.assertRaises(ValueError):
            rank_candidates_safe("q", ["a"], [0.5], coverage_weight=0.3)


if __name__ == "__main__":
    unittest.main()

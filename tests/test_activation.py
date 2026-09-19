import random
import unittest

from context_stamps.activation import StampSchema, activate
from context_stamps.guarded_activation import activate_constrained
from context_stamps.spherical import SphericalStamp
from context_stamps.workflow import ContextGraph, ContextNode
from stamps import Family


class ActivationTests(unittest.TestCase):
    def setUp(self):
        self.stamp = SphericalStamp.encode({"content": [1, 2]}, {"content": Family("test", 2, 64)})

    def test_compact_roundtrip_and_corruption(self):
        schema = StampSchema.for_stamp(self.stamp)
        payload = schema.pack(self.stamp)
        self.assertEqual(schema.unpack(payload), self.stamp)
        for value in (payload[:-3], payload + "!", payload.replace("scqc1", "scqc2"), "x" * 1025):
            with self.assertRaises(ValueError):
                schema.unpack(value)

    def test_activation_limits_and_empty(self):
        self.assertEqual(activate(self.stamp, [], threshold=.5), [])
        self.assertEqual(activate(self.stamp, [self.stamp], threshold=1.01), [])
        self.assertEqual(activate(self.stamp, [self.stamp], threshold=1)[0]["index"], 0)
        for kw in ({"threshold": float("nan")}, {"threshold": .5, "limit": -1}):
            with self.assertRaises(ValueError):
                activate(self.stamp, [self.stamp], **kw)

    def test_collision_cannot_override_entity(self):
        hits = activate_constrained(self.stamp, [self.stamp, self.stamp],
                                    metadata=[{"entity": "wrong"}, {"entity": "right"}],
                                    required={"entity": "right"}, threshold=.5, limit=1)
        self.assertEqual([r["index"] for r in hits], [1])

    def test_missing_exact_metadata_abstains(self):
        self.assertEqual(activate_constrained(self.stamp, [self.stamp], metadata=[{}],
                                            required={"entity": "right"}, threshold=.5), [])

    def test_random_graph_closure_matches_independent_fixed_point(self):
        rng = random.Random(85129)
        for _ in range(100):
            graph = ContextGraph()
            ids = [str(i) for i in range(20)]
            for key in ids:
                graph.put(ContextNode(key, key, "1", frozenset({"worker"})))
            pairs = {(rng.choice(ids), rng.choice(ids)) for _ in range(35)}
            pairs = {(a, b) for a, b in pairs if a != b}
            for a, b in pairs:
                graph.link(a, b, "depends_on", provenance="random fixture")
            root = rng.choice(ids)
            expected = {root}
            while True:
                expanded = expected | {b for a, b in pairs if a in expected}
                if expanded == expected:
                    break
                expected = expanded
            packet = graph.handoff([root], role="worker", revisions=dict.fromkeys(ids, "1"), budget_bytes=65536)
            self.assertEqual(set(packet.sources), expected)
            reverse = {root}
            while True:
                expanded = reverse | {a for a, b in pairs if b in reverse}
                if expanded == reverse:
                    break
                reverse = expanded
            self.assertEqual(set(graph.affected([root])), reverse)

    def test_unauthorized_dependency_does_not_leak_identifier(self):
        graph = ContextGraph()
        graph.put(ContextNode("public", "public", "1", frozenset({"worker"})))
        graph.put(ContextNode("secret-name", "secret-text", "1", frozenset({"admin"})))
        graph.link("public", "secret-name", "depends_on", provenance="trusted")
        result = graph.handoff(["public"], role="worker", revisions={"public": "1", "secret-name": "1"})
        self.assertNotIn("secret", repr(result))
        self.assertEqual(result.text, "")


if __name__ == "__main__":
    unittest.main()

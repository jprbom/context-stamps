import json
import unittest

from context_stamps.facet_model import FacetModel
from context_stamps.spherical import SphericalStamp
from context_stamps.workflow import ContextGraph, ContextNode
from stamps import Family


class SphericalTests(unittest.TestCase):
    def stamp(self, vector=(1, 2), seed=1):
        return SphericalStamp.encode({"intent": vector}, {"intent": Family("fixture", 2, 64, seed)})

    def test_scale_invariance_and_antipodes(self):
        a = self.stamp()
        self.assertEqual(a, self.stamp((10, 20)))
        self.assertEqual(a.score(self.stamp((-1, -2))), 0)
        self.assertEqual(a.bits, 64)

    def test_payload(self):
        a = self.stamp()
        self.assertEqual(a, SphericalStamp.from_payload(a.to_payload()))
        for payload in ('{"format":"scqr1","format":"scqr1","views":{}}',
                        json.dumps({"format": "scqr1", "views": {}}), "x" * 8193):
            with self.assertRaises(ValueError):
                SphericalStamp.from_payload(payload)

    def test_schema_and_weights(self):
        a = self.stamp()
        with self.assertRaises(ValueError):
            a.compare(self.stamp(seed=2))
        for weights in ({}, {"intent": -1}, {"intent": float("nan")}, {"intent": 0}):
            with self.assertRaises(ValueError):
                a.score(a, weights)

    def test_zero_and_nonfinite_vectors(self):
        for vector in ((0, 0), (1, float("nan")), (1, float("inf"))):
            with self.assertRaises(ValueError):
                self.stamp(vector)

    def test_model_fit_roundtrip(self):
        model = FacetModel.fit(self.stamp(), [[0], [1], [0.2], [0.8]], [0, 1, 0, 1])
        copy = FacetModel.from_json(model.to_json())
        self.assertGreater(copy.score(self.stamp(), self.stamp()), copy.score(self.stamp(), self.stamp((-1, -2))))
        with self.assertRaises(ValueError):
            copy.score(self.stamp(seed=9), self.stamp(seed=9))


class WorkflowTests(unittest.TestCase):
    def setUp(self):
        self.graph = ContextGraph()
        for key in ("a", "b", "c", "d"):
            self.graph.put(ContextNode(key, "Evidence " + key, "1", frozenset({"engineer"})))
        self.graph.link("a", "b", "depends_on", provenance="test")
        self.graph.link("b", "c", "depends_on", provenance="test")
        self.revisions = dict.fromkeys(("a", "b", "c", "d"), "1")

    def handoff(self, **kw):
        return self.graph.handoff(["a"], **({"role": "engineer", "revisions": self.revisions} | kw))

    def test_closure_and_reverse_invalidation(self):
        self.assertEqual(self.handoff().sources, ("a", "b", "c"))
        self.assertEqual(self.graph.affected(["c"]), ("a", "b", "c"))
        self.assertEqual(self.graph.affected(["b"]), ("a", "b"))

    def test_cycle_terminates(self):
        self.graph.link("c", "a", "depends_on", provenance="test")
        self.assertEqual(self.handoff().status, "complete")

    def test_changed_content_same_revision_rejected(self):
        self.graph.put(ContextNode("c", "changed", "1", frozenset({"engineer"})))
        self.assertEqual(self.handoff().status, "insufficient")

    def test_stale_and_permission_fail_closed(self):
        self.assertEqual(self.handoff(role="outsider").text, "")
        self.revisions["c"] = "2"
        self.assertEqual(self.handoff().sources, ())

    def test_exact_budget_and_conflict(self):
        packet = self.handoff()
        self.assertEqual(self.handoff(budget_bytes=packet.units).status, "complete")
        self.assertEqual(self.handoff(budget_bytes=packet.units - 1).text, "")
        self.graph.link("a", "c", "contradicts", provenance="review")
        self.assertEqual(self.handoff().reason, "conflicting_context")

    def test_revision_binding_even_if_text_unchanged(self):
        self.graph.put(ContextNode("c", "Evidence c", "2", frozenset({"engineer"})))
        self.revisions["c"] = "2"
        self.assertEqual(self.handoff().status, "insufficient")


if __name__ == "__main__":
    unittest.main()

import dataclasses
import unittest

from context_stamps.context_state import AccessScope, ContextState
from context_stamps.trajectory import ObservedEpisode, ObservedStep, trajectory_views


class TrajectoryTests(unittest.TestCase):
    def episode(self):
        return ObservedEpisode("case-1", "Update a setting", (
            ObservedStep(0, "heading\nstatus: pending\nfooter", "", "local/page"),
            ObservedStep(2, "heading\nstatus: accepted\nfooter", "click(confirm)", "local/page")))

    def test_old_and_new_direction_and_no_truth_promotion(self):
        views = list(trajectory_views(self.episode()))
        change = next(v for v in views if v.channel == "change")
        self.assertEqual(change.steps, (0, 2))
        self.assertLess(change.text.index("pending"), change.text.index("accepted"))
        self.assertIn("not a causal assertion", change.text)
        for view in views:
            node = view.canonical_node(tenant="a", roles=("reader",), observed_at=7)
            self.assertEqual(node.claims, ())
            self.assertIsNone(node.validation_revision)
            self.assertEqual(node.kind, "OBSERVATION" if view.channel == "state" else "DERIVED_RESULT")
            self.assertEqual(node.temporal.event_time, 7)  # view ingestion, not source chronology

    def test_full_episode_mutation_changes_all_source_bindings(self):
        original = self.episode()
        modified = dataclasses.replace(original, steps=(original.steps[0], dataclasses.replace(original.steps[1], action="cancel")))
        self.assertNotEqual(original.revision, modified.revision)
        for view in trajectory_views(modified):
            self.assertEqual(view.source_revision, modified.revision)
        a = next(iter(trajectory_views(original)))
        b = next(iter(trajectory_views(modified)))
        self.assertEqual(a.text, b.text)
        self.assertNotEqual(a.canonical_node(tenant="a", roles=("r",), observed_at=1).ref,
                            b.canonical_node(tenant="a", roles=("r",), observed_at=1).ref)

    def test_canonical_acl_revocation_invalidates_receipt(self):
        state = ContextState(tenant="a", policy_revision="v1", clock=lambda: 7)
        node = next(iter(trajectory_views(self.episode()))).canonical_node(tenant="a", roles=("reader",), observed_at=7)
        state.put(node)
        snap = state.snapshot(AccessScope("a", "u", "v1", ("reader",)), at=7, known_at=7)
        seal = state.seal(snap, node.text)
        self.assertTrue(state.verify_binding(snap, node.text, seal))
        state.set_roles(node.key, ("admin",))
        self.assertFalse(state.verify_binding(snap, node.text, seal))

    def test_bounds_order_and_unicode(self):
        with self.assertRaises(ValueError):
            ObservedEpisode("x", "", (ObservedStep(2, "a"), ObservedStep(1, "b")))
        with self.assertRaises(ValueError):
            ObservedStep(True, "a")
        with self.assertRaises(ValueError):
            ObservedStep(1, "x" * (2*1024*1024+1))
        ep = ObservedEpisode("unicode", "", (ObservedStep(0, "नमस्ते 🌍\n"*2000),))
        views = list(trajectory_views(ep, chunk_chars=256))
        self.assertGreater(len(views), 10)
        self.assertTrue(all(v.text.encode().decode() == v.text for v in views))

    def test_unchanged_observation_has_no_invented_change(self):
        ep = ObservedEpisode("x", "", (ObservedStep(0, "same"), ObservedStep(1, "same", "click(x)")))
        self.assertNotIn("change", [v.channel for v in trajectory_views(ep)])


if __name__ == "__main__":
    unittest.main()

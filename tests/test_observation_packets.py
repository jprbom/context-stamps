import dataclasses
import unittest

from context_stamps.context_state import AccessScope, ContextState
from context_stamps.observation_packets import (
    ObservationPacket,
    RelationOccurrence,
    label,
    observation_packets,
    page_hint,
    plan_pages,
)
from context_stamps.trajectory import ObservedEpisode, ObservedStep


class ObservationPacketTests(unittest.TestCase):
    def episode(self, text):
        return ObservedEpisode("local", "Inspect a setting", (ObservedStep(0, text),))

    def test_quoted_attributes_are_not_executed_or_reinterpreted(self):
        self.assertEqual(label("button 'required=True, $(bad)', visible"),
                         ("button", "'required=True, $(bad)'", ", visible"))
        self.assertEqual(label("button 'a\\\'b', visible")[1], "'a\\\'b'")
        self.assertEqual(label("not a supported name")[0], "opaque")

    def test_page_paths_keep_nested_frames(self):
        text = "RootWebArea 'Shell'\n  Iframe 'Work'\n    RootWebArea 'Settings'\n      button 'Save'"
        v = next(observation_packets(self.episode(text)))
        self.assertEqual(v.page, ("'Shell'", "'Settings'"))
        self.assertEqual(v.occurrences[0].lines, (1, 3, 4))
        self.assertEqual(page_hint(text), v.page)

    def test_table_member_order_and_no_visual_position_claim(self):
        text = "RootWebArea 'Inventory'\n  table ''\n    row ''\n      columnheader 'Item'\n      columnheader 'Quantity'\n      columnheader 'Location'"
        v = next(observation_packets(self.episode(text)))
        self.assertEqual(v.relation, "ordered_members")
        self.assertLess(v.body.index("Quantity"), v.body.index("Location"))
        self.assertIn("source order", v.body)
        self.assertNotIn("right", v.body)
        self.assertEqual(v.occurrences[0].lines, (1, 2, 3, 4, 5, 6))

    def test_identical_headers_keep_distinct_container_ownership(self):
        text = "RootWebArea 'A'\n  table ''\n    row ''\n      columnheader 'Name'\n  table ''\n    row ''\n      columnheader 'Name'"
        views = list(observation_packets(self.episode(text)))
        self.assertEqual(len(views), 2)
        self.assertIn("ancestor at source line 2", views[0].body)
        self.assertIn("ancestor at source line 5", views[1].body)

    def test_controls_keep_names_values_and_options(self):
        text = "RootWebArea 'Settings'\n  combobox 'Mode' value='Strict'\n    option 'Strict', selected=True\n    option 'Relaxed', selected=False"
        views = list(observation_packets(self.episode(text)))
        self.assertEqual(len(views), 1)
        self.assertIn("combobox 'Mode' value='Strict'", views[0].body)
        self.assertIn("option 'Relaxed', selected=False", views[0].body)

    def test_exact_occurrences_dedup_and_value_mutation(self):
        ep = self.episode("RootWebArea 'Settings'\n  textbox 'Name' value='A'")
        ep = dataclasses.replace(ep, steps=ep.steps+(dataclasses.replace(ep.steps[0], index=1),))
        views = list(observation_packets(ep))
        self.assertEqual(len(views), 1)
        self.assertEqual(len(views[0].occurrences), 2)
        changed = dataclasses.replace(ep, goal="Different source")
        self.assertNotEqual(views[0].key, next(observation_packets(changed)).key)

    def test_source_binding_and_authorization_revocation(self):
        v = next(observation_packets(self.episode("RootWebArea 'A'\n  button 'Save'")))
        node = v.canonical_node(tenant="a", roles=("reader",), observed_at=1)
        self.assertEqual(node.kind, "DERIVED_RESULT")
        self.assertFalse(node.claims)
        state = ContextState(tenant="a", policy_revision="v1", clock=lambda: 1)
        state.put(node)
        snap = state.snapshot(AccessScope("a", "user", "v1", ("reader",)), at=1, known_at=1)
        seal = state.seal(snap, v.text)
        state.set_roles(node.key, ("admin",))
        self.assertFalse(state.verify_binding(snap, v.text, seal))

    def test_bounds_fail_closed(self):
        with self.assertRaises(ValueError):
            RelationOccurrence(0, (2, 1))
        with self.assertRaises(ValueError):
            ObservationPacket("x", "a"*64, (), "", "invented_cause", "x", (RelationOccurrence(0, (1,)),))
        with self.assertRaises(ValueError):
            list(observation_packets(self.episode("x"*8193)))
        with self.assertRaises(ValueError):
            plan_pages("query", (("Title",),), relative_floor=float("nan"))

    def test_specific_anchor_excludes_common_site_title(self):
        pages = tuple((f"'Topic {i} / Admin'",) for i in range(20)) + (("'Inventory balance / Admin'",),)
        result = plan_pages("In our Admin, open Inventory balance", pages)
        self.assertEqual(result.pages, (("'Inventory balance / Admin'",),))
        self.assertNotIn("admin", result.anchor_terms)
        self.assertFalse(result.fallback)
        self.assertTrue(plan_pages("Please help with Admin", pages).fallback)

    def test_nested_rows_are_not_flattened_into_parent_row(self):
        text = "RootWebArea 'A'\n  row 'Outer'\n    gridcell 'One'\n    row 'Inner'\n      gridcell 'Two'"
        views = list(observation_packets(self.episode(text)))
        ordered = [v for v in views if v.relation == "ordered_members"]
        self.assertEqual(len(ordered), 2)
        self.assertTrue(all(not ("One" in v.body and "Two" in v.body) for v in ordered))


if __name__ == "__main__":
    unittest.main()

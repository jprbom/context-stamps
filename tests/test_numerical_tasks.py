"""Task binding and numerical boundaries; no perceptual model benefit implied."""

import json
import secrets
import tempfile
import unittest
from dataclasses import asdict, replace
from pathlib import Path
from unittest.mock import patch

from context_stamps.context_state import AccessScope, CanonicalNode, ContextState, TemporalScope
from context_stamps.numerical_tasks import (
    NumericalCell,
    NumericalTable,
    NumericalTask,
    Selector,
    execute_numerical,
    numerical_view,
    prepare_numerical,
)
from context_stamps.state_store import ContextStore


def cell(key, label, value, *, unit="tasks", series=None):
    coordinates = (("entity", label),) + ((("series", series),) if series else ())
    return NumericalCell(key, coordinates, value, unit)


def selector(label, series=None):
    return Selector((("entity", label),) + ((("series", series),) if series else ()))


class NumericalTests(unittest.TestCase):
    def setUp(self):
        self.state = ContextState(tenant="local", policy_revision="p1", clock=lambda: 10)
        self.scope = AccessScope("local", "owner", "p1", ("reader",))
        self.source = CanonicalNode(key="observed-table", revision="v1", tenant="local", text="Unverified source",
            kind="MODEL_OUTPUT", temporal=TemporalScope(1, 1), roles=("reader",), provenance="test-fixture")
        self.state.put(self.source)

    def prepare(self, rows, task, state=None):
        state = state or self.state
        table = NumericalTable(tuple(rows))
        node = numerical_view(self.source, table, key="numeric-view")
        state.put(node)
        snapshot = state.snapshot(self.scope, at=10, known_at=10)
        return prepare_numerical(state, snapshot, node.ref, task)

    def test_exact_rational_sum_and_epistemic_status(self):
        packet = self.prepare([cell("a", "A", "0.1"), cell("b", "B", "0.2")],
            NumericalTask("sum", (selector("A"), selector("B"))))
        result = execute_numerical(self.state, packet)
        self.assertEqual((result.answer, result.rational, result.unit), ("0.3", (3, 10), "tasks"))
        self.assertTrue(result.task_execution_verified)
        self.assertEqual(result.source_kind, "MODEL_OUTPUT")
        self.assertFalse(result.source_semantics_verified)
        self.assertFalse(result.question_interpretation_verified)

    def test_absent_subject_cannot_be_replaced_by_other_cells(self):
        packet = self.prepare([cell("c0", "Ravi", "11"), cell("c1", "Uma", "44"), cell("c2", "Tao", "22")],
            NumericalTask("lookup", (selector("Zeta"),)))
        result = execute_numerical(self.state, packet)
        self.assertEqual(result.status, "selector_not_in_view")
        self.assertIsNone(result.answer)
        self.assertFalse(result.task_execution_verified)

    def test_no_substring_case_or_unicode_aliases(self):
        for label in ("Al", "alpha", "Αlpha"):
            state = ContextState(tenant="local", policy_revision="p1", clock=lambda: 10)
            state.put(self.source)
            packet = self.prepare([cell("a", "Alpha", "3")], NumericalTask("lookup", (selector(label),)), state)
            self.assertEqual(execute_numerical(state, packet).status, "selector_not_in_view")

    def test_duplicate_labels_require_declared_series(self):
        rows = [cell("a", "A", "3", series="before"), cell("b", "A", "7", series="after")]
        packet = self.prepare(rows, NumericalTask("lookup", (selector("A"),)))
        self.assertEqual(execute_numerical(self.state, packet).status, "ambiguous_selector")
        snapshot = self.state.snapshot(self.scope, at=10, known_at=10)
        packet = prepare_numerical(self.state, snapshot, packet.snapshot.records[0].node.ref,
            NumericalTask("lookup", (selector("A", "after"),)))
        self.assertEqual(execute_numerical(self.state, packet).answer, "7")

    def test_aliasing_two_selectors_cannot_double_count_one_cell(self):
        packet = self.prepare([cell("a", "A", "3", series="before")],
            NumericalTask("sum", (selector("A"), selector("A", "before"))))
        self.assertEqual(execute_numerical(self.state, packet).status, "duplicate_selected_cell")

    def test_unreadable_is_not_zero(self):
        for value, status, answer in ((None, "unreadable_value", None), ("0", "computed", "0")):
            state = ContextState(tenant="local", policy_revision="p1", clock=lambda: 10)
            state.put(self.source)
            packet = self.prepare([cell("a", "A", value)], NumericalTask("lookup", (selector("A"),)), state)
            result = execute_numerical(state, packet)
            self.assertEqual((result.status, result.answer), (status, answer))

    def test_units_scale_and_zero_division(self):
        for rows, op, status, answer in (
            ([cell("a", "A", "1", unit="USD"), cell("b", "B", "2", unit="USD million")], "sum", "incompatible_units", None),
            ([cell("a", "A", "25", unit="%"), cell("b", "B", "50", unit="%")], "ratio", "computed", "0.5"),
            ([cell("a", "A", "25"), cell("b", "B", "100")], "percent_ratio", "computed", "25"),
            ([cell("a", "A", "1"), cell("b", "B", "0")], "ratio", "undefined_division", None),
        ):
            state = ContextState(tenant="local", policy_revision="p1", clock=lambda: 10)
            state.put(self.source)
            packet = self.prepare(rows, NumericalTask(op, (selector("A"), selector("B"))), state)
            result = execute_numerical(state, packet)
            self.assertEqual((result.status, result.answer), (status, answer))

    def test_declared_population_and_ties(self):
        with self.assertRaises(ValueError):
            NumericalTask("argmax", (), "entity")
        packet = self.prepare([cell("a", "A", "5"), cell("b", "B", "5"), cell("c", "C", "99")],
            NumericalTask("argmax", (selector("A"), selector("B")), "entity"))
        self.assertEqual(execute_numerical(self.state, packet).status, "ambiguous_extremum")

    def test_operation_results_and_explicit_population(self):
        expectations = {"sum": "9", "mean": "4.5", "min": "3", "max": "6", "subtract": "-3",
            "absdiff": "3", "ratio": "0.5", "percent_ratio": "50", "greater": "No", "less": "Yes",
            "equal": "No", "count": "2", "argmin": "A", "argmax": "B"}
        for operation, expected in expectations.items():
            with self.subTest(operation=operation):
                state = ContextState(tenant="local", policy_revision="p1", clock=lambda: 10)
                state.put(self.source)
                task = NumericalTask(operation, (selector("A"), selector("B")),
                    "entity" if operation in ("argmin", "argmax") else None)
                packet = self.prepare([cell("a", "A", "3"), cell("b", "B", "6"), cell("unrelated", "C", "999")], task, state)
                result = execute_numerical(state, packet)
                self.assertEqual(result.answer, expected)
                self.assertEqual(result.selected_cells, ("a", "b"))

    def test_scope_changes_and_expiry_prevent_preparation(self):
        packet = self.prepare([cell("a", "A", "3")], NumericalTask("lookup", (selector("A"),)))
        altered = replace(packet, snapshot=replace(packet.snapshot, scope=replace(self.scope, principal="other")))
        self.assertEqual(execute_numerical(self.state, altered).status, "unavailable_context")
        state = ContextState(tenant="local", policy_revision="p1", clock=lambda: 10)
        expired = replace(self.source, temporal=TemporalScope(1, 1, valid_until=5))
        state.put(expired)
        view = numerical_view(expired, NumericalTable((cell("a", "A", "3"),)), key="numeric-view")
        state.put(view)
        with self.assertRaises(ValueError):
            prepare_numerical(state, state.snapshot(self.scope, at=10, known_at=10), view.ref, packet.task)

    def test_order_matters_and_reordered_source_does_not(self):
        task = NumericalTask("subtract", (selector("B"), selector("A")))
        packet = self.prepare([cell("b", "B", "10"), cell("a", "A", "-2")], task)
        result = execute_numerical(self.state, packet)
        self.assertEqual((result.answer, result.selected_cells), ("12", ("b", "a")))
        for changed in (replace(task, operation="absdiff"), NumericalTask("subtract", tuple(reversed(task.selectors)))):
            self.assertEqual(execute_numerical(self.state, replace(packet, task=changed)).status, "unavailable_context")

    def test_source_revocation_and_mid_execution_changes_invalidate(self):
        packet = self.prepare([cell("a", "A", "3")], NumericalTask("lookup", (selector("A"),)))
        self.state.set_roles(self.source.key, ("private",))
        self.assertEqual(execute_numerical(self.state, packet).status, "unavailable_context")
        self.state.set_roles(self.source.key, ("reader",))
        snapshot = self.state.snapshot(self.scope, at=10, known_at=10)
        packet = prepare_numerical(self.state, snapshot, packet.snapshot.records[0].node.ref, packet.task)
        original = NumericalTable.from_payload
        def revoke(text):
            result = original(text)
            self.state.set_roles(self.source.key, ("private",))
            return result
        with patch.object(NumericalTable, "from_payload", side_effect=revoke):
            result = execute_numerical(self.state, packet)
        self.assertEqual(result.status, "unavailable_context")
        self.assertIsNone(result.answer)

    def test_durable_source_revocation_survives_restart(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"state.sqlite"
            args = dict(tenant="local", signing_key=secrets.token_bytes(32), clock=lambda: 10,
                authorize=lambda scope, operation, subject: scope == self.scope)
            view = numerical_view(self.source, NumericalTable((cell("a", "A", "3"),)), key="numeric-view")
            with ContextStore(path, create=True, policy_revision="p1", **args) as store:
                store.put(self.source, scope=self.scope, mutation_id="source")
                store.put(view, scope=self.scope, mutation_id="view")
                packet = prepare_numerical(store, store.snapshot(self.scope, at=10, known_at=10), view.ref,
                    NumericalTask("lookup", (selector("A"),)))
                self.assertEqual(execute_numerical(store, packet).answer, "3")
                checkpoint = store.checkpoint(scope=self.scope, verify=True)
            with ContextStore(path, checkpoint=checkpoint, **args) as store:
                self.assertEqual(execute_numerical(store, packet).answer, "3")
                with ContextStore(path, checkpoint=checkpoint, **args) as writer:
                    writer.set_roles(self.source.key, (), scope=self.scope, mutation_id="revoke")
                self.assertEqual(execute_numerical(store, packet).status, "unavailable_context")
                self.assertFalse(store.snapshot(self.scope, at=10, known_at=10).records)

    def test_bounded_numeric_and_json_inputs(self):
        for value in (True, 3.0, "NaN", "1e99", "25%", "1,000", "1/3", "9"*100):
            with self.assertRaises(ValueError):
                cell("a", "A", value)
        for text in ('{"schema":"x","schema":"y","cells":[]}', '['*1500+']'*1500,
                     '{"schema":"numerical-task-v1","cells":[1e999]}'):
            with self.assertRaises(ValueError):
                NumericalTable.from_payload(text)
        table = NumericalTable((cell("a", "A", "1"),))
        self.assertEqual(NumericalTable.from_payload(table.payload), table)
        data = json.loads(table.payload)
        data["cells"][0]["coordinates"] = [["entity", "A"], ["entity", "B"]]
        with self.assertRaises(ValueError):
            NumericalTable.from_payload(json.dumps(data))
        self.assertEqual(asdict(execute_numerical(self.state,
            self.prepare(table.cells, NumericalTask("lookup", (selector("A"),)))))["rational"], (1, 1))


if __name__ == "__main__":
    unittest.main()

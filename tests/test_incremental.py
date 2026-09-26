import json
import tempfile
import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from context_stamps.context_state import AccessScope, CanonicalNode, ContextClaim, ContextState, TemporalScope
from context_stamps.experience import ResourceUse
from context_stamps.incremental import (
    ComputeAdapter,
    ComputeOutput,
    ComputeSpec,
    DAGBudget,
    IncrementalExecutor,
    active_order,
    affected_nodes,
)
from context_stamps.state_store import ContextStore


def node(key, number, revision="v1", **kwargs):
    return CanonicalNode(key, revision, "lab", str(number), "OBSERVATION", TemporalScope(0, 10),
                         ("reader",), "fixture", claims=(ContextClaim("number", str(number)),), **kwargs)


def expected(inputs):
    return str(sum(int(c.value) for r in inputs.records for c in r.node.claims if c.name == "number")
               + sum(int(p.text) for p in inputs.parents) + json.loads(inputs.spec.request).get("offset", 0))


class IncrementalTests(unittest.TestCase):
    def setUp(self):
        self.scope = AccessScope("lab", "scientist", "p1", ("reader",))
        self.state = ContextState(tenant="lab", policy_revision="p1", clock=lambda: 100)
        self.a, self.b, self.x = node("raw-a", 1), node("raw-b", 2), node("raw-x", 10)
        for source in (self.a, self.b, self.x):
            self.state.put(source)
        self.calls, self.inputs = [], []
        self.adapter = ComputeAdapter("sum-v1", "verify-v1", self.compute, lambda i, text: text == expected(i))
        self.engine = IncrementalExecutor(self.state, (self.adapter,), authorize=lambda scope, spec: True,
                                          signing_key=b"k" * 32, clock=lambda: 200)
        self.graph = (
            ComputeSpec("a", "sum-v1", "verify-v1", (self.a.ref,)),
            ComputeSpec("b", "sum-v1", "verify-v1", (self.b.ref,), ("a",)),
            ComputeSpec("c", "sum-v1", "verify-v1", parents=("b",), request='{"offset":1}'),
            ComputeSpec("d", "sum-v1", "verify-v1", parents=("b",), request='{"offset":2}'),
            ComputeSpec("e", "sum-v1", "verify-v1", parents=("c",), request='{"offset":3}'),
            ComputeSpec("x", "sum-v1", "verify-v1", (self.x.ref,)),
            ComputeSpec("y", "sum-v1", "verify-v1", parents=("x",), request='{"offset":5}'),
        )
        self.targets = ("d", "e", "y")

    def compute(self, inputs):
        self.calls.append(inputs.spec.key)
        self.inputs.append(inputs)
        return ComputeOutput(expected(inputs), ResourceUse(0, 0, 0, 0, 0))

    def run_graph(self, graph=None, targets=None, engine=None, **kwargs):
        values = dict(scope=self.scope, at=200, known_at=200)
        values.update(kwargs)
        return (engine or self.engine).run(graph or self.graph, targets or self.targets, **values)

    def test_branched_delta_reruns_only_affected_active_nodes(self):
        first = self.run_graph()
        self.assertEqual(first.status, "complete")
        self.assertEqual({r.key: r.text for r in first.results}, {"d": "5", "e": "7", "y": "15"})
        self.assertEqual(len(self.calls), 7)
        warm = self.run_graph()
        self.assertTrue(all(s.status == "reused" for s in warm.steps))
        changed = replace(self.b, revision="v2", text="20", claims=(ContextClaim("number", "20"),),
                          supersedes=(self.b.ref,))
        self.state.put(changed)
        graph = tuple(replace(s, sources=(changed.ref,)) if s.key == "b" else s for s in self.graph)
        delta = self.run_graph(graph)
        self.assertEqual({s.key for s in delta.steps if s.status == "computed"}, {"b", "c", "d", "e"})
        self.assertEqual({r.key: r.text for r in delta.results}, {"d": "23", "e": "25", "y": "15"})
        self.assertEqual(set(affected_nodes(graph, ("b",), self.targets)), {"b", "c", "d", "e"})
        self.assertEqual(affected_nodes(graph, ("b",), ("y",)), ())

    def test_unrelated_state_mutation_and_time_advance_preserve_exact_reuse(self):
        self.run_graph()
        self.state.put(node("unrelated", 999))
        result = self.run_graph(at=201, known_at=201)
        self.assertTrue(all(s.status == "reused" for s in result.steps))
        self.assertEqual(len(self.calls), 7)

    def test_every_declared_version_and_request_invalidates_descendants(self):
        self.run_graph()
        for field, value in (("model_revision", "m2"), ("prompt_revision", "p2"),
                             ("tokenizer_revision", "t2"), ("tool_revision", "tool2"),
                             ("request", '{"seed":2}')):
            graph = tuple(replace(s, **{field: value}) if s.key == "b" else s for s in self.graph)
            result = self.run_graph(graph)
            with self.subTest(field=field):
                self.assertEqual({s.key for s in result.steps if s.status == "computed"}, {"b", "c", "d", "e"})

    def test_changed_operation_and_verifier_versions_are_bound(self):
        adapter = replace(self.adapter, operation_revision="sum-v2", verifier_revision="verify-v2")
        engine = IncrementalExecutor(self.state, (self.adapter, adapter), authorize=lambda s, c: True)
        self.run_graph(engine=engine)
        graph = tuple(replace(s, operation_revision="sum-v2", verifier_revision="verify-v2") if s.key == "b" else s
                      for s in self.graph)
        result = self.run_graph(graph, engine=engine)
        self.assertEqual({s.key for s in result.steps if s.status == "computed"}, {"b", "c", "d", "e"})
        with self.assertRaises(ValueError):
            self.run_graph(tuple(replace(s, verifier_revision="wrong") for s in self.graph))

    def test_same_value_new_source_version_still_invalidates(self):
        self.run_graph()
        changed = replace(self.b, revision="v2", supersedes=(self.b.ref,))
        self.state.put(changed)
        graph = tuple(replace(s, sources=(changed.ref,)) if s.key == "b" else s for s in self.graph)
        result = self.run_graph(graph)
        self.assertEqual({s.key for s in result.steps if s.status == "computed"}, {"b", "c", "d", "e"})

    def test_policy_principal_and_roles_isolate_reuse(self):
        self.run_graph()
        for scope in (replace(self.scope, principal="another"), replace(self.scope, roles=("reader", "auditor"))):
            result = self.run_graph(scope=scope)
            self.assertTrue(all(s.status == "computed" for s in result.steps))
        self.state.set_policy("p2")
        self.assertEqual(self.run_graph().status, "unauthorized")
        result = self.run_graph(scope=replace(self.scope, policy_revision="p2"))
        self.assertTrue(all(s.status == "computed" for s in result.steps))

    def test_only_dependency_closed_direct_records_reach_callback(self):
        policy = CanonicalNode("policy", "v1", "lab", "Required policy", "POLICY", TemporalScope(0, 10),
                               ("reader",), "fixture", validation_revision="v1", stamp_hex="1" * 64,
                               stamp_schema="eight-facet-v1")
        self.state.put(policy)
        bound = replace(self.b, revision="v2", dependencies=(policy.ref,), supersedes=(self.b.ref,))
        self.state.put(bound)
        graph = tuple(replace(s, sources=(bound.ref,)) if s.key == "b" else s for s in self.graph)
        result = self.run_graph(graph)
        b_input = next(i for i in self.inputs if i.spec.key == "b")
        self.assertEqual({r.node.key for r in b_input.records}, {"raw-b", "policy"})
        self.assertEqual({p.binding.key for p in b_input.parents}, {"a"})
        e = next(r for r in result.results if r.key == "e")
        self.assertEqual({b.ref.key for b in e.receipt.sources}, {"raw-a", "raw-b", "policy"})
        self.assertEqual(next(b for b in e.receipt.sources if b.ref.key == "policy").stamp_hex, "1" * 64)
        self.state.set_roles("policy", ())
        self.assertEqual(self.run_graph(graph).status, "unavailable_context")

    def test_revocation_never_releases_cached_payload_or_receipts(self):
        self.run_graph()
        self.state.set_roles("raw-b", ())
        denied = self.run_graph()
        self.assertEqual(denied.status, "unavailable_context")
        self.assertFalse(denied.results)
        self.assertTrue(all(s.receipt is None for s in denied.steps))

    def test_host_authorization_is_checked_on_hits_and_final_release(self):
        allowed = {s.key for s in self.graph}
        engine = IncrementalExecutor(self.state, (self.adapter,), authorize=lambda scope, spec: spec.key in allowed)
        self.run_graph(engine=engine)
        allowed.remove("b")
        self.assertEqual(self.run_graph(engine=engine).status, "unauthorized")
        allowed.add("b")

        def compute(inputs):
            if inputs.spec.key == "e":
                allowed.remove("a")
            return self.compute(inputs)

        engine = IncrementalExecutor(self.state, (replace(self.adapter, compute=compute),),
                                      authorize=lambda scope, spec: spec.key in allowed)
        result = self.run_graph(engine=engine)
        self.assertEqual(result.status, "unauthorized")
        self.assertFalse(result.results)
        self.assertTrue(all(s.receipt is None for s in result.steps))
        self.assertEqual(engine.cache_info()["entries"], 0)

    def test_temporal_ineligibility_cannot_use_cache(self):
        future = replace(self.b, revision="v2", temporal=TemporalScope(250, 10, valid_until=300))
        self.state.put(future)
        graph = tuple(replace(s, sources=(future.ref,)) if s.key == "b" else s for s in self.graph)
        self.assertEqual(self.run_graph(graph, at=260).status, "complete")
        for at in (200, 300):
            self.assertEqual(self.run_graph(graph, at=at).status, "unavailable_context")
        self.assertEqual(self.run_graph(graph, at=260, known_at=99).status, "unavailable_context")

    def test_graph_validation_and_target_pruning(self):
        result = self.run_graph(targets=("y",))
        self.assertEqual([s.key for s in result.steps], ["x", "y"])
        for graph in (self.graph + (self.graph[0],),
                      (replace(self.graph[0], parents=("missing",)),),
                      (replace(self.graph[0], parents=("b",)), self.graph[1])):
            with self.assertRaises(ValueError):
                active_order(graph, ("a",))
        with self.assertRaises(ValueError):
            affected_nodes(self.graph, ("missing",), self.targets)

    def test_failure_and_verifier_rejection_do_not_cache_or_release(self):
        def failure(_):
            raise RuntimeError("private source content")

        for compute, verify, status in ((failure, lambda i, t: True, "compute_failed"),
                                        (lambda i: "untyped", lambda i, t: True, "invalid_output"),
                                        (self.compute, lambda i, t: False, "verification_failed"),
                                        (self.compute, lambda i, t: 1, "verification_failed")):
            adapter = replace(self.adapter, compute=compute, verify=verify)
            engine = IncrementalExecutor(self.state, (adapter,), authorize=lambda s, c: True)
            result = self.run_graph(engine=engine)
            self.assertEqual(result.status, status)
            self.assertFalse(result.results)
            self.assertEqual(engine.cache_info()["entries"], 0)
            self.assertNotIn("private source content", repr(result))

    def test_current_state_change_mid_compute_rejects_whole_run(self):
        def compute(inputs):
            if inputs.spec.key == "b":
                self.state.put(node("new", 99))
            return self.compute(inputs)

        engine = IncrementalExecutor(self.state, (replace(self.adapter, compute=compute),), authorize=lambda s, c: True)
        result = self.run_graph(engine=engine)
        self.assertEqual(result.status, "state_changed")
        self.assertFalse(result.results)
        self.assertTrue(all(s.receipt is None for s in result.steps))
        self.assertEqual(engine.cache_info()["entries"], 0)

    def test_cancellation_and_deadline_reject_late_callbacks(self):
        cancel = threading.Event()
        cancel.set()
        self.assertEqual(self.run_graph(cancel=cancel).status, "cancelled")
        self.assertFalse(self.calls)
        cancel.clear()

        def compute(inputs):
            cancel.set()
            return self.compute(inputs)

        engine = IncrementalExecutor(self.state, (replace(self.adapter, compute=compute),), authorize=lambda s, c: True)
        self.assertEqual(self.run_graph(engine=engine, cancel=cancel).status, "cancelled")

        def verify(inputs, text):
            time.sleep(0.03)
            return True

        engine = IncrementalExecutor(self.state, (replace(self.adapter, verify=verify),), authorize=lambda s, c: True)
        result = self.run_graph(engine=engine, budget=DAGBudget(milliseconds=10))
        self.assertEqual(result.status, "deadline")
        self.assertFalse(result.results)
        self.assertEqual(engine.cache_info()["entries"], 0)

    def test_single_flight_and_concurrent_clear_do_not_deadlock(self):
        entered, release = threading.Event(), threading.Event()

        def compute(inputs):
            entered.set()
            if not release.wait(5):
                raise RuntimeError("test release missing")
            return self.compute(inputs)

        engine = IncrementalExecutor(self.state, (replace(self.adapter, compute=compute),), authorize=lambda s, c: True)
        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(self.run_graph, engine=engine)
            self.assertTrue(entered.wait(5))
            try:
                with self.assertRaises(RuntimeError):
                    self.run_graph(engine=engine)
                engine.clear()
            finally:
                release.set()
            self.assertEqual(future.result(timeout=5).status, "invalidated")
        self.assertEqual(engine.cache_info()["entries"], 0)

    def test_receipt_integrity_is_not_present_authority(self):
        result = self.run_graph().results[0]
        self.assertTrue(self.engine.verify_receipt(result.receipt, result.text))
        for receipt in (replace(result.receipt, result_hash="0" * 64),
                        replace(result.receipt, scope=replace(self.scope, principal="other")),
                        replace(result.receipt, created_ms=201)):
            self.assertFalse(self.engine.verify_receipt(receipt, result.text))
        self.assertFalse(self.engine.verify_receipt(result.receipt, "wrong"))
        self.state.invalidate(self.b.ref)
        self.assertTrue(self.engine.verify_receipt(result.receipt, result.text))
        self.assertEqual(self.run_graph().status, "unavailable_context")

    def test_valid_receipt_in_wrong_cache_slot_is_rejected(self):
        self.run_graph()
        keys = list(self.engine._cache)
        self.engine._cache[keys[0]] = self.engine._cache[keys[1]]
        result = self.run_graph()
        self.assertEqual(result.status, "cache_integrity")
        self.assertFalse(result.results)

    def test_receipts_do_not_include_request_or_evidence_payloads(self):
        graph = tuple(replace(s, request='{"private_note":"do-not-export-this-text"}') if s.key == "b" else s
                      for s in self.graph)
        result = self.run_graph(graph)
        receipt = next(s.receipt for s in result.steps if s.key == "b")
        self.assertNotIn("do-not-export-this-text", repr(receipt))
        self.assertEqual(len(receipt.spec.request_hash), 64)

    def test_cache_expiry_and_quotas(self):
        with patch("context_stamps.incremental.time.perf_counter", return_value=10):
            engine = IncrementalExecutor(self.state, (self.adapter,), authorize=lambda s, c: True, ttl_seconds=1)
            self.run_graph(engine=engine)
        with patch("context_stamps.incremental.time.perf_counter", return_value=11):
            result = self.run_graph(engine=engine)
            self.assertTrue(all(s.status == "computed" for s in result.steps))
        for capacity, max_bytes in ((2, 400000), (256, 1)):
            engine = IncrementalExecutor(self.state, (self.adapter,), authorize=lambda s, c: True,
                                          capacity=capacity, max_bytes=max_bytes)
            self.assertEqual(self.run_graph(engine=engine).status, "complete")
            self.assertLessEqual(engine.cache_info()["entries"], capacity)
            self.assertLessEqual(engine.cache_info()["serialized_bytes"], max_bytes)

    def test_call_and_output_budgets_and_unknown_usage(self):
        result = self.run_graph(budget=DAGBudget(compute_calls=0))
        self.assertEqual(result.status, "compute_budget")
        self.assertEqual(result.steps[0].usage.model_calls, 0)
        self.assertFalse(self.calls)
        result = self.run_graph(budget=DAGBudget(result_bytes=1))
        self.assertEqual(result.status, "result_budget")
        self.assertFalse(result.results)
        self.assertEqual(self.engine.cache_info()["entries"], 0)
        adapter = replace(self.adapter, compute=lambda i: ComputeOutput(expected(i)))
        engine = IncrementalExecutor(self.state, (adapter,), authorize=lambda s, c: True)
        result = self.run_graph(engine=engine)
        self.assertIsNone(result.steps[0].usage.model_calls)
        self.assertEqual(self.run_graph(engine=engine).steps[0].usage.model_calls, 0)

    def test_explicit_no_reuse_and_closed_executor(self):
        graph = tuple(replace(s, reusable=False) if s.key == "b" else s for s in self.graph)
        self.run_graph(graph)
        result = self.run_graph(graph)
        self.assertEqual([s.key for s in result.steps if s.status == "computed"], ["b"])
        self.engine.close()
        self.assertEqual(self.run_graph().status, "invalidated")

    def test_invalid_specs_adapters_and_bounds(self):
        for request in ('{"x":1,"x":2}', '[1]', '{"x":NaN}', '{"x":Infinity}', '{"x":1e999}'):
            with self.assertRaises(ValueError):
                replace(self.graph[0], request=request)
        with self.assertRaises(ValueError):
            replace(self.adapter, pure=False)
        for kwargs in (dict(capacity=True), dict(max_bytes=0), dict(ttl_seconds=float("nan")), dict(signing_key=b"x")):
            with self.assertRaises(ValueError):
                IncrementalExecutor(self.state, (self.adapter,), authorize=lambda s, c: True, **kwargs)
        self.assertEqual(replace(self.graph[0], request='{"b": 2, "a": 1}').request, '{"a":1,"b":2}')

    def test_reopened_durable_source_store_integrates(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "state.sqlite"
            kwargs = dict(tenant="lab", signing_key=b"s" * 32, authorize=lambda s, op, subject: s == self.scope,
                          clock=lambda: 100)
            with ContextStore(path, create=True, policy_revision="p1", **kwargs) as state:
                for source in (self.a, self.b, self.x):
                    state.put(source, scope=self.scope, mutation_id=source.key)
                anchor = state.checkpoint(scope=self.scope)
            with ContextStore(path, checkpoint=anchor, **kwargs) as state:
                engine = IncrementalExecutor(state, (self.adapter,), authorize=lambda s, c: s == self.scope)
                self.assertEqual(self.run_graph(engine=engine).status, "complete")
                self.assertTrue(all(s.status == "reused" for s in self.run_graph(engine=engine).steps))
                state.set_roles(self.b.key, (), scope=self.scope, mutation_id="revoke")
                self.assertEqual(self.run_graph(engine=engine).status, "unavailable_context")


if __name__ == "__main__":
    unittest.main()

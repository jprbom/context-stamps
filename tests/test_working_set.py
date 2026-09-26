"""Working-set quotas, retained evidence and adversarial prefetch interleavings."""

import tempfile
import threading
import time
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from context_stamps.context_compiler import CompileBudget, ContextCompiler, ContextTask, ModelProfile
from context_stamps.context_state import (
    AccessScope,
    CanonicalNode,
    ContextClaim,
    EvidenceRequirement,
    TemporalScope,
)
from context_stamps.state_store import ContextStore
from context_stamps.working_set import ContextWorkingSet


class WorkingSetTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="context-working-set-")
        self.addCleanup(self.temp.cleanup)
        self.scope = AccessScope("lab", "researcher", "p1", ("reader",))
        self.hidden = set()
        self.store = ContextStore(Path(self.temp.name) / "state.sqlite", tenant="lab", signing_key=b"k" * 32,
            authorize=lambda s, op, subject: s == self.scope and getattr(subject, "key", None) not in self.hidden,
            create=True, policy_revision="p1", clock=lambda: 100)
        self.addCleanup(self.store.close)
        self.policy = self.node("policy", claims=(ContextClaim("limit", "32"),))
        self.first = self.node("first", dependencies=(self.policy.ref,))
        self.second = self.node("second")
        for node in (self.policy, self.first, self.second):
            self.store.put(node, scope=self.scope, mutation_id=node.key)
        self.memory = self.working(max_nodes=3)

    def node(self, key, **changes):
        args = dict(key=key, revision="v1", tenant="lab", text=key + " " + "x" * 2000, kind="POLICY",
                    temporal=TemporalScope(0, 10), roles=("reader",), provenance="fixture", validation_revision="v1")
        args.update(changes)
        return CanonicalNode(**args)

    def working(self, **kwargs):
        memory = ContextWorkingSet(self.store, scope=self.scope, **kwargs)
        self.addCleanup(memory.close)
        return memory

    def activate(self, *refs):
        return self.memory.activate(tuple(refs), at=100, known_at=100)

    def finish(self, ticket):
        deadline = time.perf_counter() + 5
        while time.perf_counter() < deadline:
            result = self.memory.prefetch_result(ticket)
            if result.status != "pending":
                return result
            time.sleep(.005)
        self.fail("prefetch did not finish")

    def test_dependency_closure_and_warm_reuse_without_disk_payload_reads(self):
        snapshot = self.activate(self.first.ref)
        self.assertEqual({r.node.ref for r in snapshot.records}, {self.first.ref, self.policy.ref})
        self.assertEqual(self.memory.stats()["cold_reads"], 2)
        with patch.object(self.store, "_row", side_effect=AssertionError("unexpected cold read")):
            self.assertEqual(self.activate(self.first.ref), snapshot)
        self.assertEqual(self.memory.stats()["cache_reuses"], 2)

    def test_pin_keeps_its_closure_and_quota_refusal_preserves_active_window(self):
        self.memory = self.working(max_nodes=2)
        snapshot = self.memory.pin(self.first.ref, at=100, known_at=100)
        with self.assertRaises(OverflowError):
            self.activate(self.second.ref)
        self.assertTrue(self.memory.is_current(snapshot))
        self.assertEqual(self.memory.stats()["resident_nodes"], 2)
        for ref in (self.first.ref, self.policy.ref):
            with self.assertRaises(ValueError):
                self.memory.page_out(ref)
        self.memory.unpin(self.first.ref)
        self.activate(self.second.ref)
        self.assertEqual(self.memory.stats()["active_nodes"], 1)

    def test_eviction_changes_residency_without_deleting_retained_evidence(self):
        self.memory = self.working(max_nodes=2)
        old = self.activate(self.first.ref)
        self.activate(self.second.ref)
        self.assertFalse(self.memory.is_current(old))
        self.assertEqual(self.memory.stats()["resident_nodes"], 2)
        self.memory.collect(idle_seconds=0)
        self.assertEqual(self.memory.stats()["resident_nodes"], 1)
        self.assertEqual(len(self.store.snapshot(self.scope, at=100, known_at=100).records), 3)
        self.activate(self.first.ref)
        self.assertEqual(self.memory.stats()["active_nodes"], 2)

    def test_page_fault_loads_missing_root_and_required_dependencies(self):
        self.activate(self.second.ref)
        snapshot = self.memory.fault(self.first.ref, at=100, known_at=100)
        self.assertEqual(len(snapshot.records), 3)
        self.assertEqual(self.memory.stats()["page_faults"], 1)
        self.assertEqual(self.memory.stats()["active_nodes"], 3)

    def test_residency_is_separate_from_retained_version_lifecycle(self):
        self.assertEqual(self.memory.residency(self.first.ref, at=100, known_at=100), "COLD")
        self.activate(self.first.ref)
        self.assertEqual(self.memory.residency(self.first.ref, at=100, known_at=100), "HOT")
        self.activate(self.second.ref)
        self.assertEqual(self.memory.residency(self.first.ref, at=100, known_at=100), "WARM")
        self.memory.page_out(self.first.ref)
        self.assertEqual(self.memory.residency(self.first.ref, at=100, known_at=100), "COLD")
        self.hidden.add("first")
        with self.assertRaises(PermissionError):
            self.memory.residency(self.first.ref, at=100, known_at=100)

    def test_serialized_byte_quota_is_checked_before_loading(self):
        memory = self.working(max_bytes=10)
        with patch.object(self.store, "_row", side_effect=AssertionError("loaded before budget check")):
            with self.assertRaises(OverflowError):
                memory.activate((self.first.ref,), at=100, known_at=100)
        self.assertEqual(memory.stats()["resident_bytes"], 0)

    def test_prefetch_is_inactive_until_demand_and_has_usefulness_accounting(self):
        active = self.activate(self.second.ref)
        ticket = self.memory.prefetch((self.first.ref,), at=100, known_at=100)
        self.assertEqual(self.finish(ticket).status, "ready")
        stats = self.memory.stats()
        self.assertEqual(stats["active_nodes"], 1)
        self.assertEqual(stats["prefetched"], 2)
        self.assertEqual(stats["prefetch_used"], 0)
        self.assertTrue(self.memory.is_current(active))
        with patch.object(self.store, "_row", side_effect=AssertionError("prefetched body reread")):
            self.activate(self.first.ref)
        self.assertEqual(self.memory.stats()["prefetch_usefulness"], 1.)

    def test_unused_prefetch_cost_is_retained_when_evicted(self):
        self.activate(self.second.ref)
        ticket = self.memory.prefetch((self.first.ref,), at=100, known_at=100)
        self.assertEqual(self.finish(ticket).status, "ready")
        self.assertEqual(self.memory.collect(idle_seconds=0), 2)
        stats = self.memory.stats()
        self.assertEqual(stats["prefetched"], stats["wasted_prefetch"])
        self.assertEqual(stats["prefetch_usefulness"], 0.)

    def block_prefetch(self, *, after_read=False):
        entered, release = threading.Event(), threading.Event()
        original = self.store.materialize
        def delayed(*args, **kwargs):
            if not threading.current_thread().name.startswith("context-prefetch"):
                return original(*args, **kwargs)
            result = original(*args, **kwargs) if after_read else None
            entered.set()
            if not release.wait(5):
                raise RuntimeError("test barrier expired")
            return result if after_read else original(*args, **kwargs)
        return entered, release, patch.object(self.store, "materialize", side_effect=delayed)

    def test_late_prefetch_does_not_override_changed_active_window(self):
        entered, release, patched = self.block_prefetch()
        with patched:
            try:
                ticket = self.memory.prefetch((self.first.ref,), at=100, known_at=100)
                self.assertTrue(entered.wait(5))
                self.activate(self.second.ref)
            finally:
                release.set()
            self.assertEqual(self.finish(ticket).status, "discarded")
        self.assertEqual(self.memory.stats()["active_nodes"], 1)
        self.assertEqual(self.memory.stats()["resident_nodes"], 1)
        self.assertEqual(self.memory.stats()["discarded_prefetch"], 1)

    def test_prefetch_cannot_cancel_foreground_page_in(self):
        entered, release, results = threading.Event(), threading.Event(), []
        original = self.store.materialize
        def delayed(*args, **kwargs):
            if threading.current_thread().name == "foreground-page":
                entered.set()
                if not release.wait(5):
                    raise RuntimeError("barrier expired")
            return original(*args, **kwargs)
        with patch.object(self.store, "materialize", side_effect=delayed):
            worker = threading.Thread(target=lambda: results.append(self.activate(self.second.ref)), name="foreground-page")
            worker.start()
            try:
                self.assertTrue(entered.wait(5))
                ticket = self.memory.prefetch((self.first.ref,), at=100, known_at=100)
                self.assertEqual(self.finish(ticket).status, "deferred")
            finally:
                release.set()
                worker.join(5)
            self.assertFalse(worker.is_alive())
        self.assertEqual(len(results), 1)
        self.assertTrue(self.memory.is_current(results[0]))

    def test_source_revocation_during_prefetch_suppresses_cached_output(self):
        entered, release, patched = self.block_prefetch(after_read=True)
        with patched:
            try:
                ticket = self.memory.prefetch((self.first.ref,), at=100, known_at=100)
                self.assertTrue(entered.wait(5))
                self.hidden.add("policy")
            finally:
                release.set()
            self.assertEqual(self.finish(ticket).status, "stale")
        self.assertEqual(self.memory.stats()["resident_nodes"], 0)
        with self.assertRaises(PermissionError):
            self.activate(self.first.ref)

    def test_running_prefetch_cancellation_discards_late_work(self):
        entered, release, patched = self.block_prefetch()
        with patched:
            try:
                ticket = self.memory.prefetch((self.first.ref,), at=100, known_at=100)
                self.assertTrue(entered.wait(5))
                self.assertTrue(self.memory.cancel_prefetch(ticket))
            finally:
                release.set()
            self.assertEqual(self.finish(ticket).status, "discarded")
        self.assertEqual(self.memory.stats()["resident_nodes"], 0)
        self.assertEqual(self.memory.stats()["cold_reads"], 2)

    def test_prefetch_queue_and_budget_are_bounded(self):
        self.memory = self.working(max_nodes=1, max_prefetch=1)
        ticket = self.memory.prefetch((self.first.ref,), at=100, known_at=100)
        with self.assertRaises(OverflowError):
            self.memory.prefetch((self.second.ref,), at=100, known_at=100)
        self.assertEqual(self.finish(ticket).status, "budget")
        self.assertEqual(self.memory.stats()["resident_nodes"], 0)

    def test_revoked_pin_blocks_new_activation_and_can_be_explicitly_unpinned(self):
        self.memory.pin(self.policy.ref, at=100, known_at=100)
        self.hidden.add("policy")
        with self.assertRaises(PermissionError):
            self.activate(self.second.ref)
        self.assertEqual(self.memory.stats()["resident_nodes"], 0)
        self.memory.unpin(self.policy.ref)
        self.assertEqual(len(self.activate(self.second.ref).records), 1)

    def test_compiler_operates_only_on_active_roots_and_pins(self):
        self.activate(self.first.ref)
        task = ContextTask("approve", "What is the limit?", (EvidenceRequirement("limit"),))
        result = ContextCompiler(self.memory).compile(task, scope=self.scope, at=100, known_at=100,
                    profile=ModelProfile("exact-v1"), budget=CompileBudget(milliseconds=5000))
        self.assertEqual(result.status, "complete")
        self.assertEqual({r.node.ref for r in result.snapshot.records}, {self.policy.ref})
        with self.assertRaises(PermissionError):
            self.memory.snapshot(replace(self.scope, principal="other"), at=100, known_at=100)

    def test_closed_working_set_does_not_close_cold_storage(self):
        self.activate(self.first.ref)
        self.memory.close()
        with self.assertRaises(RuntimeError):
            self.activate(self.second.ref)
        self.assertEqual(len(self.store.snapshot(self.scope, at=100, known_at=100).records), 3)


if __name__ == "__main__":
    unittest.main()

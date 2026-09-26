"""Persistent state equivalence, current authorization and cold payload integrity."""

import json
import multiprocessing
import os
import sqlite3
import tempfile
import unittest
from contextlib import closing
from dataclasses import asdict, replace
from pathlib import Path

from context_stamps.audit import AuditIntegrityError
from context_stamps.context_compiler import CompileBudget, ContextCompiler, ContextTask, ModelProfile
from context_stamps.context_state import (
    AccessScope,
    CanonicalNode,
    ContextClaim,
    ContextState,
    EvidenceRequirement,
    TemporalScope,
)
from context_stamps.state_store import ContextStore


def racing_put(path, node, scope, output):
    with ContextStore(path, tenant="lab", signing_key=b"k" * 32, authorize=lambda *args: True) as store:
        receipt = store.put(node, scope=scope, mutation_id="shared-write")
        Path(output).write_text(json.dumps(asdict(receipt), sort_keys=True))


def crash_mutation(path, node, scope, before):
    def authorize(actor, operation, subject):
        if before and operation == "put" and subject is not None:
            # The second typed authorization check is after INSERT, before COMMIT.
            authorize.calls += 1
            if authorize.calls == 2:
                os._exit(21)
        return True
    authorize.calls = 0
    with ContextStore(path, tenant="lab", signing_key=b"k" * 32, authorize=authorize) as store:
        store.put(node, scope=scope, mutation_id="crash-write")
        os._exit(22)


class StateStoreTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="context-state-test-")
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "state.sqlite"
        self.now, self.allowed, self.hidden = 100, True, set()
        self.scope = AccessScope("lab", "researcher", "p1", ("reader",))
        self.store = self.open(create=True, policy_revision="p1")

    def authorize(self, scope, operation, subject):
        return self.allowed and scope.principal == "researcher" and getattr(subject, "key", None) not in self.hidden

    def open(self, **changes):
        args = dict(tenant="lab", signing_key=b"k" * 32, authorize=self.authorize, clock=lambda: self.now)
        args.update(changes)
        store = ContextStore(self.path, **args)
        self.addCleanup(store.close)
        return store

    def node(self, key="policy", revision="v1", **changes):
        args = dict(key=key, revision=revision, tenant="lab", text="batch limit 32", kind="POLICY",
                    temporal=TemporalScope(0, 10), roles=("reader",), provenance="fixture",
                    claims=(ContextClaim("limit", "32"),), validation_revision="checked-v1")
        args.update(changes)
        return CanonicalNode(**args)

    def put(self, node, mutation_id=None):
        return self.store.put(node, scope=self.scope, mutation_id=mutation_id or node.key + ":" + node.revision)

    def snapshot(self, **changes):
        args = dict(at=100, known_at=1000)
        args.update(changes)
        return self.store.snapshot(self.scope, **args)

    def test_restart_preserves_nodes_times_bindings_and_receipts(self):
        node = self.node()
        receipt = self.put(node)
        snapshot = self.snapshot()
        seal = self.store.seal(snapshot, "compiled packet")
        checkpoint = self.store.checkpoint(scope=self.scope, verify=True)
        self.store.close()
        self.store = self.open(checkpoint=checkpoint)
        self.assertEqual(self.snapshot(), snapshot)
        self.assertTrue(self.store.verify_binding(snapshot, "compiled packet", seal))
        self.assertEqual(self.put(node), receipt)
        self.assertEqual(snapshot.records[0].transaction_time, 100)

    def test_metadata_catalogue_does_not_retain_document_bodies(self):
        node = self.node(text="x" * 60000)
        self.put(node)
        header = self.store._catalog[node.key, node.revision].node
        self.assertFalse(hasattr(header, "text"))
        self.assertFalse(hasattr(header, "claims"))
        inventory = self.store.inventory(self.scope, at=100, known_at=1000)
        self.assertEqual(inventory.entries[0].ref, node.ref)
        self.assertGreater(inventory.entries[0].payload_bytes, 60000)
        self.assertEqual(self.snapshot().records[0].node, node)

    def test_temporal_and_supersession_parity_with_in_memory_state(self):
        state = ContextState(tenant="lab", policy_revision="p1", clock=lambda: self.now)
        old = self.node()
        self.put(old)
        state.put(old)
        self.now = 200
        new = self.node(revision="v2", text="batch limit 64", temporal=TemporalScope(50, 150, valid_until=120),
                        claims=(ContextClaim("limit", "64"),), supersedes=(old.ref,))
        self.put(new)
        state.put(new)
        for at in (0, 40, 50, 100, 119, 120, 300):
            for known in (0, 99, 100, 150, 199, 200, 300):
                self.assertEqual(self.snapshot(at=at, known_at=known).records,
                                 state.snapshot(self.scope, at=at, known_at=known).records)
        self.store.invalidate(new.ref, scope=self.scope, mutation_id="revoke")
        state.invalidate(new.ref)
        self.assertEqual(self.snapshot().records, state.snapshot(self.scope, at=100, known_at=1000).records)

    def test_current_acl_and_hidden_dependency_filter_survive_restart(self):
        source = self.node()
        derived = self.node("derived", dependencies=(source.ref,))
        self.put(source)
        self.put(derived)
        snapshot = self.snapshot()
        self.store.set_roles("policy", (), scope=self.scope, mutation_id="remove-access")
        self.assertFalse(self.store.is_current(snapshot))
        self.assertEqual(self.snapshot().records, ())
        self.store.close()
        self.store = self.open()
        self.assertEqual(self.snapshot(at=50, known_at=100).records, ())
        with self.assertRaises(PermissionError):
            self.snapshot(roots=(derived.ref,))

    def test_inverse_lineage_respects_knowledge_cutoff_and_current_source_permissions(self):
        old = self.node()
        self.put(old)
        self.now = 200
        new = self.node(revision="v2", temporal=TemporalScope(50, 150), supersedes=(old.ref,))
        self.put(new)
        self.assertEqual(self.store.lineage(old.ref, scope=self.scope, known_at=199).superseded_by, ())
        self.assertEqual(self.store.lineage(old.ref, scope=self.scope, known_at=200).superseded_by, (new.ref,))
        self.assertEqual(self.store.lineage(new.ref, scope=self.scope, known_at=200).supersedes, (old.ref,))
        self.store.set_roles(old.key, (), scope=self.scope, mutation_id="deny-lineage")
        with self.assertRaises(PermissionError):
            self.store.lineage(old.ref, scope=self.scope, known_at=100)

    def test_host_source_denial_filters_dependents_and_cached_materialization(self):
        source = self.node()
        derived = self.node("derived", dependencies=(source.ref,))
        self.put(source)
        self.put(derived)
        snapshot = self.snapshot()
        inventory = self.store.inventory(self.scope, at=100, known_at=1000)
        self.hidden.add("policy")
        self.assertFalse(self.store.is_current(snapshot))
        self.assertEqual(self.snapshot().records, ())
        with self.assertRaises(PermissionError):
            self.store.materialize(inventory, cached=snapshot.records)

    def test_policy_change_invalidates_other_connection_and_old_scope(self):
        self.put(self.node())
        snapshot = self.snapshot()
        other = self.open()
        other.set_policy("p2", scope=self.scope, mutation_id="new-policy")
        self.assertFalse(self.store.is_current(snapshot))
        with self.assertRaises(PermissionError):
            self.snapshot()
        self.scope = replace(self.scope, policy_revision="p2")
        self.assertEqual(len(self.snapshot().records), 1)

    def test_inventory_closure_limits_and_cache_avoid_cold_payload_reads(self):
        source = self.node()
        derived = self.node("derived", dependencies=(source.ref,))
        self.put(source)
        self.put(derived)
        self.put(self.node("unrelated"))
        inventory = self.store.inventory(self.scope, at=100, known_at=1000, roots=(derived.ref,))
        self.assertEqual({e.ref for e in inventory.entries}, {source.ref, derived.ref})
        snapshot = self.store.materialize(inventory)
        original = self.store._row
        self.store._row = lambda sequence: self.fail("warm snapshot reread a cold payload")
        try:
            self.assertEqual(self.store.materialize(inventory, cached=snapshot.records), snapshot)
        finally:
            self.store._row = original
        self.store.materialize_bytes = 1
        with self.assertRaises(OverflowError):
            self.store.materialize(inventory)

    def test_stale_and_tampered_inventory_and_cached_payload_rejected(self):
        node = self.node()
        self.put(node)
        inventory = self.store.inventory(self.scope, at=100, known_at=1000)
        forged = replace(inventory, entries=(replace(inventory.entries[0], payload_bytes=0),))
        with self.assertRaises(ValueError):
            self.store.materialize(forged)
        cached = self.snapshot().records
        with self.assertRaises(ValueError):
            self.store.materialize(inventory, cached=(replace(cached[0], transaction_time=99),))
        self.put(self.node("other"))
        with self.assertRaises(ValueError):
            self.store.materialize(inventory, cached=cached)

    def test_retries_collisions_and_revision_mutation_rejected(self):
        node = self.node()
        receipt = self.put(node)
        self.assertEqual(self.put(node), receipt)
        with self.assertRaises(ValueError):
            self.put(replace(node, text="changed"))
        with self.assertRaises(ValueError):
            self.put(node, "another-id")
        self.assertEqual(self.store.checkpoint(scope=self.scope).sequence, 1)

    def test_compiler_works_after_store_restart(self):
        self.put(self.node())
        self.store.close()
        self.store = self.open()
        task = ContextTask("approve", "What is the batch limit?", (EvidenceRequirement("limit"),))
        context = ContextCompiler(self.store).compile(task, scope=self.scope, at=100, known_at=1000,
                    profile=ModelProfile("exact-v1"), budget=CompileBudget(milliseconds=5000))
        self.assertEqual(context.status, "complete")
        self.assertIn("batch limit 32", context.text)

    def test_denied_commit_rolls_back_without_mutating_catalogue(self):
        calls = []
        def authorize(scope, operation, subject):
            if subject is not None:
                calls.append(subject)
            return len(calls) < 2
        self.store._authorize = authorize
        with self.assertRaises(PermissionError):
            self.put(self.node())
        self.store._authorize = self.authorize
        self.assertEqual(self.snapshot().records, ())
        self.assertEqual(self.store.checkpoint(scope=self.scope).sequence, 0)

    def test_wrong_key_tenant_and_store_checkpoint_rejected(self):
        self.put(self.node())
        checkpoint = self.store.checkpoint(scope=self.scope)
        for changes in ({"signing_key": b"z" * 32}, {"tenant": "other"},
                        {"checkpoint": replace(checkpoint, store_id="0" * 64)}):
            with self.assertRaises(AuditIntegrityError):
                self.open(**changes)

    def test_cold_payload_tamper_detected_and_instance_poisoned(self):
        self.put(self.node())
        with closing(sqlite3.connect(self.path)) as connection, connection:
            connection.execute("UPDATE mutations SET payload=replace(payload,'batch limit 32','altered limit!') WHERE sequence=1")
        with self.assertRaises(AuditIntegrityError):
            self.snapshot()
        with self.assertRaises(AuditIntegrityError):
            self.store.checkpoint(scope=self.scope)

    def test_valid_prefix_rollback_detected_with_external_checkpoint(self):
        self.put(self.node())
        self.put(self.node("second"))
        checkpoint = self.store.checkpoint(scope=self.scope)
        self.store.close()
        with closing(sqlite3.connect(self.path)) as connection, connection:
            connection.execute("DELETE FROM mutations WHERE sequence=2")
        with self.assertRaises(AuditIntegrityError):
            self.open(checkpoint=checkpoint)

    def test_bounds_and_future_observations(self):
        with self.assertRaises(ValueError):
            self.put(self.node(temporal=TemporalScope(0, 101)))
        self.store.capacity = 1
        self.put(self.node())
        with self.assertRaises(OverflowError):
            self.put(self.node("other"))
        self.store.max_events = 1
        with self.assertRaises(OverflowError):
            self.store.set_roles("policy", (), scope=self.scope, mutation_id="roles")

    def test_two_processes_retry_one_committed_mutation(self):
        ctx = multiprocessing.get_context("spawn")
        outputs = [self.path.parent / f"writer-{i}.json" for i in range(2)]
        workers = [ctx.Process(target=racing_put, args=(str(self.path), self.node(), self.scope, str(p))) for p in outputs]
        try:
            for worker in workers:
                worker.start()
            for worker in workers:
                worker.join(15)
                self.assertEqual(worker.exitcode, 0)
            self.assertEqual(outputs[0].read_text(), outputs[1].read_text())
            self.assertEqual(self.store.checkpoint(scope=self.scope).sequence, 1)
        finally:
            for worker in workers:
                if worker.is_alive():
                    worker.kill()
                    worker.join(2)
                worker.close()

    def test_actual_process_crash_before_and_after_commit(self):
        ctx = multiprocessing.get_context("spawn")
        for before, expected in ((True, 0), (False, 1)):
            worker = ctx.Process(target=crash_mutation, args=(str(self.path), self.node(), self.scope, before))
            try:
                worker.start()
                worker.join(15)
                self.assertEqual(worker.exitcode, 21 if before else 22)
                self.assertEqual(self.store.checkpoint(scope=self.scope).sequence, expected)
                self.assertEqual(len(self.snapshot(known_at=253402300799999).records), expected)
            finally:
                if worker.is_alive():
                    worker.kill()
                    worker.join(2)
                worker.close()


if __name__ == "__main__":
    unittest.main()

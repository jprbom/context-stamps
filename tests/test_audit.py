"""Durability, ordering, authorization and integrity checks using fictional IDs."""

import json
import sqlite3
import subprocess
import sys
import tempfile
import unittest
from contextlib import closing
from dataclasses import replace
from pathlib import Path

from test_experience import fixtures

from context_stamps.audit import AuditIntegrityError, AuditStore
from context_stamps.experience import EpisodeEnd, encode_record, references


class AuditTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="context-audit-test-")
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "audit.sqlite"
        self.episode, self.plan, self.outcome = fixtures()
        self.actor = self.episode.authority
        self.allowed, self.hidden = True, set()
        self.now = 1000

    def authorize(self, actor, operation, record):
        return (self.allowed and actor == self.actor and
                (record is None or all(ref.source not in self.hidden for ref in references(record))))

    def store(self, **changes):
        args = dict(tenant="lab", signing_key=b"k" * 32, authorize=self.authorize,
                    create=True, clock=lambda: self.now)
        args.update(changes)
        store = AuditStore(self.path, **args)
        self.addCleanup(store.close)
        return store

    def append(self, store, record):
        return store.append(record, authority=self.actor)

    def history(self, store):
        return store.history("ep1", authority=self.actor)

    def test_restart_preserves_typed_records_receipts_and_checkpoint(self):
        store = self.store()
        records = (self.episode, self.plan, self.outcome, EpisodeEnd("ep1", "succeeded", 103))
        receipts = [self.append(store, r) for r in records]
        anchor = store.checkpoint(authority=self.actor)
        store.close()
        reopened = self.store(create=False, checkpoint=anchor)
        self.assertEqual([e.record for e in self.history(reopened)], list(records))
        self.assertEqual([e.receipt for e in self.history(reopened)], receipts)
        self.assertEqual(reopened.verify(authority=self.actor), anchor)

    def test_same_event_retry_is_idempotent_and_changed_retry_is_rejected(self):
        store = self.store()
        receipt = self.append(store, self.episode)
        self.assertEqual(self.append(store, self.episode), receipt)
        with self.assertRaises(ValueError):
            self.append(store, replace(self.episode, seed=29))
        self.assertEqual(len(self.history(store)), 1)

    def test_committed_plan_precedes_outcome_and_steps_are_sequential(self):
        store = self.store()
        self.append(store, self.episode)
        for record in (self.outcome, replace(self.plan, step=1)):
            with self.assertRaises(ValueError):
                self.append(store, record)
        self.append(store, self.plan)
        with self.assertRaises(ValueError):
            self.append(store, replace(self.plan, step=1))
        with self.assertRaises(ValueError):
            self.append(store, EpisodeEnd("ep1", "unknown", 103))
        self.append(store, self.outcome)
        with self.assertRaises(ValueError):
            self.append(store, replace(self.plan, step=1, proposed_ms=101))

    def test_unknown_outcome_is_retained_and_never_overwritten_as_success(self):
        store = self.store()
        self.append(store, self.episode)
        self.append(store, self.plan)
        unknown = replace(self.outcome, status="unknown", actual_action=None, verified=False,
                          failure_code="worker-lost", rewards=())
        self.append(store, unknown)
        with self.assertRaises(ValueError):
            self.append(store, self.outcome)
        with self.assertRaises(ValueError):
            self.append(store, EpisodeEnd("ep1", "succeeded", 103))
        self.append(store, EpisodeEnd("ep1", "unknown", 103))
        self.assertEqual(self.history(store)[2].record.status, "unknown")

    def test_prior_failure_remains_visible_after_later_success(self):
        store = self.store()
        failure = replace(self.outcome, status="failed", verified=False, failure_code="check-failed", rewards=())
        second = replace(self.plan, step=1, proposed_ms=103)
        successful = replace(self.outcome, step=1, observed_ms=104)
        for record in (self.episode, self.plan, failure, second, successful, EpisodeEnd("ep1", "succeeded", 105)):
            self.append(store, record)
        statuses = [e.record.status for e in self.history(store) if hasattr(e.record, "status")]
        self.assertEqual(statuses, ["failed", "succeeded", "succeeded"])

    def test_action_idempotency_key_binds_all_declared_action_inputs(self):
        store = self.store()
        for record in (self.episode, self.plan, self.outcome):
            self.append(store, record)
        changed = replace(self.plan, step=1, proposed_ms=103,
                          proposed_action=replace(self.plan.proposed_action, tool_revision="v2"))
        with self.assertRaises(ValueError):
            self.append(store, changed)

    def test_foreign_authority_and_creation_impersonation_are_rejected(self):
        store = self.store(authorize=lambda *args: True)
        for actor in (replace(self.actor, tenant="foreign"), replace(self.actor, principal="other"),
                      replace(self.actor, policy_revision="other")):
            with self.assertRaises(PermissionError):
                store.append(self.episode, authority=actor)
        self.append(store, self.episode)
        with self.assertRaises(PermissionError):
            store.append(self.plan, authority=replace(self.actor, principal="other"))

    def test_current_source_revocation_hides_entire_history(self):
        store = self.store()
        self.append(store, self.episode)
        self.append(store, self.plan)
        self.hidden.add("config")
        with self.assertRaisesRegex(PermissionError, "^unavailable audit history$"):
            self.history(store)
        self.allowed = False
        with self.assertRaises(PermissionError):
            store.checkpoint(authority=self.actor)
        with self.assertRaises(PermissionError):
            self.append(store, self.outcome)

    def test_revocation_before_commit_rolls_back_and_retry_can_succeed(self):
        calls = []
        def revoke(actor, operation, record):
            calls.append(operation)
            return len(calls) == 1
        store = self.store(authorize=revoke)
        with self.assertRaises(PermissionError):
            self.append(store, self.episode)
        store.close()
        reopened = self.store(create=False)
        self.assertEqual(self.history(reopened), ())
        self.assertEqual(self.append(reopened, self.episode).checkpoint.sequence, 1)

    def test_wrong_key_tenant_and_different_journal_anchor_fail(self):
        store = self.store()
        self.append(store, self.episode)
        anchor = store.checkpoint(authority=self.actor)
        store.close()
        for changes in ({"signing_key": b"x" * 32}, {"tenant": "elsewhere"}):
            with self.assertRaises(AuditIntegrityError):
                self.store(create=False, **changes)
        self.path = Path(self.temp.name) / "different.sqlite"
        with self.assertRaises(AuditIntegrityError):
            self.store(checkpoint=anchor)

    def test_signed_tamper_detection_poisoned_instance_and_append_only_trigger(self):
        store = self.store()
        self.append(store, self.episode)
        with closing(sqlite3.connect(self.path)) as connection, connection:
            with self.assertRaises(sqlite3.IntegrityError):
                connection.execute("DELETE FROM events")
            connection.execute("DROP TRIGGER no_events_update")
            changed = encode_record(replace(self.episode, seed=99))
            connection.execute("UPDATE events SET record_json=? WHERE sequence=1", (changed,))
        with self.assertRaises(AuditIntegrityError):
            store.verify(authority=self.actor)
        with self.assertRaises(AuditIntegrityError):
            self.history(store)
        store.close()
        with self.assertRaises(AuditIntegrityError):
            self.store(create=False)

    def test_external_checkpoint_detects_complete_valid_prefix_rollback(self):
        store = self.store()
        self.append(store, self.episode)
        old_path = Path(self.temp.name) / "old.sqlite"
        with closing(sqlite3.connect(self.path)) as source, closing(sqlite3.connect(old_path)) as target:
            source.backup(target)
        self.append(store, self.plan)
        anchor = store.checkpoint(authority=self.actor)
        store.close()
        self.path = old_path
        with self.assertRaises(AuditIntegrityError):
            self.store(create=False, checkpoint=anchor)
        # Without an independently retained anchor a valid older prefix is indistinguishable.
        self.assertEqual(len(self.history(self.store(create=False))), 1)

    def test_bounded_capacity_future_events_and_monotonic_commit_time(self):
        store = self.store(max_records=1)
        self.append(store, self.episode)
        with self.assertRaises(OverflowError):
            self.append(store, self.plan)
        store.close()
        store = self.store()
        with self.assertRaises(ValueError):
            self.append(store, replace(self.plan, proposed_ms=2000))
        self.now = 900
        receipt = self.append(store, self.plan)
        self.assertEqual(receipt.committed_ms, 1000)

    def test_new_policy_can_record_outcome_only_when_host_authorizes_it(self):
        store = self.store(authorize=lambda a, op, r: a.tenant == "lab")
        creation = self.append(store, self.episode)
        self.append(store, self.plan)
        current = replace(self.actor, policy_revision="policy-v2")
        self.assertEqual(store.append(self.episode, authority=current), creation)
        receipt = store.append(self.outcome, authority=current)
        self.assertEqual(receipt.authority, current)
        self.assertEqual(self.history(store)[-1].record, self.outcome)
        store.close()
        self.assertEqual(self.history(self.store(create=False))[-1].receipt.authority, current)

    def test_authorizer_exception_does_not_expose_diagnostics(self):
        def fails(*args):
            raise RuntimeError("private-authorization-diagnostic")
        store = self.store(authorize=fails)
        with self.assertRaisesRegex(PermissionError, "^unavailable audit history$"):
            self.append(store, self.episode)

    def test_recorded_actor_cannot_be_changed_without_invalidating_receipt(self):
        store = self.store()
        self.append(store, self.episode)
        store.close()
        with closing(sqlite3.connect(self.path)) as connection, connection:
            connection.execute("DROP TRIGGER no_events_update")
            changed = json.dumps({"tenant": "lab", "principal": "other", "policy_revision": "policy-v1"})
            connection.execute("UPDATE events SET authority_json=?", (changed,))
        with self.assertRaises(AuditIntegrityError):
            self.store(create=False)

    def test_open_without_creation_does_not_create_missing_journal(self):
        with self.assertRaises(FileNotFoundError):
            self.store(create=False)
        self.assertFalse(self.path.exists())

    def test_crash_before_commit_rolls_back_and_after_commit_preserves_plan(self):
        store = self.store()
        self.append(store, self.episode)
        store.close()
        program = '''
import os, sys
from context_stamps.audit import AuditStore
from context_stamps.experience import Authority, decode_record
calls=0
def authorize(*args):
 global calls
 calls+=1
 if sys.argv[3]=='before' and calls==2:
  os._exit(91)
 return True
store=AuditStore(sys.argv[1],tenant='lab',signing_key=b'k'*32,authorize=authorize)
store.append(decode_record(sys.argv[2]),authority=Authority('lab','researcher','policy-v1'))
os._exit(92)
'''
        for phase, code, expected in (("before", 91, 1), ("after", 92, 2)):
            run = subprocess.run([sys.executable, "-c", program, str(self.path), encode_record(self.plan), phase],
                                 capture_output=True, text=True, timeout=15)
            self.assertEqual(run.returncode, code, run.stderr)
            reopened = self.store(create=False)
            self.assertEqual(len(self.history(reopened)), expected)
            reopened.close()

    def test_two_process_writers_return_one_idempotent_committed_event(self):
        self.store().close()
        program = '''
import json, sys, time
from context_stamps.audit import AuditStore
from context_stamps.experience import decode_record
record=decode_record(sys.argv[2])
def authorize(*args):
 time.sleep(.1)
 return True
with AuditStore(sys.argv[1],tenant='lab',signing_key=b'k'*32,authorize=authorize) as store:
 receipt=store.append(record,authority=record.authority)
 print(json.dumps([receipt.checkpoint.sequence, receipt.checkpoint.chain_hash]))
'''
        children = [subprocess.Popen([sys.executable, "-c", program, str(self.path), encode_record(self.episode)],
                                    stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True) for _ in range(2)]
        replies = []
        try:
            for child in children:
                output, error = child.communicate(timeout=15)
                self.assertEqual(child.returncode, 0, error)
                replies.append(json.loads(output))
        finally:
            for child in children:
                if child.poll() is None:
                    child.kill()
                    child.communicate(timeout=5)
        self.assertEqual(replies[0], replies[1])
        self.assertEqual(replies[0][0], 1)
        self.assertEqual(len(self.history(self.store(create=False))), 1)


if __name__ == "__main__":
    unittest.main()

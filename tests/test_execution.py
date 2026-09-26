"""Real process cancellation, dispatch races and provider-effect reconciliation."""

import hashlib
import json
import multiprocessing
import os
import tempfile
import time
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from test_experience import fixtures

from context_stamps.audit import AuditStore, DispatchBlockedError
from context_stamps.execution import ExecutionAdapter, ManagedExecutor, WorkerResult, _reply
from context_stamps.experience import (
    ActionReconciliation,
    DispatchClaim,
    EvidenceRef,
    ResourceUse,
    decode_record,
    encode_record,
    references,
)


def now():
    return time.time_ns() // 1000000


def echo(payload, key, deadline):
    return WorkerResult(payload.upper(), ResourceUse(0, 0, 0, 1, 0.))


def verify_echo(payload, output):
    return output == payload.upper()


def reject(payload, output):
    return False


def hang(payload, key, deadline):
    # Deliberately ignores the deadline; the parent must terminate this worker.
    while True:
        time.sleep(.02)


def verifier_hang(payload, output):
    while True:
        time.sleep(.02)


def fail(payload, key, deadline):
    raise RuntimeError("private exception payload must never leave worker")


def crash(payload, key, deadline):
    os._exit(13)


def too_large(payload, key, deadline):
    return WorkerResult(b"x" * 65537, ResourceUse(0, 0, 0, 1, 0.))


def effect_then_crash(payload, key, deadline):
    # Offline fixture service with a committed externally observable effect.
    artifact = Path(payload.decode())
    artifact.write_text(key, encoding="ascii")
    os._exit(17)


def claim_in_child(path, plan, actor, output):
    with AuditStore(path, tenant="lab", signing_key=b"k" * 32, authorize=lambda *a: True) as store:
        try:
            entry = store.claim_dispatch(plan, authority=actor, deadline_ms=now() + 20000)
            Path(output).write_text(entry.record.attempt_id)
        except DispatchBlockedError:
            Path(output).write_text("blocked")


class ExecutionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="context-execution-test-")
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)
        self.path = self.directory / "audit.sqlite"
        episode, plan, self.fixture_outcome = fixtures()
        self.episode = replace(episode, split="development", started_ms=now() - 20)
        self.actor = self.episode.authority
        self.payload = b"fictional recorded input"
        self.plan = replace(plan, proposed_ms=now() - 10)
        self.allowed, self.current, self.hidden = True, True, set()
        self.store = AuditStore(self.path, tenant="lab", signing_key=b"k" * 32,
                                authorize=self.authorize, create=True)
        self.addCleanup(self.store.close)
        self.store.append(self.episode, authority=self.actor)

    def authorize(self, actor, operation, record):
        return (self.allowed and actor == self.actor and
                (record is None or all(r.source not in self.hidden for r in references(record))))

    def commit(self, effect="pure"):
        self.plan = replace(self.plan, proposed_action=replace(self.plan.proposed_action, effect=effect))
        self.store.append(self.plan, authority=self.actor)

    def publish(self, payload):
        fingerprint = hashlib.sha256(payload).hexdigest()
        (self.directory / fingerprint).write_bytes(payload)
        return EvidenceRef("lab", "result", "v1", fingerprint, "observation")

    def executor(self, run=echo, verify=verify_echo, **changes):
        action = self.plan.proposed_action
        args = dict(adapters=(ExecutionAdapter(action.tool, action.tool_revision, action.effect,
                    self.episode.verifier_revision, run, verify),),
                    resolve=lambda ref: self.payload, publish=self.publish, is_current=lambda p: self.current)
        args.update(changes)
        return ManagedExecutor(self.store, **args)

    def run_action(self, run=echo, verify=verify_echo, **changes):
        args = dict(authority=self.actor, deadline_ms=now() + 10000)
        args.update(changes)
        return self.executor(run, verify).run(self.plan, **args)

    def history(self):
        return self.store.history("ep1", authority=self.actor)

    def reconciliation(self, claim, effect_state="applied"):
        return ActionReconciliation("ep1", claim.step, claim.attempt_id, claim.plan_digest,
            claim.action, effect_state, (self.publish(b"provider terminal status"),),
            self.episode.verifier_revision, now())

    def test_success_replays_and_exact_v1_bytes_remain_compatible(self):
        self.commit()
        result = self.run_action()
        self.assertEqual(result.outcome.record.status, "succeeded")
        self.assertEqual(result.outcome.record.cost.model_calls, 0)
        self.assertEqual([type(e.record).__name__ for e in self.history()],
                         ["Episode", "TransitionPlan", "DispatchClaim", "TransitionOutcome"])
        self.assertEqual(json.loads(encode_record(result.outcome.record))["schema"], 1)
        claim = result.claim.record
        self.assertEqual(decode_record(encode_record(claim)), claim)
        checkpoint = self.store.verify(authority=self.actor)
        self.store.close()
        with AuditStore(self.path, tenant="lab", signing_key=b"k" * 32,
                        authorize=self.authorize, checkpoint=checkpoint) as reopened:
            self.assertEqual(reopened.verify(authority=self.actor), checkpoint)

    def test_same_plan_never_launches_twice(self):
        self.commit()
        self.run_action()
        with self.assertRaises(DispatchBlockedError):
            self.run_action()
        self.assertEqual(len(self.history()), 4)

    def test_dispatch_reserves_outcome_capacity_before_any_worker_starts(self):
        self.commit()
        self.store.max_records = 3
        with self.assertRaises(OverflowError):
            self.run_action()
        self.assertEqual(len(self.history()), 2)
        self.store.max_records = 100
        self.store.max_bytes = 65536
        with self.assertRaises(OverflowError):
            self.run_action()
        self.assertEqual(len(self.history()), 2)

    def test_other_writer_cannot_consume_a_claims_reserved_outcome_slot(self):
        self.commit()
        self.store.max_records = 4
        self.store.claim_dispatch(self.plan, authority=self.actor, deadline_ms=now() + 10000)
        self.store.close()
        with AuditStore(self.path, tenant="lab", signing_key=b"k" * 32,
                        authorize=self.authorize, max_records=4) as other:
            with self.assertRaises(OverflowError):
                other.append(replace(self.episode, episode_id="ep2", started_ms=now()), authority=self.actor)
            unknown = replace(self.fixture_outcome, status="unknown", actual_action=None, verified=False,
                              failure_code="worker_lost", observed_ms=now(), cost=ResourceUse(None, None, None, None, 1.))
            other.append(unknown, authority=self.actor)
            self.assertEqual(len(other.history("ep1", authority=self.actor)), 4)

    def test_uncommitted_plan_and_unregistered_or_mismatched_adapter_rejected(self):
        with self.assertRaises(ValueError):
            self.run_action()
        self.commit()
        adapter = ExecutionAdapter("another", "tool-v1", "pure", "verifier-v1", echo, verify_echo)
        with self.assertRaises(ValueError):
            self.executor(adapters=(adapter,)).run(self.plan, authority=self.actor, deadline_ms=now() + 10000)
        wrong = replace(adapter, tool="inspect", verifier_revision="other-verifier")
        with self.assertRaises(ValueError):
            self.executor(adapters=(wrong,)).run(self.plan, authority=self.actor, deadline_ms=now() + 10000)
        self.assertEqual(len(self.history()), 2)

    def test_input_hash_and_byte_bound_are_checked_before_claim(self):
        self.commit()
        for bad in (b"changed", b"x" * (1024 * 1024 + 1), "not bytes"):
            self.payload = bad
            with self.assertRaises(ValueError):
                self.run_action()
        self.assertEqual(len(self.history()), 2)

    def test_pre_cancelled_action_does_not_start_and_records_known_zero_usage(self):
        self.commit()
        result = self.run_action(cancelled=lambda: True)
        self.assertEqual(result.outcome.record.status, "cancelled")
        self.assertIsNone(result.outcome.record.actual_action)
        self.assertEqual(result.outcome.record.cost.tool_calls, 0)

    def test_noncooperative_worker_deadline_and_unknown_usage(self):
        self.commit()
        start = time.monotonic()
        result = self.run_action(hang, deadline_ms=now() + 700)
        self.assertLess(time.monotonic() - start, 5)
        self.assertEqual(result.outcome.record.status, "timed_out")
        self.assertIsNone(result.outcome.record.cost.model_calls)
        self.assertIsNone(result.outcome.record.cost.input_tokens)
        self.assertEqual(json.loads(encode_record(result.outcome.record))["schema"], 2)
        self.assertEqual(decode_record(encode_record(result.outcome.record)), result.outcome.record)

    def test_verifier_has_the_same_killable_deadline(self):
        self.commit()
        result = self.run_action(verify=verifier_hang, deadline_ms=now() + 700)
        self.assertEqual(result.outcome.record.status, "timed_out")
        self.assertFalse(result.outcome.record.verified)

    def test_inflight_cancellation_stops_worker(self):
        self.commit()
        after = time.monotonic() + .4
        result = self.run_action(hang, cancelled=lambda: time.monotonic() >= after)
        self.assertEqual(result.outcome.record.status, "cancelled")
        self.assertIsNone(result.outcome.record.actual_action)
        self.assertEqual(result.claim.record.action, self.plan.proposed_action)

    def test_wrong_result_is_not_published_or_promoted(self):
        self.commit()
        result = self.run_action(verify=reject)
        self.assertEqual(result.outcome.record.failure_code, "verification_failed")
        self.assertEqual(result.outcome.record.observations, ())
        self.assertEqual(result.outcome.record.cost.tool_calls, 1)

    def test_worker_errors_are_sanitized(self):
        self.commit()
        result = self.run_action(fail)
        self.assertEqual(result.outcome.record.failure_code, "adapter_error")
        self.assertNotIn("private exception", encode_record(result.outcome.record))

    def test_result_protocol_rejects_partial_duplicate_and_oversized_messages(self):
        reply = self.directory / "reply.json"
        for raw in (b'{"status":', b'{"status":"verified","status":"adapter_error"}',
                    b'{"status":"verified","output":"","usage":{}}', b"x" * 131073):
            reply.write_bytes(raw)
            with self.assertRaises(ValueError):
                _reply(self.directory)

    def test_unknown_usage_cannot_be_serialized_as_schema_one(self):
        outcome = replace(self.fixture_outcome, cost=ResourceUse(None, None, None, None, 1.))
        raw = json.loads(encode_record(outcome))
        self.assertEqual(raw["schema"], 2)
        raw["schema"] = 1
        with self.assertRaises(ValueError):
            decode_record(json.dumps(raw))

    def test_worker_crash_is_retained(self):
        self.commit()
        result = self.run_action(crash)
        self.assertEqual(result.outcome.record.failure_code, "worker_lost")
        self.assertEqual(result.outcome.record.status, "failed")

    def test_oversized_worker_output_is_rejected(self):
        self.commit()
        result = self.run_action(too_large)
        self.assertEqual(result.outcome.record.failure_code, "adapter_error")
        self.assertFalse(result.outcome.record.verified)

    def test_stale_context_before_and_during_execution(self):
        self.commit()
        self.current = False
        with self.assertRaises(ValueError):
            self.run_action()
        self.assertEqual(len(self.history()), 2)
        after = time.monotonic() + .3
        def current(plan):
            return time.monotonic() < after
        result = self.executor(hang, is_current=current).run(self.plan, authority=self.actor, deadline_ms=now() + 10000)
        self.assertEqual(result.outcome.record.failure_code, "stale_context")
        self.assertEqual(result.outcome.record.status, "abstained")

    def test_permission_revocation_kills_worker_and_leaves_claim_unresolved(self):
        self.commit()
        after = time.monotonic() + .3
        def revoke():
            if time.monotonic() >= after:
                self.allowed = False
            return False
        with self.assertRaises(PermissionError):
            self.run_action(hang, cancelled=revoke)
        self.allowed = True
        self.assertEqual(len(self.history()), 3)
        self.assertIsInstance(self.history()[-1].record, DispatchClaim)

    def test_unreadable_observation_blocks_claim_even_when_arguments_authorized(self):
        extra = replace(self.plan.observations[0], source="restricted-policy")
        self.plan = replace(self.plan, observations=(extra,))
        self.commit()
        self.hidden.add("restricted-policy")
        with self.assertRaises(PermissionError):
            self.store.claim_dispatch(self.plan, authority=self.actor, deadline_ms=now() + 10000)
        self.hidden.clear()
        self.assertEqual(len(self.history()), 2)

    def test_bad_published_digest_leaves_claim_unresolved(self):
        self.commit()
        with self.assertRaises(ValueError):
            self.executor(publish=lambda b: self.plan.observations[0]).run(
                self.plan, authority=self.actor, deadline_ms=now() + 10000)
        self.assertEqual(len(self.history()), 3)

    def test_startup_failure_preserves_claim_and_known_zero_usage(self):
        self.commit()
        with patch("multiprocessing.process.BaseProcess.start", side_effect=OSError("private path")):
            result = self.run_action()
        self.assertEqual(result.outcome.record.failure_code, "startup_failed")
        self.assertEqual(result.outcome.record.cost.tool_calls, 0)
        self.assertEqual(len(self.history()), 4)

    def test_mutation_while_publishing_suppresses_output(self):
        self.commit()
        def publish(raw):
            self.current = False
            return self.publish(raw)
        result = self.executor(publish=publish).run(self.plan, authority=self.actor, deadline_ms=now() + 10000)
        self.assertEqual(result.outcome.record.status, "abstained")
        self.assertEqual(result.outcome.record.observations, ())

    def test_effect_key_cannot_escape_blocking_by_changing_episode(self):
        self.commit("side_effect")
        self.run_action()
        episode = replace(self.episode, episode_id="ep2", started_ms=now())
        self.store.append(episode, authority=self.actor)
        plan = replace(self.plan, episode_id="ep2", proposed_ms=now())
        self.store.append(plan, authority=self.actor)
        with self.assertRaises(DispatchBlockedError):
            self.store.claim_dispatch(plan, authority=self.actor, deadline_ms=now() + 10000)

    def test_late_success_outcome_rejected(self):
        self.commit()
        claim = self.store.claim_dispatch(self.plan, authority=self.actor, deadline_ms=now() + 100)
        outcome = replace(self.fixture_outcome, observed_ms=claim.record.deadline_ms + 1)
        with self.assertRaises(ValueError):
            self.store.append(outcome, authority=self.actor)

    def test_reconciliation_binds_claim_verifier_and_observation_tenant(self):
        self.commit("side_effect")
        result = self.run_action(crash)
        recon = self.reconciliation(result.claim.record)
        for bad in (replace(recon, attempt_id="0" * 64), replace(recon, verifier_revision="wrong"),
                    replace(recon, observations=(replace(recon.observations[0], tenant="other"),))):
            with self.assertRaises(ValueError):
                self.store.reconcile(bad, authority=self.actor, verify=lambda r: True)
        self.assertEqual(len(self.history()), 4)

    def test_timeout_after_possible_effect_cannot_be_retried_without_reconciliation(self):
        self.commit("side_effect")
        result = self.run_action(hang, deadline_ms=now() + 500)
        self.assertEqual(result.outcome.record.status, "unknown")
        self.plan = replace(self.plan, step=1, proposed_ms=now())
        self.store.append(self.plan, authority=self.actor)
        with self.assertRaises(DispatchBlockedError):
            self.run_action()
        recon = self.reconciliation(result.claim.record, "not_applied")
        with self.assertRaises(ValueError):
            self.store.reconcile(recon, authority=self.actor, verify=lambda r: False)
        self.store.reconcile(recon, authority=self.actor, verify=lambda r: True)
        self.assertEqual(self.run_action().outcome.record.status, "succeeded")
        self.assertEqual(self.history()[3].record.status, "unknown")

    def test_crash_after_real_effect_requires_applied_reconciliation_and_stays_nonrepeatable(self):
        artifact = self.directory / "external-effect.txt"
        self.payload = str(artifact).encode()
        ref = replace(self.plan.proposed_action.arguments, digest=hashlib.sha256(self.payload).hexdigest())
        self.plan = replace(self.plan, proposed_action=replace(self.plan.proposed_action, arguments=ref))
        self.commit("idempotent")
        result = self.run_action(effect_then_crash)
        self.assertEqual(artifact.read_text(), self.plan.proposed_action.idempotency_key)
        self.assertEqual(result.outcome.record.status, "unknown")
        recon = self.reconciliation(result.claim.record)
        self.store.reconcile(recon, authority=self.actor,
                             verify=lambda r: artifact.read_text() == r.action.idempotency_key)
        self.plan = replace(self.plan, step=1, proposed_ms=now())
        self.store.append(self.plan, authority=self.actor)
        with self.assertRaises(DispatchBlockedError):
            self.run_action()
        self.assertEqual(self.history()[3].record.status, "unknown")
        self.store.verify(authority=self.actor)

    def test_reconciliation_cannot_rewrite_or_contradict_verified_success(self):
        self.commit("side_effect")
        result = self.run_action()
        with self.assertRaises(ValueError):
            self.store.reconcile(self.reconciliation(result.claim.record, "not_applied"),
                                 authority=self.actor, verify=lambda r: True)
        recon = self.reconciliation(result.claim.record)
        self.store.reconcile(recon, authority=self.actor, verify=lambda r: True)
        with self.assertRaises(ValueError):
            self.store.reconcile(replace(recon, effect_state="not_applied"), authority=self.actor, verify=lambda r: True)

    def test_claim_survives_crash_before_start_and_direct_control_append_is_rejected(self):
        self.commit("side_effect")
        claim = self.store.claim_dispatch(self.plan, authority=self.actor, deadline_ms=now() + 10000)
        with self.assertRaises(ValueError):
            self.store.append(claim.record, authority=self.actor)
        self.store.close()
        with AuditStore(self.path, tenant="lab", signing_key=b"k" * 32, authorize=self.authorize) as reopened:
            with self.assertRaises(DispatchBlockedError):
                reopened.claim_dispatch(self.plan, authority=self.actor, deadline_ms=now() + 10000)

    def test_two_processes_get_exactly_one_dispatch_claim(self):
        self.commit()
        ctx = multiprocessing.get_context("spawn")
        outputs = [self.directory / f"claim-{i}.txt" for i in range(2)]
        workers = [ctx.Process(target=claim_in_child, args=(str(self.path), self.plan, self.actor, str(p))) for p in outputs]
        try:
            for worker in workers:
                worker.start()
            for worker in workers:
                worker.join(15)
                self.assertEqual(worker.exitcode, 0)
            replies = [p.read_text() for p in outputs]
            self.assertEqual(replies.count("blocked"), 1)
            self.assertEqual(len(self.history()), 3)
            self.store.verify(authority=self.actor)
        finally:
            for worker in workers:
                if worker.is_alive():
                    worker.kill()
                    worker.join(2)
                worker.close()


if __name__ == "__main__":
    unittest.main()

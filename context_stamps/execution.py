"""Managed execution of trusted host adapters with durable dispatch claims.

Copyright (c) 2026 Prashant Jagtap. MIT License.
Workers are killable processes, not a sandbox for untrusted code. Adapters must
not launch unmanaged child processes. Remote effects require reconciliation.
"""

from __future__ import annotations

import base64
import hashlib
import json
import multiprocessing
import os
import tempfile
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path

from .audit import AuditEntry, AuditStore
from .experience import EvidenceRef, ResourceUse, TransitionOutcome, TransitionPlan, _id, _unique_object

MAX_INPUT_BYTES = 1024 * 1024
MAX_OUTPUT_BYTES = 65536
MAX_MESSAGE_BYTES = 2 * MAX_OUTPUT_BYTES


@dataclass(frozen=True)
class WorkerResult:
    output: bytes
    usage: ResourceUse

    def __post_init__(self):
        if type(self.output) is not bytes or len(self.output) > MAX_OUTPUT_BYTES or type(self.usage) is not ResourceUse:
            raise ValueError("bounded output bytes and typed adapter-reported usage required")


@dataclass(frozen=True)
class ExecutionAdapter:
    """Registered trusted functions must be importable by Python spawn workers.

    run(payload_bytes, idempotency_key, deadline_ms) -> WorkerResult
    verify(payload_bytes, output_bytes) -> bool
    Verification runs in the same killable worker and shares its deadline.
    """

    tool: str
    revision: str
    effect: str
    verifier_revision: str
    run: object
    verify: object

    def __post_init__(self):
        for value in (self.tool, self.revision, self.verifier_revision):
            _id(value)
        if self.effect not in ("pure", "idempotent", "side_effect") or not callable(self.run) or not callable(self.verify):
            raise ValueError("host-owned adapter functions and explicit effect required")


@dataclass(frozen=True)
class ExecutionResult:
    claim: AuditEntry
    outcome: AuditEntry


def _worker(adapter, payload, key, deadline_ms, directory):
    # The parent reads bounded JSON only after exit. No result is unpickled, and
    # partial writes cannot masquerade as complete replies.
    message = {"status": "adapter_error"}
    try:
        if time.time_ns() // 1000000 >= deadline_ms:
            message = {"status": "deadline"}
        else:
            result = adapter.run(payload, key, deadline_ms)
            if type(result) is not WorkerResult:
                raise ValueError("typed adapter result required")
            accepted = adapter.verify(payload, result.output) is True
            message = {"status": "verified" if accepted else "verification_failed",
                       "output": base64.b64encode(result.output).decode("ascii") if accepted else "",
                       "usage": asdict(result.usage)}
    except Exception:
        # No payloads, provider exception messages or tracebacks cross the boundary.
        message = {"status": "adapter_error"}
    raw = json.dumps(message, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    if len(raw) > MAX_MESSAGE_BYTES:
        raw = b'{"status":"protocol_error"}'
    pending = Path(directory) / "reply.pending"
    pending.write_bytes(raw)
    os.replace(pending, Path(directory) / "reply.json")


def _reply(directory):
    path = Path(directory) / "reply.json"
    if path.is_symlink() or not path.is_file() or path.stat().st_size > MAX_MESSAGE_BYTES:
        raise ValueError("missing or oversized worker reply")
    with path.open("rb") as stream:
        raw = stream.read(MAX_MESSAGE_BYTES + 1)
    if len(raw) > MAX_MESSAGE_BYTES:
        raise ValueError("worker reply limit exceeded")
    message = json.loads(raw, object_pairs_hook=_unique_object)
    if type(message) is not dict or message.get("status") not in (
            "verified", "verification_failed", "adapter_error", "deadline", "protocol_error"):
        raise ValueError("invalid worker status")
    if message["status"] in ("verified", "verification_failed"):
        if set(message) != {"status", "output", "usage"} or type(message["usage"]) is not dict:
            raise ValueError("invalid worker reply fields")
        if set(message["usage"]) != set(ResourceUse.__dataclass_fields__):
            raise ValueError("invalid usage fields")
        output = base64.b64decode(message["output"], validate=True)
        result = WorkerResult(output, ResourceUse(**message["usage"]))
        return message["status"], result
    if set(message) != {"status"}:
        raise ValueError("invalid failure reply")
    return message["status"], None


def _stop(process):
    if process.is_alive():
        process.terminate()
        process.join(.5)
    if process.is_alive():
        process.kill()
        process.join(.5)
    if process.is_alive():
        raise RuntimeError("owned worker could not be stopped; dispatch remains unresolved")
    process.close()


class ManagedExecutor:
    """Execute a committed proposal once per journal, with current host checks.

    Host callbacks resolve/publish private evidence and check current compiled
    context. They run in the parent and must return promptly. Only run/verify
    callbacks have process cancellation. This is not distributed exactly-once
    execution or an authentication, credential isolation or sandbox service.
    """

    def __init__(self, store, *, adapters, resolve, publish, is_current, poll_seconds=.02):
        if not isinstance(store, AuditStore) or not all(callable(fn) for fn in (resolve, publish, is_current)):
            raise ValueError("audit store and trusted host evidence/current-state callbacks required")
        if type(poll_seconds) not in (float, int) or not .001 <= poll_seconds <= .1:
            raise ValueError("bounded polling interval required")
        if type(adapters) is not tuple or not 1 <= len(adapters) <= 128 or any(type(a) is not ExecutionAdapter for a in adapters):
            raise ValueError("bounded immutable registered adapter tuple required")
        self._adapters = {(a.tool, a.revision): a for a in adapters}
        if len(self._adapters) != len(adapters):
            raise ValueError("duplicate adapter registration")
        self.store, self.resolve, self.publish, self.is_current = store, resolve, publish, is_current
        self.poll_seconds = poll_seconds

    def run(self, plan, *, authority, deadline_ms, cancelled=None):
        if type(plan) is not TransitionPlan:
            raise ValueError("typed committed proposal required")
        action = plan.proposed_action
        adapter = self._adapters.get((action.tool, action.tool_revision))
        if adapter is None or adapter.effect != action.effect:
            raise ValueError("proposal does not match a registered adapter")
        if cancelled is not None and not callable(cancelled):
            raise ValueError("host cancellation callback required")
        # Read authorization also checks all evidence references via the host.
        history = self.store.history(plan.episode_id, authority=authority)
        if (not history or history[0].record.verifier_revision != adapter.verifier_revision
                or not any(entry.record == plan for entry in history)):
            raise ValueError("committed proposal and exact verifier revision required")
        if self.is_current(plan) is not True:
            raise ValueError("context is not current")
        started = time.perf_counter()
        if type(deadline_ms) is not int or not 0 < deadline_ms - time.time_ns() // 1000000 <= 3600000:
            raise ValueError("future deadline of at most one hour required")
        monotonic_deadline = started + (deadline_ms - time.time_ns() // 1000000) / 1000
        payload = self.resolve(action.arguments)
        if (type(payload) is not bytes or len(payload) > MAX_INPUT_BYTES
                or hashlib.sha256(payload).hexdigest() != action.arguments.digest):
            raise ValueError("resolved inputs do not match the bounded committed artifact")
        claim_entry = self.store.claim_dispatch(plan, authority=authority, deadline_ms=deadline_ms)
        claim = claim_entry.record
        worker_started = False
        code, result, output_ref = "adapter_error", None, None

        def abort_code():
            if time.perf_counter() >= monotonic_deadline or time.time_ns() // 1000000 >= deadline_ms:
                return "deadline"
            if cancelled is not None and cancelled() is True:
                return "cancelled"
            self.store.check_dispatch(claim, authority=authority)
            if self.is_current(plan) is not True:
                return "stale_context"
            return None

        with tempfile.TemporaryDirectory(prefix="context-worker-") as directory:
            process = multiprocessing.get_context("spawn").Process(
                target=_worker, args=(adapter, payload, action.idempotency_key, deadline_ms, directory), daemon=True)
            try:
                code = abort_code()
                if code is None:
                    try:
                        process.start()
                        worker_started = True
                    except Exception:
                        code = "startup_failed"
                    if worker_started:
                        while process.is_alive():
                            code = abort_code()
                            if code is not None:
                                break
                            process.join(min(self.poll_seconds, max(0., monotonic_deadline - time.perf_counter())))
                        code = code or abort_code()
                        if code is None:
                            if process.exitcode != 0:
                                code = "worker_lost"
                            else:
                                try:
                                    code, result = _reply(directory)
                                except (ValueError, TypeError, OSError, RecursionError):
                                    code = "protocol_error"
            finally:
                if worker_started:
                    _stop(process)
                else:
                    process.close()

        code = abort_code() or code
        if code == "verified":
            output_ref = self.publish(result.output)
            if (type(output_ref) is not EvidenceRef or output_ref.kind != "observation"
                    or output_ref.tenant != authority.tenant
                    or output_ref.digest != hashlib.sha256(result.output).hexdigest()):
                raise ValueError("published result does not bind the verified bytes; dispatch unresolved")
            code = abort_code() or code
        success = code == "verified"
        if success:
            status = "succeeded"
        elif worker_started and action.effect != "pure":
            status = "unknown"
        else:
            status = {"deadline": "timed_out", "cancelled": "cancelled", "stale_context": "abstained"}.get(code, "failed")
        usage = (result.usage if result is not None else ResourceUse(
            *([None] * 4 if worker_started else [0] * 4), wall_ms=0.))
        usage = replace(usage, wall_ms=(time.perf_counter() - started) * 1000)
        # A started OS process does not prove that its adapter reached the action.
        # The claim preserves what was dispatched; actual_action needs a reply.
        outcome = TransitionOutcome(plan.episode_id, plan.step, action if result is not None else None,
            (output_ref,) if success else (), status, success, adapter.verifier_revision,
            None if success else code, usage, (), time.time_ns() // 1000000)
        receipt = self.store.append(outcome, authority=authority)
        # A denied commit/revocation raises and leaves a durable unresolved claim.
        self.store.history(plan.episode_id, authority=authority)
        if success and (self.is_current(plan) is not True or time.perf_counter() >= monotonic_deadline
                        or time.time_ns() // 1000000 >= deadline_ms):
            raise RuntimeError("committed result is no longer deliverable; inspect authorized history")
        return ExecutionResult(claim_entry, AuditEntry(outcome, receipt))

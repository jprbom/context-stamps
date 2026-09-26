"""Bounded transactional reference-only audit journal.

Copyright (c) 2026 Prashant Jagtap. MIT License.
Host-owned local storage and signing key; no remote authentication or executor.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import secrets
import sqlite3
import threading
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass, replace
from pathlib import Path

from .context_state import canonical, digest, timestamp
from .experience import (
    MAX_RECORD_BYTES,
    ActionReconciliation,
    Authority,
    DispatchClaim,
    Episode,
    EpisodeEnd,
    TransitionOutcome,
    TransitionPlan,
    decode_record,
    encode_record,
    record_digest,
    validate_transition,
)
from .security import identifier


class AuditIntegrityError(ValueError):
    """Stored history or its externally retained checkpoint cannot be verified."""


class AuditBusyError(RuntimeError):
    """A competing transaction exceeded the configured local lock timeout."""


class DispatchBlockedError(ValueError):
    """This proposal was already claimed or an earlier effect is not reconciled."""


@dataclass(frozen=True)
class AuditCheckpoint:
    tenant: str
    log_id: str
    sequence: int
    chain_hash: str

    def __post_init__(self):
        identifier(self.tenant)
        digest(self.log_id)
        digest(self.chain_hash)
        if type(self.sequence) is not int or not 0 <= self.sequence <= 100000:
            raise ValueError("bounded audit position required")


@dataclass(frozen=True)
class AuditReceipt:
    checkpoint: AuditCheckpoint
    record_digest: str
    committed_ms: int
    authority: Authority


@dataclass(frozen=True)
class AuditEntry:
    record: Episode | TransitionPlan | TransitionOutcome | EpisodeEnd | DispatchClaim | ActionReconciliation
    receipt: AuditReceipt


@dataclass(frozen=True)
class _EpisodeState:
    episode: Episode
    next_step: int
    last_ms: int
    pending: TransitionPlan | None = None
    outcome: TransitionOutcome | None = None
    end: EpisodeEnd | None = None


def _identity(record):
    kinds = {Episode: "episode", TransitionPlan: "plan", TransitionOutcome: "outcome", EpisodeEnd: "end",
             DispatchClaim: "dispatch", ActionReconciliation: "reconciliation"}
    kind = kinds.get(type(record))
    if kind is None:
        raise ValueError("typed experience record required")
    step = getattr(record, "step", None)
    return canonical([record.episode_id, kind, step]), kind


def _record_time(record):
    return next(getattr(record, field) for field in ("started_ms", "proposed_ms", "observed_ms", "ended_ms", "claimed_ms")
                if hasattr(record, field))


class AuditStore:
    """One tenant per local SQLite file, with one connection per process.

    ``authorize(actor, operation, record)`` is a trusted host callback. Operations
    are append, read and audit; record=None requests store-level read/audit access.
    It must also check current access to referenced evidence before returning True.
    """

    def __init__(self, path, *, tenant, signing_key, authorize, create=False, checkpoint=None,
                 clock=None, max_records=10000, max_bytes=32 * 1024 * 1024, timeout=5.):
        identifier(tenant)
        if type(signing_key) is not bytes or len(signing_key) != 32 or not callable(authorize):
            raise ValueError("host-owned 32-byte signing key and authorizer required")
        if type(create) is not bool or type(max_records) is not int or not 1 <= max_records <= 100000:
            raise ValueError("explicit creation and bounded journal size required")
        if type(max_bytes) is not int or not MAX_RECORD_BYTES <= max_bytes <= 1024**3:
            raise ValueError("bounded journal byte capacity required")
        if type(timeout) not in (int, float) or not 0 < timeout <= 30:
            raise ValueError("bounded transaction timeout required")
        target = Path(path).absolute()
        if any(p.is_symlink() for p in (target, *target.parents)) or not target.parent.is_dir():
            raise ValueError("trusted existing local directory and regular journal path required")
        if target.exists() and not target.is_file():
            raise ValueError("regular journal file required")
        if not target.exists() and not create:
            raise FileNotFoundError("audit journal does not exist")
        self.tenant, self.max_records, self.max_bytes = tenant, max_records, max_bytes
        self._key, self._authorize = signing_key, authorize
        self._clock = clock or (lambda: time.time_ns() // 1000000)
        self._lock, self._closed, self._faulted = threading.RLock(), False, False
        self._connection = sqlite3.connect(str(target), timeout=timeout, isolation_level=None, check_same_thread=False)
        try:
            self._connection.execute("PRAGMA trusted_schema=OFF")
            self._connection.execute("PRAGMA journal_mode=WAL")
            self._connection.execute("PRAGMA synchronous=FULL")
            with self._transaction(write=True):
                tables = {r[0] for r in self._connection.execute("SELECT name FROM sqlite_master WHERE type='table'")}
                if not tables and create:
                    self._initialize()
                elif tables != {"metadata", "events"}:
                    raise AuditIntegrityError("unknown audit schema")
                metadata = dict(self._connection.execute("SELECT name, value FROM metadata"))
                if set(metadata) != {"schema", "tenant", "log_id", "genesis"}:
                    raise AuditIntegrityError("invalid audit metadata")
                self.log_id = metadata["log_id"]
                digest(self.log_id)
                self._genesis = self._mac(canonical(["audit-v1", tenant, self.log_id]))
                if (metadata["schema"] != "1" or metadata["tenant"] != tenant
                        or not hmac.compare_digest(metadata["genesis"], self._genesis)):
                    raise AuditIntegrityError("audit identity or signing key mismatch")
                self._reset()
                self._refresh()
                self._check_anchor(checkpoint)
        except Exception:
            self._connection.close()
            self._closed = True
            raise

    def _mac(self, text):
        return hmac.new(self._key, text.encode("utf-8"), hashlib.sha256).hexdigest()

    def _initialize(self):
        log_id = secrets.token_hex(32)
        metadata = {"schema": "1", "tenant": self.tenant, "log_id": log_id,
                    "genesis": self._mac(canonical(["audit-v1", self.tenant, log_id]))}
        self._connection.execute("CREATE TABLE metadata (name TEXT PRIMARY KEY, value TEXT NOT NULL)")
        self._connection.executemany("INSERT INTO metadata VALUES (?, ?)", metadata.items())
        self._connection.execute("""CREATE TABLE events (
            sequence INTEGER PRIMARY KEY, event_key TEXT NOT NULL UNIQUE,
            record_json TEXT NOT NULL CHECK(length(record_json)<=65536),
            record_digest TEXT NOT NULL, authority_json TEXT NOT NULL, committed_ms INTEGER NOT NULL,
            previous_hash TEXT NOT NULL, chain_hash TEXT NOT NULL UNIQUE)""")
        # An operational guard, not a defense against the database owner dropping triggers.
        for table in ("metadata", "events"):
            for operation in ("UPDATE", "DELETE"):
                self._connection.execute(f"CREATE TRIGGER no_{table}_{operation.lower()} BEFORE {operation} ON {table} "
                                         "BEGIN SELECT RAISE(ABORT, 'append-only audit journal'); END")

    @contextmanager
    def _transaction(self, *, write=False):
        if self._closed:
            raise RuntimeError("audit journal is closed")
        if self._faulted:
            raise AuditIntegrityError("audit journal failed integrity verification; reopen against a trusted checkpoint")
        try:
            self._connection.execute("BEGIN IMMEDIATE" if write else "BEGIN")
            yield
            self._connection.execute("COMMIT")
        except Exception as exc:
            if self._connection.in_transaction:
                self._connection.execute("ROLLBACK")
            if isinstance(exc, sqlite3.Error):
                if "locked" in str(exc).lower() or "busy" in str(exc).lower():
                    raise AuditBusyError("audit transaction busy") from None
                self._faulted = True
                raise AuditIntegrityError("unreadable or invalid audit storage") from None
            if isinstance(exc, AuditIntegrityError):
                self._faulted = True
            raise

    def _authorized(self, actor, operation, record=None):
        try:
            allowed = (type(actor) is Authority and actor.tenant == self.tenant
                       and self._authorize(actor, operation, record) is True)
        except Exception:
            allowed = False
        if not allowed:
            raise PermissionError("unavailable audit history")

    def _reset(self):
        self._sequence, self._bytes, self._committed_ms = 0, 0, 0
        self._head = self._genesis
        self._entries, self._episodes, self._histories, self._action_bindings = {}, {}, {}, {}
        self._dispatches, self._effect_attempts, self._reconciliations = {}, {}, {}
        self._active_dispatches = set()

    def _next_state(self, record):
        current = self._episodes.get(record.episode_id)
        if type(record) is Episode:
            if current is not None or record.authority.tenant != self.tenant:
                raise ValueError("episode identity collision or foreign tenant")
            return _EpisodeState(record, 0, record.started_ms)
        if current is None or (current.end is not None and type(record) is not ActionReconciliation):
            raise ValueError("open episode required")
        if _record_time(record) < current.last_ms:
            raise ValueError("episode time cannot move backward")
        if type(record) is DispatchClaim:
            plan = current.pending
            if (plan is None or record.step != plan.step or record.plan_digest != record_digest(plan)
                    or record.action != plan.proposed_action or record.attempt_id in self._dispatches):
                raise ValueError("dispatch requires the exact committed pending plan")
            earlier = self._effect_attempts.get(record.action.idempotency_key)
            if earlier is not None:
                reconciliation = self._reconciliations.get(earlier)
                if reconciliation is None or reconciliation.effect_state != "not_applied":
                    raise DispatchBlockedError("earlier effect requires terminal not-applied reconciliation before retry")
            return replace(current, last_ms=record.claimed_ms)
        if type(record) is ActionReconciliation:
            claim = self._dispatches.get(record.attempt_id)
            if (claim is None or any(getattr(record, key) != getattr(claim, key) for key in
                                    ("episode_id", "step", "plan_digest", "action"))
                    or record.verifier_revision != current.episode.verifier_revision
                    or record.attempt_id in self._reconciliations
                    or canonical([record.episode_id, "outcome", record.step]) not in self._entries):
                raise ValueError("terminal reconciliation requires a matching claim and recorded outcome")
            if any(ref.tenant != self.tenant for ref in record.observations):
                raise ValueError("foreign tenant reconciliation")
            outcome = self._entries[canonical([record.episode_id, "outcome", record.step])].record
            if record.effect_state == "not_applied" and outcome.status == "succeeded":
                raise ValueError("not-applied reconciliation conflicts with verified success")
            return replace(current, last_ms=record.observed_ms)
        if type(record) is TransitionPlan:
            validate_transition(current.episode, record)
            if current.pending is not None or record.step != current.next_step:
                raise ValueError("one sequential unresolved plan per episode")
            bound = self._action_bindings.get(record.proposed_action.idempotency_key)
            if bound is not None and bound != record.proposed_action:
                raise ValueError("idempotency key cannot bind different action inputs")
            return replace(current, pending=record, last_ms=record.proposed_ms)
        if type(record) is TransitionOutcome:
            if current.pending is None:
                raise ValueError("a committed plan must precede its outcome")
            validate_transition(current.episode, current.pending, record)
            claim_entry = self._entries.get(canonical([record.episode_id, "dispatch", record.step]))
            if claim_entry is not None and record.status == "succeeded" and record.observed_ms > claim_entry.record.deadline_ms:
                raise ValueError("late result cannot be promoted to on-time success")
            return replace(current, pending=None, outcome=record, next_step=record.step + 1, last_ms=record.observed_ms)
        if current.pending is not None:
            raise ValueError("unresolved plan requires an explicit outcome before episode closure")
        if record.status == "succeeded" and (current.outcome is None or current.outcome.status != "succeeded"):
            raise ValueError("successful closure requires a verified latest action outcome")
        return replace(current, end=record, last_ms=record.ended_ms)

    def _apply(self, entry, event_key, size, next_state):
        receipt, record = entry.receipt, entry.record
        self._sequence, self._head = receipt.checkpoint.sequence, receipt.checkpoint.chain_hash
        self._committed_ms = receipt.committed_ms
        self._bytes += size
        self._entries[event_key] = entry
        self._histories.setdefault(record.episode_id, []).append(entry)
        self._episodes[record.episode_id] = next_state
        if type(record) is TransitionPlan:
            self._action_bindings[record.proposed_action.idempotency_key] = record.proposed_action
        if type(record) is DispatchClaim:
            self._dispatches[record.attempt_id] = record
            self._active_dispatches.add((record.episode_id, record.step))
            if record.action.effect != "pure":
                self._effect_attempts[record.action.idempotency_key] = record.attempt_id
        if type(record) is TransitionOutcome:
            self._active_dispatches.discard((record.episode_id, record.step))
        if type(record) is ActionReconciliation:
            self._reconciliations[record.attempt_id] = record

    def _chain(self, sequence, event_key, record_digest, actor_text, committed_ms, previous):
        return self._mac(canonical([self.tenant, self.log_id, sequence, event_key, record_digest,
                                   actor_text, committed_ms, previous]))

    def _refresh(self):
        total = self._connection.execute("SELECT count(*), coalesce(max(sequence),0) FROM events").fetchone()
        if total[0] != total[1] or not self._sequence <= total[0] <= self.max_records:
            raise AuditIntegrityError("audit sequence gap, rollback or capacity violation")
        if self._sequence:
            row = self._connection.execute("SELECT chain_hash FROM events WHERE sequence=?", (self._sequence,)).fetchone()
            if row is None or row[0] != self._head:
                raise AuditIntegrityError("previously verified audit head changed")
        rows = self._connection.execute("SELECT * FROM events WHERE sequence>? ORDER BY sequence", (self._sequence,))
        for sequence, event_key, text, fingerprint, actor_text, committed, previous, chain in rows:
            if type(text) is not str or len(text.encode("utf-8")) > MAX_RECORD_BYTES:
                raise AuditIntegrityError("oversized audit entry")
            size = len(text.encode("utf-8"))
            if self._bytes + size > self.max_bytes:
                raise AuditIntegrityError("audit byte capacity exceeded")
            try:
                record = decode_record(text)
                if type(actor_text) is not str or len(actor_text) > 512:
                    raise ValueError("bounded recorded authority required")
                actor = Authority(**json.loads(actor_text))
                timestamp(committed)
                digest(fingerprint)
                digest(chain)
                if (sequence != self._sequence + 1 or event_key != _identity(record)[0]
                        or committed < max(self._committed_ms, _record_time(record))
                        or text != encode_record(record) or previous != self._head
                        or fingerprint != hashlib.sha256(text.encode()).hexdigest()
                        or actor_text != canonical(asdict(actor))
                        or not hmac.compare_digest(chain, self._chain(sequence, event_key, fingerprint, actor_text, committed, previous))):
                    raise ValueError("invalid signed entry")
                state = self._next_state(record)
                if (actor.tenant != self.tenant or actor.principal != state.episode.authority.principal
                        or type(record) is Episode and actor != record.authority):
                    raise ValueError("recorded actor binding mismatch")
            except (ValueError, TypeError, OverflowError, RecursionError):
                raise AuditIntegrityError("invalid audit entry or transition") from None
            entry = AuditEntry(record, AuditReceipt(AuditCheckpoint(self.tenant, self.log_id, sequence, chain),
                                                   fingerprint, committed, actor))
            self._apply(entry, event_key, size, state)

    def _check_anchor(self, checkpoint):
        if checkpoint is None:
            return
        if (type(checkpoint) is not AuditCheckpoint or checkpoint.tenant != self.tenant
                or checkpoint.log_id != self.log_id or checkpoint.sequence > self._sequence):
            raise AuditIntegrityError("external checkpoint does not match this journal")
        expected = self._genesis if checkpoint.sequence == 0 else self._connection.execute(
            "SELECT chain_hash FROM events WHERE sequence=?", (checkpoint.sequence,)).fetchone()[0]
        if not hmac.compare_digest(expected, checkpoint.chain_hash):
            raise AuditIntegrityError("external checkpoint chain mismatch")

    def append(self, record, *, authority):
        if type(record) in (DispatchClaim, ActionReconciliation):
            raise ValueError("use claim_dispatch or reconcile for execution control records")
        return self._append(record, authority=authority)

    def _append(self, record, *, authority, related=None, exclusive=False):
        event_key, _ = _identity(record)
        text = encode_record(record)
        size = len(text.encode("utf-8"))
        fingerprint = hashlib.sha256(text.encode()).hexdigest()
        self._authorized(authority, "append", record)
        with self._lock:
            with self._transaction(write=True):
                self._refresh()
                if related is not None:
                    self._authorized(authority, "append", related)
                existing = self._entries.get(event_key)
                episode = record if type(record) is Episode else self._episodes.get(record.episode_id)
                if isinstance(episode, _EpisodeState):
                    episode = episode.episode
                if episode is not None and authority.principal != episode.authority.principal:
                    raise PermissionError("unavailable audit history")
                if existing is None and type(record) is Episode and authority != record.authority:
                    raise PermissionError("episode must bind its authenticated creation authority")
                if existing is not None:
                    if exclusive:
                        raise DispatchBlockedError("proposal already has a durable dispatch claim")
                    if existing.receipt.record_digest != fingerprint:
                        raise ValueError("audit event collision; history cannot be replaced")
                    self._authorized(authority, "append", record)
                    return existing.receipt
                reserved = len(self._active_dispatches)
                if type(record) is DispatchClaim:
                    reserved += 1
                elif type(record) is TransitionOutcome and (record.episode_id, record.step) in self._active_dispatches:
                    reserved -= 1
                if (self._sequence + 1 + reserved > self.max_records
                        or self._bytes + size + reserved * MAX_RECORD_BYTES > self.max_bytes):
                    raise OverflowError("audit capacity reached; action must not start")
                state = self._next_state(record)
                now = self._clock()
                timestamp(now)
                now = max(now, self._committed_ms)
                if _record_time(record) > now:
                    raise ValueError("future-dated audit event")
                if type(record) is TransitionOutcome and record.status == "succeeded":
                    dispatch = self._entries.get(canonical([record.episode_id, "dispatch", record.step]))
                    if dispatch is not None and now > dispatch.record.deadline_ms:
                        raise ValueError("success commit exceeded dispatch deadline")
                actor_text = canonical(asdict(authority))
                chain = self._chain(self._sequence + 1, event_key, fingerprint, actor_text, now, self._head)
                entry = AuditEntry(record, AuditReceipt(AuditCheckpoint(self.tenant, self.log_id, self._sequence + 1, chain),
                                                       fingerprint, now, authority))
                self._connection.execute("INSERT INTO events VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    (self._sequence + 1, event_key, text, fingerprint, actor_text, now, self._head, chain))
                self._authorized(authority, "append", record)
                if related is not None:
                    self._authorized(authority, "append", related)
            self._apply(entry, event_key, size, state)
            return entry.receipt

    def claim_dispatch(self, plan, *, authority, deadline_ms):
        """Atomically claim once; only a newly committed claim authorizes launch.

        A failed caller response or crash never permits replaying this launch.
        Hosts must retain the same journal and share it across their workers.
        """
        if type(plan) is not TransitionPlan:
            raise ValueError("typed committed proposal required")
        claim = DispatchClaim(plan.episode_id, plan.step, secrets.token_hex(32), record_digest(plan),
                              plan.proposed_action, self._clock(), deadline_ms)
        receipt = self._append(claim, authority=authority, related=plan, exclusive=True)
        return AuditEntry(claim, receipt)

    def check_dispatch(self, claim, *, authority):
        """Check the recorded claim, unresolved plan and current host permissions."""
        if type(claim) is not DispatchClaim:
            raise ValueError("typed claim required")
        self._authorized(authority, "read", claim)
        with self._lock, self._transaction():
            self._refresh()
            entry = self._entries.get(_identity(claim)[0])
            state = self._episodes.get(claim.episode_id)
            if entry is None or entry.record != claim or state.pending is None or state.pending.step != claim.step:
                raise DispatchBlockedError("dispatch no longer pending")
            self._authorized(authority, "append", state.pending)
            self._authorized(authority, "append", claim)
            if authority.principal != state.episode.authority.principal:
                raise PermissionError("unavailable dispatch")
        return True

    def reconcile(self, record, *, authority, verify):
        """Host verifies a provider's terminal status, including absence of late effects.

        For not_applied, a temporary 'not found' response is insufficient: the
        provider must guarantee that the earlier attempt can no longer take effect.
        """
        if type(record) is not ActionReconciliation or not callable(verify):
            raise ValueError("typed reconciliation and host verifier required")
        self._authorized(authority, "append", record)
        try:
            accepted = verify(record) is True
        except Exception:
            accepted = False
        if not accepted:
            raise ValueError("provider reconciliation was not verified")
        return self._append(record, authority=authority)

    def history(self, episode_id, *, authority):
        identifier(episode_id)
        self._authorized(authority, "read")
        with self._lock, self._transaction():
            self._refresh()
            result = tuple(self._histories.get(episode_id, ()))
            for entry in result:
                self._authorized(authority, "read", entry.record)
            self._authorized(authority, "read")
            return result

    def checkpoint(self, *, authority):
        self._authorized(authority, "audit")
        with self._lock, self._transaction():
            self._refresh()
            self._authorized(authority, "audit")
            return AuditCheckpoint(self.tenant, self.log_id, self._sequence, self._head)

    def verify(self, *, authority, checkpoint=None):
        """Full disk replay; a separately retained checkpoint detects prefix rollback."""
        self._authorized(authority, "audit")
        with self._lock, self._transaction():
            prior = AuditCheckpoint(self.tenant, self.log_id, self._sequence, self._head)
            self._reset()
            self._refresh()
            self._check_anchor(prior)
            self._check_anchor(checkpoint)
            self._authorized(authority, "audit")
            return AuditCheckpoint(self.tenant, self.log_id, self._sequence, self._head)

    def close(self):
        with self._lock:
            if not self._closed:
                self._connection.close()
                self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

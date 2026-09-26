"""Durable temporal context with a metadata catalogue and cold payloads.

Copyright (c) 2026 Prashant Jagtap. MIT License.
Host-owned private SQLite storage; no authentication, encryption or data deletion
service. One connection per process; exact immutable versions and current ACLs.
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
from dataclasses import asdict, dataclass
from pathlib import Path

from .audit import AuditBusyError, AuditIntegrityError
from .context_state import (
    AccessScope,
    CanonicalNode,
    ContextClaim,
    ContextSnapshot,
    NodeRef,
    StoredNode,
    TemporalScope,
    canonical,
    digest,
    timestamp,
    typed_tuple,
    visible_records,
)
from .experience import _unique_object
from .security import bounded_text, identifier, prepare_database

MAX_EVENT_BYTES = 512 * 1024


@dataclass(frozen=True)
class StateCheckpoint:
    tenant: str
    store_id: str
    sequence: int
    chain_hash: str

    def __post_init__(self):
        identifier(self.tenant)
        digest(self.store_id)
        digest(self.chain_hash)
        if type(self.sequence) is not int or not 0 <= self.sequence <= 100000:
            raise ValueError("bounded state position required")


@dataclass(frozen=True)
class StateReceipt:
    checkpoint: StateCheckpoint
    mutation_id: str
    operation: str
    command_digest: str
    scope: AccessScope
    committed_ms: int


@dataclass(frozen=True)
class ContextEntry:
    ref: NodeRef
    transaction_time: int
    payload_bytes: int
    kind: str
    dependencies: tuple[NodeRef, ...]


@dataclass(frozen=True)
class ContextInventory:
    scope: AccessScope
    at: int
    known_at: int
    entries: tuple[ContextEntry, ...]
    binding: str


@dataclass(frozen=True)
class SourceLineage:
    ref: NodeRef
    transaction_time: int
    supersedes: tuple[NodeRef, ...]
    superseded_by: tuple[NodeRef, ...]
    currently_invalidated: bool


@dataclass(frozen=True)
class _Header:
    ref: NodeRef
    key: str
    revision: str
    temporal: TemporalScope
    dependencies: tuple[NodeRef, ...]
    supersedes: tuple[NodeRef, ...]
    lifecycle: str
    kind: str
    event_sequence: int
    payload_bytes: int


def _node(data):
    if type(data) is not dict or set(data) != set(CanonicalNode.__dataclass_fields__):
        raise ValueError("exact canonical node fields required")
    data = dict(data)
    data["temporal"] = TemporalScope(**data["temporal"])
    for name, cls, bound in (("roles", str, 32), ("claims", ContextClaim, 32),
                              ("dependencies", NodeRef, 128), ("supersedes", NodeRef, 32)):
        if type(data[name]) is not list or len(data[name]) > bound:
            raise ValueError("bounded node array required")
        data[name] = tuple(v if cls is str else cls(**v) for v in data[name])
    return CanonicalNode(**data)


class ContextStore:
    """Replay metadata eagerly, materialize authorized payloads on demand.

    authorize(scope, operation, subject) is a current trusted host check. Read
    subjects are None (store access) or NodeRef; write subjects are typed nodes,
    references, (source, roles), or policy revisions. It must authenticate scope
    and enforce who may assign authority, ACLs, provenance and validation labels.
    """

    def __init__(self, path, *, tenant, signing_key, authorize, create=False, policy_revision=None,
                 checkpoint=None, capacity=2048, max_events=10000, max_bytes=64 * 1024 * 1024,
                 materialize_bytes=16 * 1024 * 1024, clock=None, timeout=5.):
        identifier(tenant)
        if type(signing_key) is not bytes or len(signing_key) != 32 or not callable(authorize):
            raise ValueError("host key and authorizer required")
        for value, minimum, maximum in ((capacity, 1, 10000), (max_events, 1, 100000),
                (max_bytes, MAX_EVENT_BYTES, 1024**3), (materialize_bytes, 1, 256 * 1024 * 1024)):
            if type(value) is not int or not minimum <= value <= maximum:
                raise ValueError("bounded store capacity required")
        if type(create) is not bool or type(timeout) not in (float, int) or not 0 < timeout <= 30:
            raise ValueError("explicit creation and bounded lock timeout required")
        target = Path(path).absolute()
        if not target.parent.is_dir() or any(p.is_symlink() for p in (target, *target.parents)):
            raise ValueError("trusted local directory required")
        if not target.exists() and not create:
            raise FileNotFoundError("context store does not exist")
        prepare_database(target)
        self.tenant, self.capacity = tenant, capacity
        self.max_events, self.max_bytes, self.materialize_bytes = max_events, max_bytes, materialize_bytes
        self._key, self._authorize, self._clock = signing_key, authorize, clock or (lambda: time.time_ns() // 1000000)
        self._lock, self._closed, self._faulted = threading.RLock(), False, False
        self._db = sqlite3.connect(str(target), isolation_level=None, timeout=timeout, check_same_thread=False)
        try:
            self._db.execute("PRAGMA trusted_schema=OFF")
            self._db.execute("PRAGMA journal_mode=WAL")
            self._db.execute("PRAGMA synchronous=FULL")
            with self._transaction(write=True):
                tables = {r[0] for r in self._db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
                if not tables and create:
                    identifier(policy_revision)
                    store_id = secrets.token_hex(32)
                    metadata = {"schema": "1", "tenant": tenant, "store_id": store_id, "initial_policy": policy_revision}
                    metadata["genesis"] = self._mac(canonical(metadata))
                    self._db.execute("CREATE TABLE metadata (name TEXT PRIMARY KEY, value TEXT NOT NULL)")
                    self._db.executemany("INSERT INTO metadata VALUES (?, ?)", metadata.items())
                    self._db.execute("""CREATE TABLE mutations (
                        sequence INTEGER PRIMARY KEY, mutation_id TEXT NOT NULL UNIQUE,
                        operation TEXT NOT NULL, payload TEXT NOT NULL, actor TEXT NOT NULL,
                        committed INTEGER NOT NULL, previous TEXT NOT NULL, chain TEXT NOT NULL UNIQUE)""")
                elif tables != {"metadata", "mutations"}:
                    raise AuditIntegrityError("unknown context storage schema")
                metadata = dict(self._db.execute("SELECT name, value FROM metadata"))
                if set(metadata) != {"schema", "tenant", "store_id", "initial_policy", "genesis"}:
                    raise AuditIntegrityError("invalid context metadata")
                self.store_id, self._initial_policy = metadata["store_id"], metadata["initial_policy"]
                digest(self.store_id)
                identifier(self._initial_policy)
                self._genesis = metadata.pop("genesis")
                if (metadata["schema"] != "1" or metadata["tenant"] != tenant
                        or not hmac.compare_digest(self._genesis, self._mac(canonical(metadata)))
                        or policy_revision is not None and policy_revision != self._initial_policy):
                    raise AuditIntegrityError("context identity, policy or key mismatch")
                self._reset()
                self._refresh()
                self._anchor(checkpoint)
        except Exception:
            self._db.close()
            self._closed = True
            raise

    def _mac(self, material):
        return hmac.new(self._key, material.encode(), hashlib.sha256).hexdigest()

    @contextmanager
    def _transaction(self, *, write=False):
        if self._closed:
            raise RuntimeError("context store closed")
        if self._faulted:
            raise AuditIntegrityError("context store failed integrity verification; reopen against trusted checkpoint")
        try:
            self._db.execute("BEGIN IMMEDIATE" if write else "BEGIN")
            yield
            self._db.execute("COMMIT")
        except Exception as error:
            if self._db.in_transaction:
                self._db.execute("ROLLBACK")
            if isinstance(error, sqlite3.Error):
                if "locked" in str(error).lower() or "busy" in str(error).lower():
                    raise AuditBusyError("context transaction busy") from None
                self._faulted = True
                raise AuditIntegrityError("invalid context storage") from None
            if isinstance(error, AuditIntegrityError):
                self._faulted = True
            raise

    def _reset(self):
        self._sequence, self._bytes, self._last_time = 0, 0, 0
        self._head, self._policy = self._genesis, self._initial_policy
        self._catalog, self._acl, self._revoked, self._receipts = {}, {}, set(), {}

    def _allowed(self, scope, operation, subject=None):
        try:
            return (type(scope) is AccessScope and scope.tenant == self.tenant and scope.policy_revision == self._policy
                    and self._authorize(scope, operation, subject) is True)
        except Exception:
            return False

    def _authorized(self, scope, operation, subject=None):
        if not self._allowed(scope, operation, subject):
            raise PermissionError("unavailable context")

    def _decode(self, operation, payload, committed, *, retry=False):
        if type(payload) is not str or len(payload.encode()) > MAX_EVENT_BYTES:
            raise ValueError("bounded context event required")
        data = json.loads(payload, object_pairs_hook=_unique_object)
        if canonical(data) != payload:
            raise ValueError("canonical event required")
        if operation == "put":
            value = _node(data)
            if value.tenant != self.tenant or value.temporal.observed_at > committed:
                raise ValueError("same tenant and nonfuture observation required")
            existing = self._catalog.get((value.key, value.revision))
            if existing is not None and (not retry or existing.node.ref != value.ref):
                raise ValueError("revision already exists; retry the original mutation ID")
            if existing is None and len(self._catalog) >= self.capacity:
                raise OverflowError("context node capacity reached")
            for ref in value.dependencies + value.supersedes:
                target = self._catalog.get((ref.key, ref.revision))
                if target is None or target.node.ref != ref:
                    raise ValueError("exact previously ingested reference required")
            if any(ref.key != value.key for ref in value.supersedes):
                raise ValueError("supersession stays within a logical source")
        elif operation == "roles":
            if type(data) is not dict or set(data) != {"source", "roles"} or type(data["roles"]) is not list:
                raise ValueError("exact role update fields required")
            identifier(data["source"])
            roles = tuple(data["roles"])
            typed_tuple(roles, str, 32)
            for role in roles:
                identifier(role)
            if data["source"] not in self._acl:
                raise ValueError("unknown source")
            value = (data["source"], tuple(sorted(roles)))
        elif operation == "policy":
            if type(data) is not dict or set(data) != {"revision"}:
                raise ValueError("exact policy fields required")
            identifier(data["revision"])
            value = data["revision"]
        elif operation == "invalidate":
            value = NodeRef(**data)
            existing = self._catalog.get((value.key, value.revision))
            if existing is None or existing.node.ref != value:
                raise ValueError("unknown exact reference")
        else:
            raise ValueError("unsupported context operation")
        return value

    def _install(self, receipt, value, payload_bytes):
        op, sequence, committed = receipt.operation, receipt.checkpoint.sequence, receipt.committed_ms
        if op == "put":
            header = _Header(value.ref, value.key, value.revision, value.temporal, value.dependencies,
                             value.supersedes, value.lifecycle, value.kind, sequence, payload_bytes)
            self._catalog[value.key, value.revision] = StoredNode(header, committed)
            self._acl.setdefault(value.key, value.roles)
        elif op == "roles":
            self._acl[value[0]] = value[1]
        elif op == "policy":
            self._policy = value
        else:
            self._revoked.add(value)
        self._sequence, self._head, self._last_time = sequence, receipt.checkpoint.chain_hash, committed
        self._bytes += payload_bytes
        self._receipts[receipt.mutation_id] = receipt

    def _chain(self, sequence, mutation_id, operation, payload, actor, committed, previous):
        return self._mac(canonical(["state-v1", self.tenant, self.store_id, sequence, mutation_id, operation,
                                    hashlib.sha256(payload.encode()).hexdigest(), actor, committed, previous]))

    def _checked_row(self, row):
        sequence, mutation_id, operation, payload, actor, committed, previous, chain = row
        try:
            identifier(mutation_id)
            timestamp(committed)
            digest(previous)
            digest(chain)
            if type(actor) is not str or len(actor.encode()) > 32768:
                raise ValueError("bounded actor required")
            data = json.loads(actor, object_pairs_hook=_unique_object)
            if type(data["roles"]) is not list:
                raise ValueError("actor roles required")
            scope = AccessScope(**{**data, "roles": tuple(data["roles"])})
            if (actor != canonical(asdict(scope)) or scope.tenant != self.tenant
                    or type(payload) is not str or len(payload.encode()) > MAX_EVENT_BYTES
                    or not hmac.compare_digest(chain, self._chain(*row[:-1]))):
                raise ValueError("invalid context signature")
            return StateReceipt(StateCheckpoint(self.tenant, self.store_id, sequence, chain), mutation_id, operation,
                                hashlib.sha256(canonical([operation, payload]).encode()).hexdigest(), scope, committed)
        except (ValueError, TypeError, KeyError, RecursionError, OverflowError):
            raise AuditIntegrityError("invalid signed context mutation") from None

    def _refresh(self):
        count, maximum = self._db.execute("SELECT count(*), coalesce(max(sequence),0) FROM mutations").fetchone()
        if count != maximum or not self._sequence <= count <= self.max_events:
            raise AuditIntegrityError("context sequence gap, rollback or capacity violation")
        if self._sequence:
            prior = self._db.execute("SELECT chain FROM mutations WHERE sequence=?", (self._sequence,)).fetchone()
            if prior is None or prior[0] != self._head:
                raise AuditIntegrityError("previously verified state changed")
        for sequence, size, actor_size in self._db.execute(
                "SELECT sequence,length(CAST(payload AS BLOB)),length(CAST(actor AS BLOB)) FROM mutations WHERE sequence>? ORDER BY sequence",
                (self._sequence,)):
            if size > MAX_EVENT_BYTES or actor_size > 32768 or self._bytes + size > self.max_bytes:
                raise AuditIntegrityError("context storage byte limit")
            row = self._db.execute("SELECT * FROM mutations WHERE sequence=?", (sequence,)).fetchone()
            receipt = self._checked_row(row)
            if row[6] != self._head or receipt.committed_ms < self._last_time or receipt.scope.policy_revision != self._policy:
                raise AuditIntegrityError("invalid context transition order")
            try:
                value = self._decode(receipt.operation, row[3], receipt.committed_ms)
            except (ValueError, TypeError, KeyError, RecursionError, OverflowError):
                raise AuditIntegrityError("invalid context mutation") from None
            self._install(receipt, value, size)

    def _row(self, sequence):
        lengths = self._db.execute(
            "SELECT length(CAST(payload AS BLOB)), length(CAST(actor AS BLOB)) FROM mutations WHERE sequence=?",
            (sequence,)).fetchone()
        if lengths is None or lengths[0] > MAX_EVENT_BYTES or lengths[1] > 32768:
            raise AuditIntegrityError("missing or oversized cold payload")
        return self._db.execute("SELECT * FROM mutations WHERE sequence=?", (sequence,)).fetchone()

    def mutate(self, operation, payload, *, scope, mutation_id):
        """Internal-format command entry point; prefer the typed convenience methods."""
        identifier(mutation_id)
        text = canonical(payload)
        if len(text.encode()) > MAX_EVENT_BYTES:
            raise ValueError("context event byte limit")
        with self._lock:
            with self._transaction(write=True):
                self._refresh()
                self._authorized(scope, operation)
                fingerprint = hashlib.sha256(canonical([operation, text]).encode()).hexdigest()
                existing = self._receipts.get(mutation_id)
                if existing is not None:
                    if existing.command_digest != fingerprint or existing.scope.principal != scope.principal:
                        raise ValueError("mutation ID collision")
                    value = self._decode(operation, text, existing.committed_ms, retry=True)
                    self._authorized(scope, operation, value)
                    return existing
                if self._sequence >= self.max_events or self._bytes + len(text.encode()) > self.max_bytes:
                    raise OverflowError("context store capacity reached")
                committed = max(self._clock(), self._last_time)
                timestamp(committed)
                value = self._decode(operation, text, committed)
                self._authorized(scope, operation, value)
                actor = canonical(asdict(scope))
                row = (self._sequence + 1, mutation_id, operation, text, actor, committed, self._head)
                chain = self._chain(*row)
                receipt = self._checked_row((*row, chain))
                self._db.execute("INSERT INTO mutations VALUES (?,?,?,?,?,?,?,?)", (*row, chain))
                self._authorized(scope, operation, value)
            self._install(receipt, value, len(text.encode()))
            return receipt

    def put(self, node, *, scope, mutation_id):
        if type(node) is not CanonicalNode:
            raise ValueError("typed canonical node required")
        return self.mutate("put", asdict(node), scope=scope, mutation_id=mutation_id)

    def set_roles(self, source, roles, *, scope, mutation_id):
        typed_tuple(roles, str, 32)
        return self.mutate("roles", {"source": source, "roles": list(roles)}, scope=scope, mutation_id=mutation_id)

    def set_policy(self, revision, *, scope, mutation_id):
        return self.mutate("policy", {"revision": revision}, scope=scope, mutation_id=mutation_id)

    def invalidate(self, ref, *, scope, mutation_id):
        if type(ref) is not NodeRef:
            raise ValueError("exact reference required")
        return self.mutate("invalidate", asdict(ref), scope=scope, mutation_id=mutation_id)

    def _visible(self, scope, at, known_at):
        self._authorized(scope, "read")
        initial = visible_records(self._catalog.values(), self._acl, self._revoked, scope, at=at, known_at=known_at)
        denied = {r.node.ref for r in initial if not self._allowed(scope, "read", r.node.ref)}
        return visible_records(self._catalog.values(), self._acl, self._revoked | denied, scope, at=at, known_at=known_at)

    def _signature(self, scope, at, known_at, records):
        return self._mac(canonical(["snapshot-v1", self.store_id, self._head, asdict(scope), at, known_at,
                                    [(asdict(r.node.ref), r.transaction_time) for r in records]]))

    def _inventory_signature(self, scope, at, known_at, entries):
        return self._mac(canonical(["inventory-v1", self.store_id, self._head, asdict(scope), at, known_at,
                                    [asdict(entry) for entry in entries]]))

    def inventory(self, scope, *, at, known_at, roots=None):
        """Authorized metadata and dependency closure; no document bodies returned."""
        timestamp(at)
        timestamp(known_at)
        if roots is not None:
            typed_tuple(roots, NodeRef, self.capacity)
        with self._lock, self._transaction():
            self._refresh()
            headers = self._visible(scope, at, known_at)
            if roots is not None:
                available = {r.node.ref: r for r in headers}
                selected, pending = set(), list(roots)
                while pending:
                    ref = pending.pop()
                    if ref not in available:
                        raise PermissionError("unavailable context")
                    if ref not in selected:
                        selected.add(ref)
                        pending.extend(available[ref].node.dependencies)
                headers = tuple(r for r in headers if r.node.ref in selected)
            entries = tuple(ContextEntry(r.node.ref, r.transaction_time, r.node.payload_bytes,
                                         r.node.kind, r.node.dependencies) for r in headers)
            self._authorized(scope, "read")
            if any(not self._allowed(scope, "read", r.ref) for r in entries):
                raise PermissionError("unavailable context")
            return ContextInventory(scope, at, known_at, entries, self._inventory_signature(scope, at, known_at, entries))

    def materialize(self, inventory, *, cached=()):
        """Read cold payloads missing from a host cache; recheck its exact identities."""
        if type(inventory) is not ContextInventory:
            raise ValueError("typed inventory required")
        typed_tuple(cached, StoredNode, self.capacity)
        with self._lock, self._transaction():
            self._refresh()
            scope, at, known_at, entries = inventory.scope, inventory.at, inventory.known_at, inventory.entries
            self._authorized(scope, "read")
            if (len(entries) > self.capacity or not hmac.compare_digest(inventory.binding,
                    self._inventory_signature(scope, at, known_at, entries))):
                raise ValueError("state changed; acquire a new inventory")
            if any(not self._allowed(scope, "read", entry.ref) for entry in entries):
                raise PermissionError("unavailable context")
            if sum(entry.payload_bytes for entry in entries) > self.materialize_bytes:
                raise OverflowError("materialized context budget exceeded; select bounded roots")
            available = {r.node.ref: r for r in cached if type(r.node) is CanonicalNode}
            records = []
            for entry in entries:
                if entry.ref in available:
                    stored = available[entry.ref]
                    if stored.transaction_time != entry.transaction_time:
                        raise ValueError("cached transaction time mismatch")
                    records.append(stored)
                    continue
                header = self._catalog[entry.ref.key, entry.ref.revision]
                row = self._row(header.node.event_sequence)
                receipt = self._checked_row(row)
                if receipt != self._receipts[receipt.mutation_id] or receipt.operation != "put":
                    raise AuditIntegrityError("cold payload changed")
                try:
                    node = _node(json.loads(row[3], object_pairs_hook=_unique_object))
                except (ValueError, TypeError, KeyError, RecursionError, OverflowError):
                    raise AuditIntegrityError("invalid cold payload") from None
                if node.ref != header.node.ref:
                    raise AuditIntegrityError("cold payload identity changed")
                records.append(StoredNode(node, header.transaction_time))
            self._authorized(scope, "read")
            if any(not self._allowed(scope, "read", r.node.ref) for r in records):
                raise PermissionError("unavailable context")
            records = tuple(records)
            return ContextSnapshot(scope, at, known_at, records, self._signature(scope, at, known_at, records))

    def snapshot(self, scope, *, at, known_at, roots=None):
        return self.materialize(self.inventory(scope, at=at, known_at=known_at, roots=roots))

    def lineage(self, ref, *, scope, known_at):
        """Inspect authorized historical supersession; never grants decision eligibility."""
        if type(ref) is not NodeRef:
            raise ValueError("exact reference required")
        timestamp(known_at)
        with self._lock, self._transaction():
            self._refresh()
            self._authorized(scope, "history", ref)
            stored = self._catalog.get((ref.key, ref.revision))
            if (stored is None or stored.node.ref != ref or stored.transaction_time > known_at
                    or stored.node.temporal.observed_at > known_at
                    or not set(scope.roles).intersection(self._acl[ref.key])):
                raise PermissionError("unavailable context")
            self._authorized(scope, "read", ref)
            before = stored.node.supersedes
            after = tuple(r.node.ref for r in self._catalog.values()
                          if ref in r.node.supersedes and r.transaction_time <= known_at and r.node.temporal.observed_at <= known_at)
            if any(not self._allowed(scope, "read", r) for r in before + after):
                raise PermissionError("unavailable context")
            self._authorized(scope, "history", ref)
            return SourceLineage(ref, stored.transaction_time, before, after, ref in self._revoked)

    def _current(self, snapshot):
        if type(snapshot) is not ContextSnapshot or not self._allowed(snapshot.scope, "read"):
            return False
        try:
            return (len(snapshot.records) <= self.capacity
                    and all(self._allowed(snapshot.scope, "read", r.node.ref) for r in snapshot.records)
                    and hmac.compare_digest(snapshot.state_revision,
                                            self._signature(snapshot.scope, snapshot.at, snapshot.known_at, snapshot.records)))
        except (TypeError, ValueError, AttributeError):
            return False

    def is_current(self, snapshot):
        with self._lock, self._transaction():
            self._refresh()
            return self._current(snapshot)

    def subset(self, snapshot, refs):
        typed_tuple(refs, NodeRef, self.capacity)
        with self._lock, self._transaction():
            self._refresh()
            if not self._current(snapshot):
                raise ValueError("state changed")
            available = {r.node.ref: r for r in snapshot.records}
            if not set(refs) <= available.keys():
                raise ValueError("selection outside authorized snapshot")
            records = tuple(r for r in snapshot.records if r.node.ref in refs)
            return ContextSnapshot(snapshot.scope, snapshot.at, snapshot.known_at, records,
                                   self._signature(snapshot.scope, snapshot.at, snapshot.known_at, records))

    def seal(self, snapshot, payload):
        bounded_text(payload, 1048576)
        with self._lock, self._transaction():
            self._refresh()
            if not self._current(snapshot):
                raise ValueError("state changed")
            return self._mac(canonical(["packet-v1", self.store_id, snapshot.state_revision, payload]))

    def verify_binding(self, snapshot, payload, seal):
        try:
            return type(seal) is str and hmac.compare_digest(self.seal(snapshot, payload), seal)
        except (ValueError, TypeError, PermissionError):
            return False

    def _anchor(self, checkpoint):
        if checkpoint is None:
            return
        if (type(checkpoint) is not StateCheckpoint or checkpoint.tenant != self.tenant
                or checkpoint.store_id != self.store_id or checkpoint.sequence > self._sequence):
            raise AuditIntegrityError("state checkpoint identity mismatch")
        expected = self._genesis if checkpoint.sequence == 0 else self._db.execute(
            "SELECT chain FROM mutations WHERE sequence=?", (checkpoint.sequence,)).fetchone()[0]
        if not hmac.compare_digest(expected, checkpoint.chain_hash):
            raise AuditIntegrityError("state checkpoint mismatch")

    def checkpoint(self, *, scope, verify=False):
        with self._lock, self._transaction():
            self._refresh()
            self._authorized(scope, "audit")
            prior = StateCheckpoint(self.tenant, self.store_id, self._sequence, self._head)
            if verify:
                self._reset()
                self._refresh()
                self._anchor(prior)
            self._authorized(scope, "audit")
            return prior

    def close(self):
        with self._lock:
            if not self._closed:
                self._db.close()
                self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

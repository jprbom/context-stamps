"""Bounded, host-owned temporal evidence with explicit epistemic semantics.

Copyright (c) 2026 Prashant Jagtap. MIT License.
An in-process state owner, not an authentication server or a truth oracle.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import math
import secrets
import threading
import time
from dataclasses import asdict, dataclass
from functools import cached_property

from .security import bounded_text, identifier

KINDS = frozenset({"FACT", "OBSERVATION", "CLAIM", "INFERENCE", "POLICY", "ASSUMPTION",
                   "PREDICTION", "USER_ASSERTION", "MODEL_OUTPUT", "DERIVED_RESULT"})
STATUSES = frozenset({"PRESENT", "UNKNOWN", "KNOWN_ABSENT", "NOT_APPLICABLE", "REDACTED", "UNAUTHORIZED", "STALE"})
LIFECYCLES = frozenset({"HOT", "WARM", "COLD", "SUPERSEDED", "INVALID", "ARCHIVED"})
MODALITIES = frozenset({"text", "image", "audio", "video", "table", "code"})
VERIFIABLE_KINDS = frozenset({"FACT", "OBSERVATION", "POLICY", "DERIVED_RESULT"})


def timestamp(value):
    if type(value) is not int or not 0 <= value <= 253402300799999:
        raise ValueError("bounded UTC epoch milliseconds required")


def category(value, choices):
    if not isinstance(value, str) or value not in choices:
        raise ValueError("unknown category")


def digest(value):
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError("SHA-256 hexadecimal digest required")


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def typed_tuple(values, cls, maximum):
    if type(values) is not tuple or len(values) > maximum or any(type(v) is not cls for v in values):
        raise ValueError("bounded immutable typed tuple required")
    if len(set(values)) != len(values):
        raise ValueError("duplicate entries")


@dataclass(frozen=True)
class AccessScope:
    tenant: str
    principal: str
    policy_revision: str
    roles: tuple[str, ...]

    def __post_init__(self):
        for value in (self.tenant, self.principal, self.policy_revision):
            identifier(value)
        typed_tuple(self.roles, str, 32)
        if not self.roles:
            raise ValueError("at least one authenticated role required")
        for role in self.roles:
            identifier(role)
        object.__setattr__(self, "roles", tuple(sorted(self.roles)))


@dataclass(frozen=True)
class TemporalScope:
    valid_from: int
    observed_at: int
    valid_until: int | None = None
    effective_at: int | None = None
    event_time: int | None = None

    def __post_init__(self):
        timestamp(self.valid_from)
        timestamp(self.observed_at)
        if self.effective_at is None:
            object.__setattr__(self, "effective_at", self.valid_from)
        if self.event_time is None:
            object.__setattr__(self, "event_time", self.observed_at)
        timestamp(self.effective_at)
        timestamp(self.event_time)
        if self.valid_until is not None:
            timestamp(self.valid_until)
            if self.valid_until <= self.start:
                raise ValueError("validity interval must be nonempty")

    @property
    def start(self):
        return max(self.valid_from, self.effective_at)

    def applies(self, at):
        return self.start <= at and (self.valid_until is None or at < self.valid_until)


@dataclass(frozen=True)
class NodeRef:
    key: str
    revision: str
    fingerprint: str

    def __post_init__(self):
        identifier(self.key)
        identifier(self.revision)
        digest(self.fingerprint)


@dataclass(frozen=True)
class ContextClaim:
    name: str
    value: str | None
    status: str = "PRESENT"

    def __post_init__(self):
        identifier(self.name)
        category(self.status, STATUSES)
        if self.status == "PRESENT":
            if not isinstance(self.value, str):
                raise ValueError("present claim requires a value")
            bounded_text(self.value, 4096)
        elif self.value is not None:
            raise ValueError("negative/unknown states cannot contain an asserted value")


@dataclass(frozen=True)
class CanonicalNode:
    key: str
    revision: str
    tenant: str
    text: str
    kind: str
    temporal: TemporalScope
    roles: tuple[str, ...]
    provenance: str
    claims: tuple[ContextClaim, ...] = ()
    dependencies: tuple[NodeRef, ...] = ()
    supersedes: tuple[NodeRef, ...] = ()
    validation_revision: str | None = None
    modality: str = "text"
    artifact_digest: str | None = None
    stamp_hex: str | None = None
    stamp_schema: str | None = None
    authority: int = 0
    reliability: float | None = None
    lifecycle: str = "COLD"

    def __post_init__(self):
        for value in (self.key, self.revision, self.tenant, self.provenance):
            identifier(value)
        bounded_text(self.text)
        category(self.kind, KINDS)
        category(self.modality, MODALITIES)
        category(self.lifecycle, LIFECYCLES)
        if type(self.temporal) is not TemporalScope:
            raise ValueError("typed temporal scope required")
        typed_tuple(self.roles, str, 32)
        if not self.roles:
            raise ValueError("nonempty initial ACL required")
        for role in self.roles:
            identifier(role)
        object.__setattr__(self, "roles", tuple(sorted(self.roles)))
        typed_tuple(self.claims, ContextClaim, 32)
        if len({c.name for c in self.claims}) != len(self.claims):
            raise ValueError("a node cannot assert two values for one claim")
        typed_tuple(self.dependencies, NodeRef, 128)
        typed_tuple(self.supersedes, NodeRef, 32)
        if self.validation_revision is not None:
            identifier(self.validation_revision)
        if self.modality in ("image", "audio", "video") and self.artifact_digest is None:
            raise ValueError("external media digest required")
        if self.artifact_digest is not None:
            digest(self.artifact_digest)
        if (self.stamp_hex is None) != (self.stamp_schema is None):
            raise ValueError("stamp and schema must be supplied together")
        if self.stamp_hex is not None:
            digest(self.stamp_hex)
            identifier(self.stamp_schema)
        if type(self.authority) is not int or not 0 <= self.authority <= 100:
            raise ValueError("bounded host-assigned authority required")
        if self.reliability is not None and (type(self.reliability) not in (int, float)
                or not math.isfinite(self.reliability) or not 0 <= self.reliability <= 1):
            raise ValueError("bounded source reliability required")
        if any(c.status in ("KNOWN_ABSENT", "NOT_APPLICABLE") for c in self.claims):
            if self.temporal.valid_until is None or self.validation_revision is None:
                raise ValueError("negative knowledge needs validation and finite expiry")

    @cached_property
    def ref(self):
        return NodeRef(self.key, self.revision, hashlib.sha256(canonical(asdict(self)).encode()).hexdigest())


@dataclass(frozen=True)
class StoredNode:
    node: CanonicalNode
    transaction_time: int


@dataclass(frozen=True)
class ContextSnapshot:
    scope: AccessScope
    at: int
    known_at: int
    records: tuple[StoredNode, ...]
    state_revision: str


@dataclass(frozen=True)
class EvidenceRequirement:
    name: str
    kinds: tuple[str, ...] = ("FACT", "POLICY")
    statuses: tuple[str, ...] = ("PRESENT",)
    verified: bool = True

    def __post_init__(self):
        identifier(self.name)
        typed_tuple(self.kinds, str, len(KINDS))
        typed_tuple(self.statuses, str, 3)
        if not self.kinds or not self.statuses or type(self.verified) is not bool:
            raise ValueError("explicit evidence requirements required")
        for kind in self.kinds:
            category(kind, KINDS)
        for status in self.statuses:
            category(status, {"PRESENT", "KNOWN_ABSENT", "NOT_APPLICABLE"})
        if self.verified and not set(self.kinds) <= VERIFIABLE_KINDS:
            raise ValueError("assumptions, claims and model outputs cannot satisfy verified requirements")

    def accepts(self, node, claim):
        return (claim.name == self.name and claim.status in self.statuses and node.kind in self.kinds
                and (not self.verified or node.validation_revision is not None))


class ContextState:
    """Tenant-local append-only versions with current host-supplied ACLs.

    Changes invalidate outstanding snapshots conservatively. Snapshot seals are
    in-process integrity checks, never bearer credentials or remote signatures.
    """

    def __init__(self, *, tenant, policy_revision, clock=None, capacity=2048):
        identifier(tenant)
        identifier(policy_revision)
        if type(capacity) is not int or not 1 <= capacity <= 10000:
            raise ValueError("bounded state capacity required")
        self.tenant, self.policy_revision, self.capacity = tenant, policy_revision, capacity
        self._clock = clock or (lambda: time.time_ns() // 1000000)
        self._records, self._acl, self._revoked = {}, {}, set()
        self._epoch, self._last_time = 0, 0
        self._secret, self._lock = secrets.token_bytes(32), threading.RLock()

    def put(self, node):
        if type(node) is not CanonicalNode or node.tenant != self.tenant:
            raise ValueError("same-tenant canonical node required")
        with self._lock:
            key = node.key, node.revision
            previous = self._records.get(key)
            if previous:
                if previous.node != node:
                    raise ValueError("revision collision; create a new revision")
                return previous
            if len(self._records) >= self.capacity:
                raise ValueError("state capacity reached")
            now = self._clock()
            timestamp(now)
            now = max(now, self._last_time)
            if node.temporal.observed_at > now:
                raise ValueError("observed evidence cannot arrive from the future")
            for ref in node.dependencies + node.supersedes:
                target = self._records.get((ref.key, ref.revision))
                if target is None or target.node.ref != ref:
                    raise ValueError("exact previously ingested reference required")
            if any(ref.key != node.key for ref in node.supersedes):
                raise ValueError("supersession stays within one logical source")
            stored = StoredNode(node, now)
            self._records[key] = stored
            # Ingesting an old observation must not overwrite a newer ACL.
            self._acl.setdefault(node.key, node.roles)
            self._last_time = now
            self._epoch += 1
            return stored

    def set_roles(self, key, roles):
        identifier(key)
        typed_tuple(roles, str, 32)
        for role in roles:
            identifier(role)
        with self._lock:
            if key not in self._acl:
                raise ValueError("unknown source")
            self._acl[key] = tuple(sorted(roles))
            self._epoch += 1

    def set_policy(self, revision):
        identifier(revision)
        with self._lock:
            self.policy_revision = revision
            self._epoch += 1

    def invalidate(self, ref):
        if type(ref) is not NodeRef:
            raise ValueError("exact source reference required")
        with self._lock:
            stored = self._records.get((ref.key, ref.revision))
            if stored is None or stored.node.ref != ref:
                raise ValueError("unknown source reference")
            self._revoked.add(ref)
            self._epoch += 1

    def _authorized(self, scope):
        return type(scope) is AccessScope and scope.tenant == self.tenant and scope.policy_revision == self.policy_revision

    def _signature(self, scope, at, known_at, records):
        payload = {"epoch": self._epoch, "scope": asdict(scope), "at": at, "known_at": known_at,
                   "records": [(asdict(r.node.ref), r.transaction_time) for r in records]}
        return hmac.new(self._secret, canonical(payload).encode(), hashlib.sha256).hexdigest()

    def snapshot(self, scope, *, at, known_at):
        timestamp(at)
        timestamp(known_at)
        with self._lock:
            if not self._authorized(scope):
                raise PermissionError("unavailable context")
            known = {r.node.ref: r for r in self._records.values()
                     if r.transaction_time <= known_at and r.node.temporal.observed_at <= known_at}
            # Expiry/revocation of a replacement does not resurrect an older version.
            retired = set()
            pending = [ref for r in known.values() if r.node.temporal.start <= at for ref in r.node.supersedes]
            while pending:
                ref = pending.pop()
                if ref not in retired:
                    retired.add(ref)
                    if ref in known:
                        pending.extend(known[ref].node.supersedes)
            excluded = retired | self._revoked
            visible = {ref: r for ref, r in known.items() if ref not in excluded
                and r.node.temporal.applies(at) and r.node.lifecycle not in ("INVALID", "SUPERSEDED")
                and set(scope.roles).intersection(self._acl[r.node.key])}
            # Do not leak hidden dependency IDs through an otherwise visible node.
            dependents, blocked = {}, []
            for ref, record in visible.items():
                for dependency in record.node.dependencies:
                    dependents.setdefault(dependency, []).append(ref)
                if any(dependency not in visible for dependency in record.node.dependencies):
                    blocked.append(ref)
            while blocked:
                ref = blocked.pop()
                if ref in visible:
                    del visible[ref]
                    blocked.extend(dependents.get(ref, ()))
            records = tuple(sorted(visible.values(), key=lambda r: (r.node.key, r.node.revision)))
            return ContextSnapshot(scope, at, known_at, records, self._signature(scope, at, known_at, records))

    def is_current(self, snapshot):
        with self._lock:
            if type(snapshot) is not ContextSnapshot or not self._authorized(snapshot.scope):
                return False
            try:
                expected = self._signature(snapshot.scope, snapshot.at, snapshot.known_at, snapshot.records)
                return isinstance(snapshot.state_revision, str) and hmac.compare_digest(snapshot.state_revision, expected)
            except (AttributeError, TypeError, ValueError):
                return False

    def subset(self, snapshot, refs):
        typed_tuple(refs, NodeRef, self.capacity)
        with self._lock:
            if not self.is_current(snapshot):
                raise ValueError("state changed")
            available = {r.node.ref: r for r in snapshot.records}
            if not set(refs) <= available.keys():
                raise ValueError("selection outside authorized snapshot")
            records = tuple(sorted((available[ref] for ref in refs), key=lambda r: (r.node.key, r.node.revision)))
            return ContextSnapshot(snapshot.scope, snapshot.at, snapshot.known_at, records,
                                   self._signature(snapshot.scope, snapshot.at, snapshot.known_at, records))

    def seal(self, snapshot, payload):
        """Host-only binding of a compiled packet to its still-current snapshot."""
        bounded_text(payload, 1048576)
        with self._lock:
            if not self.is_current(snapshot):
                raise ValueError("state changed")
            return hmac.new(self._secret, (snapshot.state_revision + "\0" + payload).encode(), hashlib.sha256).hexdigest()

    def verify_binding(self, snapshot, payload, seal):
        if not isinstance(seal, str):
            return False
        try:
            return hmac.compare_digest(self.seal(snapshot, payload), seal)
        except (ValueError, TypeError):
            return False

"""Bounded, process-local evidence reuse with fail-closed invalidation.

Opaque 32-byte receipt IDs are separate from semantic 32-byte stamps. Neither
is an authorization credential. The host supplies authenticated roles/versions.
"""

import secrets
import threading
import time
from collections import OrderedDict

from .security import identifier
from .workflow import ContextGraph, Handoff


class ContextSession:
    """Owns its graph; serializes mutations and reads under a reentrant lock.

    Any successful mutation invalidates all receipts, including dependencies.
    This deliberately conservative strategy cannot miss an affected dependency.
    Reuse saves packet construction/transfer, not the model's evidence tokens.
    """

    def __init__(self, *, max_entries=128, max_bytes=8388608, ttl_seconds=300):
        if type(max_entries) is not int or not 1 <= max_entries <= 1000:
            raise ValueError("max_entries must be between one and 1000")
        if type(max_bytes) is not int or not 1 <= max_bytes <= 67108864:
            raise ValueError("max_bytes must be between one and 64 MiB")
        if not isinstance(ttl_seconds, (int, float)) or not 0 < ttl_seconds <= 3600:
            raise ValueError("TTL must be positive and at most one hour")
        self._graph, self._lock = ContextGraph(), threading.RLock()
        self._entries, self._keys = OrderedDict(), {}
        self._bytes = 0
        self._max_entries, self._max_bytes, self._ttl = max_entries, max_bytes, ttl_seconds

    def _clear(self):
        self._entries.clear()
        self._keys.clear()
        self._bytes = 0

    def put(self, node):
        with self._lock:
            self._graph.put(node)
            self._clear()

    def link(self, source, target, kind, *, provenance):
        with self._lock:
            self._graph.link(source, target, kind, provenance=provenance)
            self._clear()

    def _remove(self, token):
        key, packet, _, _ = self._entries.pop(token)
        del self._keys[key]
        self._bytes -= packet.units

    def _expire(self):
        now = time.monotonic()
        for token, (_, _, expires, _) in list(self._entries.items()):
            if expires <= now:
                self._remove(token)

    def issue(self, roots, *, role, revisions, budget_bytes=8192):
        """Return (packet, receipt-or-None, reused). Failed packets are never cached."""
        identifier(role)
        if isinstance(roots, str) or not 1 <= len(roots) <= 100:
            raise ValueError("one to 100 roots required")
        roots = tuple(roots)
        for root in roots:
            identifier(root)
        if not isinstance(revisions, dict) or len(revisions) > 1000:
            raise ValueError("bounded revision mapping required")
        revisions = dict(revisions)
        if type(budget_bytes) is not int or not 0 <= budget_bytes <= 1048576:
            raise ValueError("budget must be between zero and 1 MiB")
        key = (tuple(sorted(set(roots))), role, budget_bytes)
        with self._lock:
            self._expire()
            token = self._keys.get(key)
            if token is not None:
                if any(revisions.get(source) != revision for source, revision in self._entries[token][3]):
                    return Handoff("insufficient", "", (), 0, "unavailable_context"), None, False
                self._entries.move_to_end(token)
                return self._entries[token][1], token, True
            packet = self._graph.handoff(roots, role=role, revisions=revisions, budget_bytes=budget_bytes)
            if packet.status != "complete" or packet.units > self._max_bytes:
                return packet, None, False
            while self._entries and (len(self._entries) >= self._max_entries
                                     or self._bytes + packet.units > self._max_bytes):
                self._remove(next(iter(self._entries)))
            token = secrets.token_bytes(32)
            while token in self._entries:
                token = secrets.token_bytes(32)
            versions = tuple((source, revisions[source]) for source in packet.sources)
            self._entries[token] = key, packet, time.monotonic() + self._ttl, versions
            self._keys[key] = token
            self._bytes += packet.units
            return packet, token, False

    def resolve(self, token, *, role, revisions, budget_bytes=8192):
        """Resolve a live receipt under the same host authorization and versions.

        Missing, expired, revoked and cross-role receipts have the same response.
        The lock protects this operation only; remote use needs host transactions.
        """
        if not isinstance(token, bytes) or len(token) != 32:
            raise ValueError("exactly 32 receipt bytes required")
        identifier(role)
        with self._lock:
            self._expire()
            item = self._entries.get(token)
            if item is None or item[0][1] != role:
                return Handoff("insufficient", "", (), 0, "unavailable_context")
            roots = item[0][0]
            packet, _, _ = self.issue(roots, role=role, revisions=revisions, budget_bytes=budget_bytes)
            return packet

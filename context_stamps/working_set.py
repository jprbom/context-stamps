"""Bounded context residency over a durable, currently authorized state owner.

Copyright (c) 2026 Prashant Jagtap. MIT License.
Eviction removes working copies, never retained evidence. Prediction is external;
prefetch is explicit and remains inactive until demanded.
"""

from __future__ import annotations

import secrets
import threading
import time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

from .audit import AuditIntegrityError
from .context_state import AccessScope, ContextSnapshot, NodeRef, typed_tuple
from .state_store import ContextStore


@dataclass(frozen=True)
class PageResult:
    status: str
    cold_reads: int | None = None
    reused: int | None = None


@dataclass(frozen=True)
class _Resident:
    record: object
    size: int
    touched: float
    prefetched: bool


class ContextWorkingSet:
    """One authenticated scope, byte/node quota, active closure and pinned roots.

    Resident bytes count canonical serialized nodes, not Python allocator overhead
    or model tokens. ContextCompiler separately counts the complete model packet.
    One background prefetch thread and a bounded ticket queue are supported.
    """

    def __init__(self, store, *, scope, max_nodes=128, max_bytes=4 * 1024 * 1024, max_prefetch=4):
        if not isinstance(store, ContextStore) or type(scope) is not AccessScope:
            raise ValueError("durable context owner and authenticated scope required")
        for value, upper in ((max_nodes, store.capacity), (max_bytes, store.materialize_bytes), (max_prefetch, 16)):
            if type(value) is not int or not 1 <= value <= upper:
                raise ValueError("bounded per-scope working set required")
        self.store, self.scope = store, scope
        self.max_nodes, self.max_bytes, self.max_prefetch = max_nodes, max_bytes, max_prefetch
        self._lock, self._closed, self._generation = threading.RLock(), False, 0
        self._demand_lock, self._foreground_generation = threading.RLock(), None
        self._cache, self._tickets, self._cancelled = OrderedDict(), {}, set()
        self._roots, self._pins, self._active_refs = (), set(), set()
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="context-prefetch")
        self._counts = dict(cold_reads=0, cache_reuses=0, prefetched=0, prefetch_used=0,
                            evictions=0, wasted_prefetch=0, discarded_prefetch=0, page_faults=0)

    def _open(self):
        if self._closed:
            raise RuntimeError("working set closed")

    def _closure(self, roots):
        result, pending = set(), list(roots)
        while pending:
            ref = pending.pop()
            if ref not in result:
                if ref not in self._cache:
                    raise ValueError("required context is not resident")
                result.add(ref)
                pending.extend(self._cache[ref].record.node.dependencies)
        return result

    def _remove(self, ref):
        item = self._cache.pop(ref)
        self._counts["evictions"] += 1
        self._counts["wasted_prefetch"] += int(item.prefetched)

    def _clear(self):
        for ref in tuple(self._cache):
            self._remove(ref)
        self._roots, self._active_refs = (), set()
        self._generation += 1

    def _load(self, roots, *, at, known_at, prefetch=False, pin=None, ticket=None):
        typed_tuple(roots, NodeRef, self.max_nodes)
        with self._lock:
            self._open()
            if ticket in self._cancelled:
                return PageResult("cancelled", 0, 0)
            if prefetch and self._foreground_generation is not None:
                return PageResult("deferred", 0, 0)
            if not prefetch:
                self._generation += 1
                self._foreground_generation = self._generation
            generation = self._generation
            pins = self._pins | ({pin} if pin is not None else set())
            requested = set(roots) | pins
            if prefetch:
                requested.update(self._roots)
            cached = tuple(item.record for item in self._cache.values())
        # Disk and host callbacks run outside the working-set lock. A generation
        # check prevents a late prefetch from modifying a changed active window.
        try:
            inventory = self.store.inventory(self.scope, at=at, known_at=known_at,
                                              roots=tuple(sorted(requested, key=lambda r: (r.key, r.revision, r.fingerprint))))
            if len(inventory.entries) > self.max_nodes or sum(e.payload_bytes for e in inventory.entries) > self.max_bytes:
                raise OverflowError("dependency closure exceeds working-set quota")
            old_refs = {r.node.ref for r in cached}
            cold = sum(e.ref not in old_refs for e in inventory.entries)
            reused = len(inventory.entries) - cold
            snapshot = self.store.materialize(inventory, cached=cached)
            with self._lock:
                self._counts["cold_reads"] += cold
                self._counts["cache_reuses"] += reused
                if self._closed or generation != self._generation or ticket in self._cancelled:
                    self._counts["discarded_prefetch"] += int(prefetch)
                    return PageResult("discarded", cold, reused)
                if not self.store.is_current(snapshot):
                    self._clear()
                    self._counts["discarded_prefetch"] += int(prefetch)
                    raise ValueError("state changed during page-in")
                incoming = {r.node.ref for r in snapshot.records}
                # Every active/pinned dependency is part of the request above.
                sizes = {entry.ref: entry.payload_bytes for entry in inventory.entries}
                retained = set(self._cache) | incoming
                def total():
                    return sum(sizes[r] if r in sizes else self._cache[r].size for r in retained)
                candidates = sorted((r for r in self._cache if r not in incoming),
                                    key=lambda r: (not self._cache[r].prefetched, self._cache[r].touched))
                victims = []
                for ref in candidates:
                    if len(retained) <= self.max_nodes and total() <= self.max_bytes:
                        break
                    retained.remove(ref)
                    victims.append(ref)
                if len(retained) > self.max_nodes or total() > self.max_bytes:
                    raise OverflowError("protected context exceeds working-set quota")
                for ref in victims:
                    self._remove(ref)
                for record in snapshot.records:
                    ref = record.node.ref
                    old = self._cache.get(ref)
                    speculative = (old.prefetched if old is not None else prefetch)
                    if prefetch and old is None:
                        self._counts["prefetched"] += 1
                    if not prefetch and speculative:
                        self._counts["prefetch_used"] += 1
                        speculative = False
                    self._cache[ref] = _Resident(record, sizes[ref], time.perf_counter(), speculative)
                    self._cache.move_to_end(ref)
                self._pins = pins
                if not prefetch:
                    self._roots, self._active_refs = roots, incoming
                self._generation += 1
                return PageResult("ready", cold, reused) if prefetch else snapshot
        except PermissionError:
            with self._lock:
                self._clear()
            raise
        finally:
            if not prefetch:
                with self._lock:
                    if self._foreground_generation == generation:
                        self._foreground_generation = None

    def activate(self, roots, *, at, known_at):
        """Replace the active root set; include exact dependencies and pinned roots."""
        with self._demand_lock:
            result = self._load(roots, at=at, known_at=known_at)
        if type(result) is not ContextSnapshot:
            raise ValueError("working set changed during activation; retry with current state")
        return result

    def pin(self, ref, *, at, known_at):
        typed_tuple((ref,), NodeRef, 1)
        with self._lock:
            roots = self._roots
        with self._demand_lock:
            result = self._load(roots, at=at, known_at=known_at, pin=ref)
        if type(result) is not ContextSnapshot:
            raise ValueError("working set changed during pin")
        return result

    def unpin(self, ref):
        with self._lock:
            self._open()
            self._pins.discard(ref)
            self._active_refs = self._closure(set(self._roots) | self._pins) if self._cache else set()
            self._generation += 1

    def fault(self, ref, *, at, known_at):
        with self._lock:
            self._open()
            self._counts["page_faults"] += 1
            roots = tuple(dict.fromkeys((*self._roots, ref)))
        return self.activate(roots, at=at, known_at=known_at)

    def page_out(self, ref):
        typed_tuple((ref,), NodeRef, 1)
        with self._lock:
            self._open()
            if ref in self._active_refs or ref in self._closure(self._pins):
                raise ValueError("active or pinned dependency cannot be evicted")
            if ref in self._cache:
                self._remove(ref)
                self._generation += 1
                return True
            return False

    def residency(self, ref, *, at, known_at):
        """Current per-scope residency; unavailable evidence has no visible state."""
        typed_tuple((ref,), NodeRef, 1)
        with self._lock:
            self._open()
            self.store.inventory(self.scope, at=at, known_at=known_at, roots=(ref,))
            return "HOT" if ref in self._active_refs else "WARM" if ref in self._cache else "COLD"

    def collect(self, *, idle_seconds):
        """Discard idle unprotected working copies; no persistent evidence deletion."""
        if type(idle_seconds) not in (float, int) or not 0 <= idle_seconds <= 86400:
            raise ValueError("bounded idle interval required")
        with self._lock:
            self._open()
            protected = self._active_refs | self._closure(self._pins)
            expired = [r for r, item in self._cache.items() if r not in protected and time.perf_counter() - item.touched >= idle_seconds]
            for ref in expired:
                self._remove(ref)
            self._generation += int(bool(expired))
            return len(expired)

    def prefetch(self, roots, *, at, known_at):
        typed_tuple(roots, NodeRef, self.max_nodes)
        with self._lock:
            self._open()
            if len(self._tickets) >= self.max_prefetch:
                raise OverflowError("prefetch ticket quota reached; consume completed tickets")
            ticket = secrets.token_hex(16)
            self._tickets[ticket] = self._pool.submit(self._prefetch, roots, at, known_at, ticket)
            return ticket

    def _prefetch(self, roots, at, known_at, ticket):
        try:
            return self._load(roots, at=at, known_at=known_at, prefetch=True, ticket=ticket)
        except PermissionError:
            return PageResult("unavailable")
        except OverflowError:
            return PageResult("budget")
        except AuditIntegrityError:
            return PageResult("integrity")
        except ValueError:
            return PageResult("stale")
        except Exception:
            return PageResult("error")

    def cancel_prefetch(self, ticket):
        with self._lock:
            self._open()
            future = self._tickets[ticket]
            if future.done():
                return False
            self._cancelled.add(ticket)
            future.cancel()
            return True

    def prefetch_result(self, ticket, *, consume=True):
        with self._lock:
            self._open()
            future = self._tickets[ticket]
            if not future.done():
                return PageResult("pending")
            result = PageResult("cancelled") if future.cancelled() else future.result()
            if consume:
                del self._tickets[ticket]
                self._cancelled.discard(ticket)
            return result

    def snapshot(self, scope, *, at, known_at):
        if scope != self.scope:
            raise PermissionError("unavailable working set")
        with self._lock:
            roots = self._roots
        return self.activate(roots, at=at, known_at=known_at)

    def is_current(self, snapshot):
        with self._lock:
            self._open()
            return (type(snapshot) is ContextSnapshot and snapshot.scope == self.scope
                    and {r.node.ref for r in snapshot.records} <= self._active_refs
                    and self.store.is_current(snapshot))

    def subset(self, snapshot, refs):
        with self._lock:
            if not self.is_current(snapshot):
                raise ValueError("working context changed")
            return self.store.subset(snapshot, refs)

    def seal(self, snapshot, payload):
        with self._lock:
            if not self.is_current(snapshot):
                raise ValueError("working context changed")
            return self.store.seal(snapshot, payload)

    def verify_binding(self, snapshot, payload, seal):
        with self._lock:
            return self.is_current(snapshot) and self.store.verify_binding(snapshot, payload, seal)

    def stats(self):
        with self._lock:
            self._open()
            self.store.inventory(self.scope, at=0, known_at=0, roots=())
            return {**self._counts, "resident_nodes": len(self._cache),
                    "resident_bytes": sum(item.size for item in self._cache.values()),
                    "active_nodes": len(self._active_refs), "pinned_roots": len(self._pins),
                    "prefetch_usefulness": self._counts["prefetch_used"] / self._counts["prefetched"] if self._counts["prefetched"] else None}

    def close(self):
        with self._lock:
            if self._closed:
                return
            self._closed = True
            self._clear()
            for ticket, future in self._tickets.items():
                self._cancelled.add(ticket)
                future.cancel()
        self._pool.shutdown(wait=True, cancel_futures=True)

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

"""Bounded pure-computation DAGs with exact reuse and authenticated receipts.

Copyright (c) 2026 Prashant Jagtap. MIT License.
Hosts register pure callbacks and complete versioned inputs. This is not an
effect executor, a process sandbox, or a distributed exactly-once scheduler.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import math
import secrets
import threading
import time
from collections import OrderedDict
from dataclasses import asdict, dataclass, replace

from .context_state import (
    AccessScope,
    ContextSnapshot,
    NodeRef,
    StoredNode,
    canonical,
    digest,
    timestamp,
    typed_tuple,
)
from .experience import ResourceUse
from .security import bounded_text, identifier


def _hash(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def _json_object(text):
    def pairs(values):
        result = {}
        for key, value in values:
            if key in result:
                raise ValueError("duplicate request key")
            result[key] = value
        return result

    def constant(_):
        raise ValueError("finite JSON required")

    bounded_text(text, 16384)
    try:
        value = json.loads(text, object_pairs_hook=pairs, parse_constant=constant)
        if type(value) is not dict:
            raise ValueError("request must be a JSON object")
        return canonical(value)
    except (RecursionError, OverflowError) as error:
        raise ValueError("request nesting or numeric range exceeded") from error


@dataclass(frozen=True)
class ComputeSpec:
    key: str
    operation_revision: str
    verifier_revision: str
    sources: tuple[NodeRef, ...] = ()
    parents: tuple[str, ...] = ()
    request: str = "{}"
    model_revision: str = "none"
    prompt_revision: str = "none"
    tokenizer_revision: str = "none"
    tool_revision: str = "none"
    reusable: bool = True

    def __post_init__(self):
        for value in (self.key, self.operation_revision, self.verifier_revision, self.model_revision,
                      self.prompt_revision, self.tokenizer_revision, self.tool_revision):
            identifier(value)
        typed_tuple(self.sources, NodeRef, 128)
        typed_tuple(self.parents, str, 32)
        for parent in self.parents:
            identifier(parent)
        if self.key in self.parents or type(self.reusable) is not bool:
            raise ValueError("self dependency or invalid reuse declaration")
        object.__setattr__(self, "sources", tuple(sorted(self.sources, key=lambda r: (r.key, r.revision))))
        object.__setattr__(self, "parents", tuple(sorted(self.parents)))
        object.__setattr__(self, "request", _json_object(self.request))


@dataclass(frozen=True)
class ComputeDefinition:
    """Receipt-safe identity: request contents stay outside the receipt."""

    key: str
    operation_revision: str
    verifier_revision: str
    model_revision: str
    prompt_revision: str
    tokenizer_revision: str
    tool_revision: str
    request_hash: str
    spec_hash: str
    reusable: bool

    def __post_init__(self):
        for value in (self.key, self.operation_revision, self.verifier_revision, self.model_revision,
                      self.prompt_revision, self.tokenizer_revision, self.tool_revision):
            identifier(value)
        digest(self.request_hash)
        digest(self.spec_hash)
        if type(self.reusable) is not bool:
            raise ValueError("boolean reuse declaration required")

    @classmethod
    def from_spec(cls, spec):
        return cls(spec.key, spec.operation_revision, spec.verifier_revision, spec.model_revision,
                   spec.prompt_revision, spec.tokenizer_revision, spec.tool_revision,
                   hashlib.sha256(spec.request.encode()).hexdigest(), _hash(asdict(spec)), spec.reusable)


@dataclass(frozen=True)
class SourceBinding:
    ref: NodeRef
    transaction_time: int
    stamp_hex: str | None
    stamp_schema: str | None

    def __post_init__(self):
        if type(self.ref) is not NodeRef:
            raise ValueError("typed source identity required")
        timestamp(self.transaction_time)
        if (self.stamp_hex is None) != (self.stamp_schema is None):
            raise ValueError("stamp identity and schema required together")
        if self.stamp_hex is not None:
            digest(self.stamp_hex)
            identifier(self.stamp_schema)


@dataclass(frozen=True)
class ParentBinding:
    key: str
    computation_hash: str
    result_hash: str

    def __post_init__(self):
        identifier(self.key)
        digest(self.computation_hash)
        digest(self.result_hash)


@dataclass(frozen=True)
class ComputationReceipt:
    scope: AccessScope
    spec: ComputeDefinition
    computation_hash: str
    sources: tuple[SourceBinding, ...]
    parents: tuple[ParentBinding, ...]
    result_hash: str
    created_ms: int
    signature: str

    def __post_init__(self):
        if type(self.scope) is not AccessScope or type(self.spec) is not ComputeDefinition:
            raise ValueError("typed scope and specification required")
        typed_tuple(self.sources, SourceBinding, 2048)
        typed_tuple(self.parents, ParentBinding, 32)
        for value in (self.computation_hash, self.result_hash, self.signature):
            digest(value)
        timestamp(self.created_ms)


@dataclass(frozen=True)
class ParentValue:
    binding: ParentBinding
    text: str


@dataclass(frozen=True)
class ComputeInput:
    scope: AccessScope
    spec: ComputeSpec
    records: tuple[StoredNode, ...]
    parents: tuple[ParentValue, ...]


@dataclass(frozen=True)
class ComputeOutput:
    text: str
    usage: ResourceUse | None = None

    def __post_init__(self):
        bounded_text(self.text, 65536)
        if self.usage is not None and type(self.usage) is not ResourceUse:
            raise ValueError("typed usage or unknown required")


@dataclass(frozen=True)
class ComputeAdapter:
    operation_revision: str
    verifier_revision: str
    compute: object
    verify: object
    pure: bool = True

    def __post_init__(self):
        identifier(self.operation_revision)
        identifier(self.verifier_revision)
        if self.pure is not True or not callable(self.compute) or not callable(self.verify):
            raise ValueError("registered pure computation and verifier required")


@dataclass(frozen=True)
class DAGBudget:
    compute_calls: int = 256
    result_bytes: int = 4 * 1024 * 1024
    milliseconds: float = 120000

    def __post_init__(self):
        if type(self.compute_calls) is not int or not 0 <= self.compute_calls <= 256:
            raise ValueError("invalid computation call bound")
        if type(self.result_bytes) is not int or not 0 <= self.result_bytes <= 16 * 1024 * 1024:
            raise ValueError("invalid result byte bound")
        if (type(self.milliseconds) not in (int, float) or not math.isfinite(self.milliseconds)
                or not 0 < self.milliseconds <= 120000):
            raise ValueError("invalid deadline")


@dataclass(frozen=True)
class ComputeStep:
    key: str
    status: str
    receipt: ComputationReceipt | None
    usage: ResourceUse


@dataclass(frozen=True)
class ComputationResult:
    key: str
    text: str
    receipt: ComputationReceipt


@dataclass(frozen=True)
class DAGResult:
    status: str
    steps: tuple[ComputeStep, ...]
    results: tuple[ComputationResult, ...]
    elapsed_ms: float


def active_order(graph, targets):
    typed_tuple(graph, ComputeSpec, 256)
    typed_tuple(targets, str, 32)
    nodes = {node.key: node for node in graph}
    if not targets or len(nodes) != len(graph) or not set(targets) <= nodes.keys():
        raise ValueError("unique graph keys and existing targets required")
    if any(not set(node.parents) <= nodes.keys() for node in graph):
        raise ValueError("missing computation parent")
    # Check the entire bounded graph, including currently inactive branches.
    remaining, ordered = set(nodes), []
    while remaining:
        ready = sorted(key for key in remaining if not set(nodes[key].parents).intersection(remaining))
        if not ready:
            raise ValueError("cyclic computation graph")
        ordered.extend(ready)
        remaining.difference_update(ready)
    active, pending = set(), list(targets)
    while pending:
        key = pending.pop()
        if key not in active:
            active.add(key)
            pending.extend(nodes[key].parents)
    return tuple(nodes[key] for key in ordered if key in active)


def affected_nodes(graph, changed, targets):
    """Changed computations plus their active descendants; not a reuse permission."""
    order = active_order(graph, targets)
    typed_tuple(changed, str, 256)
    if not set(changed) <= {s.key for s in graph}:
        raise ValueError("unknown changed computation")
    affected = set(changed)
    for spec in order:
        if affected.intersection(spec.parents):
            affected.add(spec.key)
    return tuple(spec.key for spec in order if spec.key in affected)


def _source_bindings(records):
    return tuple(SourceBinding(r.node.ref, r.transaction_time, r.node.stamp_hex, r.node.stamp_schema)
                 for r in sorted(records, key=lambda r: (r.node.key, r.node.revision)))


class _Stopped(Exception):
    pass


class IncrementalExecutor:
    """Single-flight, in-process pure DAG execution with a bounded private LRU.

    Host callbacks run outside cache locks. Cancellation/deadlines reject late
    output but cannot interrupt Python callbacks; use managed workers for that.
    Fresh state and host authorization are required on hits and before release.
    Receipts attest origin/integrity, never grant present access or prove truth.
    """

    def __init__(self, state, adapters, *, authorize, signing_key=None, capacity=256,
                 max_bytes=4 * 1024 * 1024, ttl_seconds=300, clock=None):
        typed_tuple(adapters, ComputeAdapter, 128)
        if len({a.operation_revision for a in adapters}) != len(adapters) or not adapters:
            raise ValueError("unique registered operation revisions required")
        if not callable(authorize):
            raise ValueError("host authorization required")
        if type(capacity) is not int or not 1 <= capacity <= 4096:
            raise ValueError("invalid cache capacity")
        if type(max_bytes) is not int or not 1 <= max_bytes <= 64 * 1024 * 1024:
            raise ValueError("invalid cache byte bound")
        if (type(ttl_seconds) not in (int, float) or not math.isfinite(ttl_seconds)
                or not 0 < ttl_seconds <= 86400):
            raise ValueError("invalid cache TTL")
        if signing_key is not None and (type(signing_key) is not bytes or len(signing_key) != 32):
            raise ValueError("32-byte receipt key required")
        self._state, self._authorize = state, authorize
        self._adapters = {a.operation_revision: a for a in adapters}
        self._key = signing_key if signing_key is not None else secrets.token_bytes(32)
        self._capacity, self._max_bytes, self._ttl = capacity, max_bytes, ttl_seconds
        self._clock = clock or (lambda: time.time_ns() // 1000000)
        self._cache, self._bytes = OrderedDict(), 0
        self._lock, self._flight = threading.RLock(), threading.Lock()
        self._generation, self._closed = 0, False

    def _signature(self, receipt):
        payload = asdict(receipt)
        payload.pop("signature")
        return hmac.new(self._key, ("computation-receipt-v1:" + canonical(payload)).encode(),
                        hashlib.sha256).hexdigest()

    def verify_receipt(self, receipt, text=None):
        if type(receipt) is not ComputationReceipt:
            return False
        if text is not None:
            try:
                bounded_text(text, 65536)
            except (TypeError, ValueError):
                return False
            if hashlib.sha256(text.encode()).hexdigest() != receipt.result_hash:
                return False
        return hmac.compare_digest(receipt.signature, self._signature(receipt))

    def _remove(self, key):
        _, _, _, size = self._cache.pop(key)
        self._bytes -= size

    def _get(self, key):
        with self._lock:
            for old, (expires, _, _, _) in tuple(self._cache.items()):
                if time.perf_counter() >= expires:
                    self._remove(old)
            found = self._cache.get(key)
            if found:
                self._cache.move_to_end(key)
                return found[1], found[2]
        return None

    def _put(self, text, receipt):
        # Count retained serialized receipt metadata as well as result payloads.
        size = len(text.encode()) + len(canonical(asdict(receipt)).encode())
        if size > self._max_bytes:
            return
        key = receipt.computation_hash
        if key in self._cache:
            self._remove(key)
        while self._cache and (len(self._cache) >= self._capacity or self._bytes + size > self._max_bytes):
            self._remove(next(iter(self._cache)))
        self._cache[key] = (time.perf_counter() + self._ttl, text, receipt, size)
        self._bytes += size

    def cache_info(self):
        with self._lock:
            return {"entries": len(self._cache), "serialized_bytes": self._bytes,
                    "capacity": self._capacity, "max_bytes": self._max_bytes}

    def clear(self):
        with self._lock:
            self._generation += 1
            self._cache.clear()
            self._bytes = 0

    def close(self):
        with self._lock:
            self._closed = True
            self.clear()

    def run(self, graph, targets, *, scope, at, known_at, budget=DAGBudget(), cancel=None):
        order = active_order(graph, targets)
        if type(scope) is not AccessScope or type(budget) is not DAGBudget:
            raise ValueError("typed scope and budget required")
        timestamp(at)
        timestamp(known_at)
        if cancel is not None and not isinstance(cancel, threading.Event):
            raise ValueError("threading.Event cancellation required")
        for spec in order:
            adapter = self._adapters.get(spec.operation_revision)
            if adapter is None or adapter.verifier_revision != spec.verifier_revision:
                raise ValueError("unregistered operation or mismatched verifier revision")
        if not self._flight.acquire(blocking=False):
            raise RuntimeError("executor already running")
        start, steps, values, staged = time.perf_counter(), [], {}, []
        generation = self._generation
        snapshot = None

        def check(spec=None):
            with self._lock:
                if self._closed or self._generation != generation:
                    raise _Stopped("invalidated")
            if cancel is not None and cancel.is_set():
                raise _Stopped("cancelled")
            if (time.perf_counter() - start) * 1000 >= budget.milliseconds:
                raise _Stopped("deadline")
            if snapshot is not None and not self._state.is_current(snapshot):
                raise _Stopped("state_changed")
            if spec is not None:
                try:
                    allowed = self._authorize(scope, spec) is True
                except Exception:
                    allowed = False
                if not allowed:
                    raise _Stopped("unauthorized")

        def finish(status):
            results = tuple(ComputationResult(k, values[k][0], values[k][1]) for k in targets) if status == "complete" else ()
            released_steps = tuple(steps) if status == "complete" else tuple(replace(s, receipt=None) for s in steps)
            return DAGResult(status, released_steps, results, (time.perf_counter() - start) * 1000)

        try:
            check()
            snapshot = self._state.snapshot(scope, at=at, known_at=known_at)
            if (type(snapshot) is not ContextSnapshot or snapshot.scope != scope
                    or snapshot.at != at or snapshot.known_at != known_at or len(snapshot.records) > 2048):
                raise _Stopped("invalid_snapshot")
            check()
            available = {r.node.ref: r for r in snapshot.records}
            calls, total_bytes = 0, 0
            for spec in order:
                step_start, status, receipt = time.perf_counter(), "failed", None
                usage = ResourceUse(0, 0, 0, 0, 0)
                try:
                    check(spec)
                    selected, pending = set(), list(spec.sources)
                    while pending:
                        ref = pending.pop()
                        if ref not in available:
                            raise _Stopped("unavailable_context")
                        if ref not in selected:
                            selected.add(ref)
                            pending.extend(available[ref].node.dependencies)
                    records = tuple(available[r] for r in sorted(selected, key=lambda r: (r.key, r.revision)))
                    parents = tuple(ParentValue(ParentBinding(k, values[k][1].computation_hash,
                                                             values[k][1].result_hash), values[k][0]) for k in spec.parents)
                    bindings = {b for k in spec.parents for b in values[k][1].sources}
                    bindings.update(_source_bindings(records))
                    bindings = tuple(sorted(bindings, key=lambda b: (b.ref.key, b.ref.revision)))
                    if len(bindings) > 2048:
                        raise _Stopped("source_budget")
                    definition = ComputeDefinition.from_spec(spec)
                    key = _hash({"schema": 1, "scope": asdict(scope), "spec": asdict(definition),
                                 "sources": [asdict(b) for b in _source_bindings(records)],
                                 "parents": [asdict(p.binding) for p in parents]})
                    cached = self._get(key) if spec.reusable else None
                    if cached is not None:
                        text, receipt = cached
                        if (not self.verify_receipt(receipt, text) or receipt.computation_hash != key
                                or receipt.scope != scope or receipt.spec != definition or receipt.sources != bindings
                                or receipt.parents != tuple(p.binding for p in parents)):
                            raise _Stopped("cache_integrity")
                        usage, status = ResourceUse(0, 0, 0, 0, 0), "reused"
                    else:
                        if calls >= budget.compute_calls:
                            raise _Stopped("compute_budget")
                        inputs = ComputeInput(scope, spec, records, parents)
                        adapter = self._adapters[spec.operation_revision]
                        check(spec)
                        calls += 1
                        usage = None
                        try:
                            output = adapter.compute(inputs)
                        except Exception:
                            raise _Stopped("compute_failed") from None
                        if type(output) is not ComputeOutput:
                            raise _Stopped("invalid_output")
                        usage = output.usage
                        check(spec)
                        try:
                            verified = adapter.verify(inputs, output.text) is True
                        except Exception:
                            verified = False
                        check(spec)
                        if not verified:
                            raise _Stopped("verification_failed")
                        text = output.text
                        created = self._clock()
                        timestamp(created)
                        unsigned = ComputationReceipt(scope, definition, key, bindings, tuple(p.binding for p in parents),
                                                      hashlib.sha256(text.encode()).hexdigest(), created, "0" * 64)
                        receipt = ComputationReceipt(scope, definition, key, bindings, unsigned.parents,
                                                     unsigned.result_hash, created, self._signature(unsigned))
                        staged.append((text, receipt))
                        status = "computed"
                    total_bytes += len(text.encode()) + len(canonical(asdict(receipt)).encode())
                    if total_bytes > budget.result_bytes:
                        raise _Stopped("result_budget")
                    check(spec)
                    values[spec.key] = (text, receipt)
                except _Stopped as error:
                    status, receipt = str(error), None
                    raise
                finally:
                    elapsed = (time.perf_counter() - step_start) * 1000
                    usage = ResourceUse(usage.input_tokens, usage.output_tokens, usage.model_calls,
                                        usage.tool_calls, elapsed, usage.peak_ram_bytes,
                                        usage.peak_vram_bytes, usage.energy_joules) if usage else ResourceUse(None, None, None, None, elapsed)
                    steps.append(ComputeStep(spec.key, status, receipt, usage))
            # Revalidate every active operation, not just the final target.
            for spec in order:
                check(spec)
            check()
            with self._lock:
                if self._closed or generation != self._generation:
                    raise _Stopped("invalidated")
                for text, receipt in staged:
                    if receipt.spec.reusable:
                        self._put(text, receipt)
            return finish("complete")
        except _Stopped as error:
            return finish(str(error))
        except PermissionError:
            return finish("unauthorized")
        except ValueError:
            return finish("state_invalid")
        finally:
            self._flight.release()

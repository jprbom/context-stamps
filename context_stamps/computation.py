"""Exact computation identities and bounded host-owned result reuse.

Copyright (c) 2026 Prashant Jagtap. MIT License.
No semantic hash is used as proof of identical computation or authorization.
"""

from __future__ import annotations

import hashlib
import json
import math
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation, localcontext

from .security import bounded_text, identifier


@dataclass(frozen=True)
class ComputationIdentity:
    """The trusted host supplies all IDs, digests and current permission scope.

    `request` must include every generation parameter, seed and external input;
    source values bind both revision and content. Side-effecting or time-dependent
    computations must not be cached unless their complete state is represented.
    """

    tenant: str
    principal: str
    model: str
    prompt: str
    tool: str
    policy: str
    request: str
    sources: tuple[tuple[str, str, str], ...]

    def __post_init__(self):
        for field in (self.tenant, self.principal, self.model, self.prompt, self.tool, self.policy):
            identifier(field)
        _digest(self.request)
        if not isinstance(self.sources, tuple) or not 1 <= len(self.sources) <= 128:
            raise ValueError("one to 128 immutable source bindings required")
        seen = set()
        for binding in self.sources:
            if not isinstance(binding, tuple) or len(binding) != 3:
                raise ValueError("source requires ID, revision and digest")
            key, revision, digest = binding
            identifier(key)
            identifier(revision)
            _digest(digest)
            if key in seen:
                raise ValueError("duplicate source binding")
            seen.add(key)
        object.__setattr__(self, "sources", tuple(sorted(self.sources)))

    @property
    def digest(self):
        return hashlib.sha256(json.dumps(self.__dict__, sort_keys=True,
                                        separators=(",", ":")).encode()).hexdigest()


def _digest(value):
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError("SHA-256 hexadecimal digest required")


class ComputationCache:
    """In-process LRU with bounded bytes and TTL; not a cross-tenant auth service."""

    def __init__(self, capacity=128, ttl_seconds=300, max_bytes=4 * 1024 * 1024):
        if (type(capacity) is not int or not 1 <= capacity <= 4096
                or type(max_bytes) is not int or not 1 <= max_bytes <= 64 * 1024 * 1024
                or type(ttl_seconds) not in (int, float) or not math.isfinite(ttl_seconds)
                or not 0 < ttl_seconds <= 86400):
            raise ValueError("invalid cache bounds")
        self.capacity, self.ttl_seconds, self.max_bytes = capacity, ttl_seconds, max_bytes
        self._items = OrderedDict()
        self._bytes = 0
        self._lock = threading.RLock()

    def _remove(self, key):
        _, text = self._items.pop(key)
        self._bytes -= len(text.encode("utf-8"))

    def _expire(self):
        now = time.monotonic()
        for key, (deadline, _) in list(self._items.items()):
            if now >= deadline:
                self._remove(key)

    def put(self, identity, result):
        if not isinstance(identity, ComputationIdentity):
            raise TypeError("typed computation identity required")
        bounded_text(result)
        size = len(result.encode("utf-8"))
        if size > self.max_bytes:
            raise ValueError("result exceeds cache byte budget")
        key = identity.digest
        with self._lock:
            self._expire()
            if key in self._items:
                self._remove(key)
            while self._items and (len(self._items) >= self.capacity or self._bytes + size > self.max_bytes):
                self._remove(next(iter(self._items)))
            self._items[key] = (time.monotonic() + self.ttl_seconds, result)
            self._bytes += size

    def get(self, identity):
        if not isinstance(identity, ComputationIdentity):
            raise TypeError("typed computation identity required")
        with self._lock:
            self._expire()
            value = self._items.get(identity.digest)
            if value is None:
                return None
            self._items.move_to_end(identity.digest)
            return value[1]


def exact_decimal(operation, left, right):
    """Bounded decimal arithmetic, with no eval and no implicit binary rounding.

    Host-normalized units required. Division is rejected if it is inexact at
    64-digit precision; this function does not turn approximate data into fact.
    """
    from decimal import Inexact

    if operation not in ("add", "subtract", "multiply", "divide", "compare"):
        raise ValueError("unsupported numeric operation")
    numbers = []
    for value in (left, right):
        if not isinstance(value, str) or not 1 <= len(value) <= 64:
            raise ValueError("bounded decimal strings required")
        try:
            number = Decimal(value)
        except InvalidOperation as error:
            raise ValueError("invalid decimal") from error
        if not number.is_finite() or abs(number.adjusted()) > 32:
            raise ValueError("non-finite or out-of-range decimal")
        numbers.append(number)
    a, b = numbers
    with localcontext() as ctx:
        ctx.prec = 64
        ctx.traps[Inexact] = True
        try:
            if operation == "compare":
                return str((a > b) - (a < b))
            result = {"add": lambda: a + b, "subtract": lambda: a - b,
                      "multiply": lambda: a * b, "divide": lambda: a / b}[operation]()
        except (ArithmeticError, InvalidOperation) as error:
            raise ValueError("operation is undefined or requires rounding") from error
        return format(result, "f")

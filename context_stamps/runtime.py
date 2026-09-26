"""A bounded, host-controlled context workflow with explicit verification.

Copyright (c) 2026 Prashant Jagtap. MIT License.
One instance belongs to one authenticated tenant/principal/policy. External
callbacks are trusted adapters; they must enforce their own execution timeout.
"""

import hashlib
import math
import threading
import time
from collections import deque
from dataclasses import dataclass

from .computation import ComputationCache, ComputationIdentity
from .facet_compiler import FacetCompiler
from .security import bounded_text, identifier
from .session import ContextSession
from .workflow import ContextNode


@dataclass(frozen=True)
class RuntimeBudget:
    bytes: int = 8192
    tokens: int | None = None
    milliseconds: float = 1000
    iterations: int = 3

    def __post_init__(self):
        if (type(self.bytes) is not int or not 0 <= self.bytes <= 1048576
                or self.tokens is not None and (type(self.tokens) is not int or not 0 <= self.tokens <= 262144)
                or type(self.iterations) is not int or not 1 <= self.iterations <= 8
                or type(self.milliseconds) not in (int, float) or not math.isfinite(self.milliseconds)
                or not 0 < self.milliseconds <= 120000):
            raise ValueError("invalid runtime budget")


@dataclass(frozen=True)
class ExpertRequest:
    query: str
    eligible: tuple[str, ...]
    facets: tuple[tuple[str, str], ...]
    limit: int
    deadline: float


@dataclass(frozen=True)
class ContextExpert:
    name: str
    retrieve: object
    estimated_ms: float
    approved_scopes: tuple[str, ...] = ()

    def __post_init__(self):
        identifier(self.name)
        if (not callable(self.retrieve) or type(self.estimated_ms) not in (float, int)
                or not math.isfinite(self.estimated_ms) or not 0 <= self.estimated_ms <= 120000
                or len(self.approved_scopes) > 128):
            raise ValueError("bounded callable expert required")
        object.__setattr__(self, "approved_scopes", tuple(self.approved_scopes))
        for scope in self.approved_scopes:
            identifier(scope)


@dataclass(frozen=True)
class Verification:
    complete: bool
    missing: tuple[str, ...] = ()

    def __post_init__(self):
        if type(self.complete) is not bool or len(self.missing) > 100 or self.complete and self.missing:
            raise ValueError("invalid verification")
        object.__setattr__(self, "missing", tuple(self.missing))
        for key in self.missing:
            identifier(key)


@dataclass(frozen=True)
class PreparedContext:
    status: str
    text: str
    sources: tuple[str, ...]
    receipt: bytes | None
    tokens: int | None
    reused: bool
    iterations: int
    expert: str
    reason: str
    elapsed_ms: float


class ContextRuntime:
    """Compose retrieval, exact dependency closure, budgets and verified reuse.

    A short budget never authorizes a lower-quality expert. Unknown/unapproved
    routes use the designated baseline or abstain. Model-generated text never
    mutates sources, permissions, expert registrations or model weights.
    """

    def __init__(self, *, tenant, principal, role, policy, token_counter=None):
        for value in (tenant, principal, role, policy):
            identifier(value)
        if token_counter is not None and not callable(token_counter):
            raise ValueError("token_counter must be callable")
        self.tenant, self.principal, self.role, self.policy = tenant, principal, role, policy
        self._tokens = token_counter
        self._session, self._cache = ContextSession(), ComputationCache()
        self._nodes, self._revoked = {}, set()
        self._epoch, self._lock = 0, threading.RLock()
        self._outcomes = deque(maxlen=128)

    def put(self, node):
        if not isinstance(node, ContextNode):
            raise ValueError("ContextNode required")
        with self._lock:
            self._session.put(node)
            self._nodes[node.key] = node
            self._revoked.discard(node.key)
            self._epoch += 1

    def link(self, source, target, kind, *, provenance):
        with self._lock:
            self._session.link(source, target, kind, provenance=provenance)
            self._epoch += 1

    def invalidate(self, source):
        identifier(source)
        with self._lock:
            if source not in self._nodes:
                raise ValueError("unknown source")
            self._revoked.add(source)
            self._epoch += 1

    def _versions(self):
        return {k: n.revision for k, n in self._nodes.items() if k not in self._revoked and self.role in n.roles}

    def _count(self, text):
        if self._tokens is None:
            return None
        count = self._tokens(text)
        if type(count) is not int or count < 0:
            raise ValueError("token counter must return a nonnegative integer")
        return count

    def resolve(self, receipt, *, budget=RuntimeBudget()):
        with self._lock:
            packet = self._session.resolve(receipt, role=self.role, revisions=self._versions(), budget_bytes=budget.bytes)
            if packet.status == "complete" and budget.tokens is not None:
                if self._tokens is None:
                    raise ValueError("an explicit tokenizer is required for a token budget")
                if self._count(packet.text) > budget.tokens:
                    from .workflow import Handoff
                    return Handoff("insufficient", "", (), 0, "token_budget")
            return packet

    def prepare_context(self, query, *, eligible, experts, baseline, scope, verifier,
                        choose=None, exact_key=None, limit=5, budget=RuntimeBudget()):
        bounded_text(query, 16384)
        identifier(scope)
        if (not isinstance(budget, RuntimeBudget) or not callable(verifier)
                or choose is not None and not callable(choose)
                or type(limit) is not int or not 1 <= limit <= 100
                or not isinstance(eligible, (tuple, list)) or len(eligible) > 1000
                or len(set(eligible)) != len(eligible)
                or not isinstance(experts, (tuple, list)) or not 1 <= len(experts) <= 16
                or any(not isinstance(e, ContextExpert) for e in experts)):
            raise ValueError("bounded experts, candidates and verifier required")
        for key in eligible:
            identifier(key)
        registered = {e.name: e for e in experts}
        if baseline not in registered or len(registered) != len(experts):
            raise ValueError("unique experts and registered baseline required")
        if budget.tokens is not None and self._tokens is None:
            raise ValueError("an explicit tokenizer is required for a token budget")
        started = time.perf_counter()
        deadline = time.monotonic() + budget.milliseconds / 1000
        iterations, route = 0, baseline

        def failure(reason):
            return PreparedContext("insufficient", "", (), None, None, False, iterations,
                                   route, reason, (time.perf_counter() - started) * 1000)

        with self._lock:
            epoch = self._epoch
            versions = self._versions()
            allowed = tuple(k for k in eligible if k in versions)
            if not allowed:
                return failure("unavailable_context")
            if exact_key is not None:
                identifier(exact_key)
                if exact_key not in allowed:
                    return failure("unavailable_context")
                roots, route = [exact_key], "exact"
            else:
                facets = tuple(FacetCompiler().compile(query).facets.items())
                request = ExpertRequest(query, allowed, facets, min(limit, len(allowed)), deadline)
                proposal = baseline if choose is None else choose(request)
                candidate = registered.get(proposal)
                if candidate is not None and (proposal == baseline or scope in candidate.approved_scopes):
                    route = proposal
                expert = registered[route]
                if time.monotonic() + expert.estimated_ms / 1000 > deadline:
                    return failure("latency_budget")
                roots = expert.retrieve(request)
                if (not isinstance(roots, (tuple, list)) or not 1 <= len(roots) <= request.limit
                        or any(k not in allowed for k in roots) or len(set(roots)) != len(roots)):
                    return failure("invalid_expert_result")
                roots = list(roots)
            seen = set()
            for iterations in range(1, budget.iterations + 1):
                if time.monotonic() > deadline:
                    return failure("latency_budget")
                if epoch != self._epoch:
                    return failure("context_changed")
                packet, receipt, reused = self._session.issue(roots, role=self.role, revisions=self._versions(), budget_bytes=budget.bytes)
                if packet.status != "complete":
                    return failure(packet.reason)
                tokens = self._count(packet.text)
                if budget.tokens is not None and tokens > budget.tokens:
                    return failure("token_budget")
                verdict = verifier(packet)
                if not isinstance(verdict, Verification):
                    return failure("invalid_verifier_result")
                if epoch != self._epoch:
                    return failure("context_changed")
                if time.monotonic() > deadline:
                    return failure("latency_budget")
                if verdict.complete:
                    return PreparedContext("complete", packet.text, packet.sources, receipt, tokens, reused,
                        iterations, route, "verified_context", (time.perf_counter() - started) * 1000)
                if any(key not in allowed for key in verdict.missing):
                    return failure("unavailable_context")
                state = (packet.sources, verdict.missing)
                expanded = sorted(set(roots) | set(verdict.missing))
                if state in seen or set(expanded) == set(roots):
                    return failure("no_progress")
                if len(expanded) > 100:
                    return failure("source_budget")
                seen.add(state)
                roots = expanded
            return failure("iteration_budget")

    def run_verified(self, prepared, *, request_digest, model, prompt, tool, verifier_revision, compute, verify):
        """Reuse only complete exact bindings; validate every returned result.

        The host includes every generation setting in request_digest and binds
        verifier changes to verifier_revision. Callbacks must be pure/idempotent.
        This method does not impose a hard timeout on an external model call.
        """
        if not isinstance(prepared, PreparedContext) or prepared.status != "complete" or prepared.receipt is None:
            raise ValueError("complete prepared context required")
        identifier(verifier_revision)
        identifier(tool)
        if not callable(compute) or not callable(verify):
            raise ValueError("explicit compute and verifier callbacks required")
        with self._lock:
            packet = self.resolve(prepared.receipt, budget=RuntimeBudget(bytes=1048576))
            if packet.status != "complete" or packet.text != prepared.text or packet.sources != prepared.sources:
                return {"status": "insufficient", "result": "", "reused": False}
            if len(packet.sources) > 128:
                return {"status": "insufficient", "result": "", "reused": False}
            epoch = self._epoch
            identity = ComputationIdentity(tenant=self.tenant, principal=self.principal, policy=self.policy,
                request=request_digest, model=model, prompt=prompt,
                tool=hashlib.sha256((tool + "\0" + verifier_revision).encode()).hexdigest(),
                sources=tuple((key, self._nodes[key].revision, self._nodes[key].digest) for key in packet.sources))
            value = self._cache.get(identity)
            reused = value is not None
            if value is None:
                value = compute(packet.text)
            bounded_text(value)
            verified = verify(value, packet.text) is True
            if not verified or epoch != self._epoch:
                return {"status": "unverified", "result": "", "reused": False}
            if not reused:
                self._cache.put(identity, value)
            self.record_outcome(identity.digest, verified=True, reused=reused)
            return {"status": "verified", "result": value, "reused": reused}

    def record_outcome(self, request_id, *, verified, reused=False):
        identifier(request_id)
        if type(verified) is not bool or type(reused) is not bool:
            raise ValueError("explicit boolean outcome required")
        with self._lock:
            self._outcomes.append(dict(request_id=request_id, verified=verified, reused=reused, epoch=self._epoch))

    def outcomes(self):
        with self._lock:
            return tuple(dict(row) for row in self._outcomes)

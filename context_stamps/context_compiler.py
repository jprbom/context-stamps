"""Authorization-first, model-aware compilation of explicit evidence requirements.

Copyright (c) 2026 Prashant Jagtap. MIT License.
Exact optimization is bounded to 16 candidate roots. Larger sets use a declared
greedy policy. Neither exact coverage nor declared dependencies prove truth.
"""

from __future__ import annotations

import hashlib
import itertools
import math
import time
from dataclasses import asdict, dataclass, replace

from .context_state import ContextSnapshot, EvidenceRequirement, canonical, typed_tuple
from .security import bounded_text, identifier


@dataclass(frozen=True)
class ContextTask:
    key: str
    question: str
    requirements: tuple[EvidenceRequirement, ...]

    def __post_init__(self):
        identifier(self.key)
        bounded_text(self.question, 16384)
        typed_tuple(self.requirements, EvidenceRequirement, 12)
        if not self.requirements or len({r.name for r in self.requirements}) != len(self.requirements):
            raise ValueError("one to twelve unique named requirements required")


@dataclass(frozen=True)
class ModelProfile:
    model_revision: str
    formatter_revision: str = "canonical-json-v1"
    tokenizer_revision: str | None = None
    count_tokens: object = None
    render: object = None
    verify_render: object = None

    def __post_init__(self):
        identifier(self.model_revision)
        identifier(self.formatter_revision)
        if (self.count_tokens is None) != (self.tokenizer_revision is None):
            raise ValueError("token counter and pinned tokenizer identity are required together")
        if self.count_tokens is not None:
            identifier(self.tokenizer_revision)
            if not callable(self.count_tokens):
                raise ValueError("token counter must be callable")
        if self.render is not None and not callable(self.render):
            raise ValueError("formatter must be callable")
        if self.render is not None and not callable(self.verify_render):
            raise ValueError("custom formatters require an explicit evidence-preservation verifier")
        if self.render is None and self.verify_render is not None:
            raise ValueError("render verifier requires a formatter")


@dataclass(frozen=True)
class CompileBudget:
    bytes: int = 65536
    tokens: int | None = None
    milliseconds: float = 1000
    combinations: int = 65536

    def __post_init__(self):
        if type(self.bytes) is not int or not 0 <= self.bytes <= 1048576:
            raise ValueError("invalid byte budget")
        if self.tokens is not None and (type(self.tokens) is not int or not 0 <= self.tokens <= 262144):
            raise ValueError("invalid token budget")
        if (type(self.milliseconds) not in (float, int) or not math.isfinite(self.milliseconds)
                or not 0 < self.milliseconds <= 120000):
            raise ValueError("invalid deadline")
        if type(self.combinations) is not int or not 1 <= self.combinations <= 65536:
            raise ValueError("invalid search budget")


@dataclass(frozen=True)
class ConflictPolicy:
    revision: str
    order: tuple[str, ...] = ()

    def __post_init__(self):
        identifier(self.revision)
        typed_tuple(self.order, str, 5)
        if not set(self.order) <= {"authority", "reliability", "effective_at", "observed_at", "kind"}:
            raise ValueError("unsupported conflict criteria")

    def priority(self, node):
        fields = {"authority": node.authority, "reliability": -1 if node.reliability is None else node.reliability,
                  "effective_at": node.temporal.start, "observed_at": node.temporal.observed_at,
                  "kind": {"POLICY": 4, "FACT": 3, "OBSERVATION": 2, "DERIVED_RESULT": 1}.get(node.kind, 0)}
        return tuple(fields[key] for key in self.order)


@dataclass(frozen=True)
class ConflictResolution:
    requirement: str
    status: str
    candidates: tuple[str, ...]
    selected_value: tuple[str, str | None] | None
    policy_revision: str


@dataclass(frozen=True)
class CompiledContext:
    status: str
    reason: str
    text: str
    snapshot: ContextSnapshot | None
    task: ContextTask
    profile: ModelProfile
    tokens: int | None
    bytes: int
    missing: tuple[str, ...]
    conflicts: tuple[ConflictResolution, ...]
    optimal: bool
    search_steps: int
    deadline: float
    elapsed_ms: float
    binding: str = ""


def compiled_material(context):
    return canonical({"task": asdict(context.task), "model": context.profile.model_revision,
        "formatter": context.profile.formatter_revision, "tokenizer": context.profile.tokenizer_revision,
        "text_sha256": hashlib.sha256(context.text.encode()).hexdigest(), "tokens": context.tokens,
        "bytes": context.bytes, "deadline": context.deadline, "reason": context.reason,
        "conflicts": [asdict(c) for c in context.conflicts], "optimal": context.optimal,
        "search_steps": context.search_steps})


def default_render(task, profile, records):
    return canonical({"task": task.key, "question": task.question, "model_revision": profile.model_revision,
        "formatter_revision": profile.formatter_revision, "evidence": [
            {"source": r.node.key, "revision": r.node.revision, "fingerprint": r.node.ref.fingerprint,
             "kind": r.node.kind, "modality": r.node.modality, "provenance": r.node.provenance,
             "validation_revision": r.node.validation_revision, "temporal": asdict(r.node.temporal),
             "transaction_time": r.transaction_time, "claims": [asdict(c) for c in r.node.claims],
             "dependencies": [asdict(ref) for ref in r.node.dependencies], "untrusted_source_text": r.node.text}
            for r in records]})


class ContextCompiler:
    """Compose versioned state, conflict handling, closure and serialized budgets.

    Callbacks are trusted deterministic local formatters/tokenizers. Their hard
    cancellation is not provided here. Timeout/mutation checks reject late output.
    """

    def __init__(self, state):
        self.state = state

    def compile(self, task, *, scope, at, known_at, profile, budget=CompileBudget(),
                conflict_policy=ConflictPolicy("strict-v1")):
        if (type(task) is not ContextTask or type(profile) is not ModelProfile
                or type(budget) is not CompileBudget or type(conflict_policy) is not ConflictPolicy):
            raise ValueError("typed task, model profile, budget and conflict policy required")
        if budget.tokens is not None and profile.count_tokens is None:
            raise ValueError("actual tokenizer required for token budget")
        start = time.perf_counter()
        deadline = time.monotonic() + budget.milliseconds / 1000
        steps, conflicts = 0, []

        def failure(reason, missing=()):
            visible_conflicts = (tuple(conflicts) if reason == "conflicting_context"
                                 and self.state.is_current(snapshot) else ())
            return CompiledContext("insufficient", reason, "", None, task, profile, None, 0,
                                   tuple(missing), visible_conflicts, False, steps, deadline, (time.perf_counter() - start) * 1000)

        try:
            snapshot = self.state.snapshot(scope, at=at, known_at=known_at)
        except PermissionError:
            return failure("unavailable_context")
        records = {r.node.ref: r for r in snapshot.records}
        selected_values, coverage = {}, {}
        missing = []
        for i, requirement in enumerate(task.requirements):
            matching = [(r, c) for r in records.values() for c in r.node.claims if requirement.accepts(r.node, c)]
            values = {(c.status, c.value) for _, c in matching}
            if not values:
                missing.append(requirement.name)
                continue
            if len(values) > 1:
                top = max((conflict_policy.priority(r.node) for r, _ in matching), default=())
                winners = {(c.status, c.value) for r, c in matching if conflict_policy.priority(r.node) == top}
                resolved = bool(conflict_policy.order) and len(winners) == 1
                winner = next(iter(winners)) if resolved else None
                conflicts.append(ConflictResolution(requirement.name, "RESOLVED" if resolved else "UNRESOLVED",
                    tuple(sorted({r.node.key for r, _ in matching})), winner, conflict_policy.revision))
                if not resolved:
                    return failure("conflicting_context")
                values = winners
            selected_values[requirement.name] = next(iter(values))
            for r, c in matching:
                if (c.status, c.value) in values:
                    coverage[r.node.ref] = coverage.get(r.node.ref, 0) | (1 << i)
        if missing:
            return failure("missing_evidence", missing)

        def closure(root):
            chosen, pending = set(), [root]
            while pending:
                ref = pending.pop()
                if ref in chosen:
                    continue
                if ref not in records:
                    return None
                chosen.add(ref)
                node = records[ref].node
                for requirement in task.requirements:
                    if any(requirement.accepts(node, c) and (c.status, c.value) != selected_values[requirement.name]
                           for c in node.claims):
                        return None
                pending.extend(node.dependencies)
            return frozenset(chosen)

        roots = sorted(coverage, key=lambda ref: (ref.key, ref.revision))
        closures = [(ref, closure(ref)) for ref in roots]
        closures = [(ref, members) for ref, members in closures if members is not None]
        full = (1 << len(task.requirements)) - 1

        def mask(members):
            result = 0
            for ref in members:
                result |= coverage.get(ref, 0)
            return result

        if mask(set().union(*(members for _, members in closures))) != full:
            return failure("unavailable_context")

        def pack(members):
            if time.monotonic() >= deadline:
                raise TimeoutError("compile deadline")
            selected = tuple(sorted((records[ref] for ref in members), key=lambda r: (r.node.key, r.node.revision)))
            text = (profile.render or default_render)(task, profile, selected)
            bounded_text(text, 1048576)
            if profile.render is not None and profile.verify_render(task, selected, text) is not True:
                raise ValueError("formatter failed evidence-preservation check")
            tokens = profile.count_tokens(text) if profile.count_tokens else None
            if tokens is not None and (type(tokens) is not int or tokens < 0):
                raise ValueError("token counter must return a nonnegative integer")
            size = len(text.encode("utf-8"))
            cost = (tokens if tokens is not None else size, size, len(members), tuple(sorted((r.key, r.revision) for r in members)))
            return cost, text, tokens, size

        exact = len(closures) <= 16
        if exact:
            choices = (frozenset().union(*(closures[i][1] for i in group))
                       for size in range(1, len(closures) + 1)
                       for group in itertools.combinations(range(len(closures)), size))
        else:
            # Deterministic coverage per marginal serialized cost, not a global optimum.
            def greedy():
                members, pending = frozenset(), list(closures)
                while pending and mask(members) != full:
                    options = []
                    _, _, current_tokens, current_size = pack(members)
                    current_cost = current_tokens if current_tokens is not None else current_size
                    for ref, group in pending:
                        merged = members | group
                        gain = (mask(merged) ^ mask(members)).bit_count()
                        if gain:
                            _, _, tokens, size = pack(merged)
                            marginal_cost = (tokens if tokens is not None else size) - current_cost
                            options.append((-gain / max(1, marginal_cost), ref.key, ref.revision, group))
                    if not options:
                        return
                    chosen = min(options)[3]
                    members |= chosen
                    pending = [(ref, group) for ref, group in pending if not group <= members]
                yield members
            choices = greedy()
        best, seen, exhaustive = None, set(), True
        try:
            for members in choices:
                if time.monotonic() >= deadline:
                    return failure("latency_budget")
                if steps >= budget.combinations:
                    exhaustive = False
                    break
                steps += 1
                if members in seen or mask(members) != full:
                    continue
                seen.add(members)
                cost, text, tokens, size = pack(members)
                if size > budget.bytes or budget.tokens is not None and tokens > budget.tokens:
                    continue
                if best is None or cost < best[0]:
                    best = cost, text, tokens, size, members
        except TimeoutError:
            return failure("latency_budget")
        except Exception:
            return failure("invalid_adapter_result")
        if time.monotonic() >= deadline:
            return failure("latency_budget")
        if not self.state.is_current(snapshot):
            return failure("state_changed")
        if best is None:
            return failure("search_budget" if not exhaustive else "context_budget")
        _, text, tokens, size, members = best
        try:
            selected = self.state.subset(snapshot, tuple(sorted(members, key=lambda r: (r.key, r.revision))))
        except ValueError:
            return failure("state_changed")
        result = CompiledContext("complete", "verified_requirements", text, selected, task, profile, tokens,
                                 size, (), tuple(conflicts), exact and exhaustive, steps, deadline,
                                 (time.perf_counter() - start) * 1000)
        try:
            return replace(result, binding=self.state.seal(selected, compiled_material(result)))
        except ValueError:
            return failure("state_changed")

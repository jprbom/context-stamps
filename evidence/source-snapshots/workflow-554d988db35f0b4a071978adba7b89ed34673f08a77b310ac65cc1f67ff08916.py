"""Explicit, application-owned context relationships and fail-closed handoffs.

In-memory reference implementation, not an authentication or graph database service.
Copyright (c) 2026 Prashant Jagtap. MIT License.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Mapping

from stamps import content_digest

from .security import bounded_text, identifier
from .spherical import SphericalStamp

RELATIONS = {"depends_on", "supports", "contradicts", "produced_by", "related_to"}


@dataclass(frozen=True)
class ContextNode:
    key: str
    text: str
    revision: str
    roles: frozenset[str]
    stamp: SphericalStamp | None = None

    def __post_init__(self):
        identifier(self.key)
        identifier(self.revision)
        bounded_text(self.text)
        if isinstance(self.roles, str):
            raise ValueError("roles must be a collection")
        object.__setattr__(self, "roles", frozenset(self.roles))
        if not 1 <= len(self.roles) <= 32:
            raise ValueError("one to 32 allowed roles required")
        for role in self.roles:
            identifier(role)
        if self.stamp is not None and not isinstance(self.stamp, SphericalStamp):
            raise ValueError("invalid stamp")

    @property
    def digest(self):
        return content_digest(self.text)


@dataclass(frozen=True)
class Relationship:
    source: str
    target: str
    kind: str
    provenance: str
    source_digest: str
    target_digest: str
    source_revision: str
    target_revision: str


@dataclass(frozen=True)
class Handoff:
    status: str
    text: str
    sources: tuple[str, ...]
    units: int
    reason: str


class ContextGraph:
    def __init__(self):
        self._nodes: dict[str, ContextNode] = {}
        self._edges: list[Relationship] = []

    def put(self, node: ContextNode):
        if not isinstance(node, ContextNode):
            raise ValueError("ContextNode required")
        if node.key not in self._nodes and len(self._nodes) >= 1000:
            raise ValueError("node limit reached")
        self._nodes[node.key] = node

    def link(self, source: str, target: str, kind: str, *, provenance: str):
        identifier(provenance)
        if kind not in RELATIONS or source == target:
            raise ValueError("invalid relationship")
        if source not in self._nodes or target not in self._nodes:
            raise ValueError("relationship endpoints must exist")
        if len(self._edges) >= 4096:
            raise ValueError("relationship limit reached")
        a, b = self._nodes[source], self._nodes[target]
        edge = Relationship(source, target, kind, provenance, a.digest, b.digest, a.revision, b.revision)
        # Revalidation replaces the previous binding for the same declared relationship.
        self._edges = [e for e in self._edges if (e.source, e.target, e.kind) != (source, target, kind)]
        self._edges.append(edge)

    def _current_edge(self, edge):
        a, b = self._nodes[edge.source], self._nodes[edge.target]
        return (a.digest, b.digest, a.revision, b.revision) == (
            edge.source_digest, edge.target_digest, edge.source_revision, edge.target_revision)

    def affected(self, changed):
        """Reverse dependency closure, including changed nodes; cycles terminate."""
        if isinstance(changed, str) or len(changed) > 1000:
            raise ValueError("bounded collection of changed IDs required")
        found = set(changed)
        if not found <= self._nodes.keys():
            raise ValueError("unknown changed node")
        pending = list(found)
        while pending:
            key = pending.pop()
            for edge in self._edges:
                if edge.kind == "depends_on" and edge.target == key and edge.source not in found:
                    found.add(edge.source)
                    pending.append(edge.source)
        return tuple(sorted(found))

    def handoff(self, roots, *, role: str, revisions: Mapping[str, str], budget_bytes=8192):
        """Whole dependency closure or empty packet; roles come from a trusted host."""
        identifier(role)
        if isinstance(roots, str) or not 1 <= len(roots) <= 100:
            raise ValueError("one to 100 root IDs required")
        if type(budget_bytes) is not int or not 0 <= budget_bytes <= 1048576:
            raise ValueError("budget must be between zero and 1 MiB")
        chosen, pending = set(), list(roots)
        while pending:
            key = pending.pop()
            if key in chosen:
                continue
            node = self._nodes.get(key)
            # Do not disclose which condition failed or hidden node identifiers.
            if node is None or role not in node.roles or revisions.get(key) != node.revision:
                return Handoff("insufficient", "", (), 0, "unavailable_context")
            chosen.add(key)
            for edge in self._edges:
                if edge.source == key and edge.kind == "depends_on":
                    if not self._current_edge(edge):
                        return Handoff("insufficient", "", (), 0, "unavailable_context")
                    pending.append(edge.target)
        for edge in self._edges:
            if edge.kind == "contradicts" and edge.source in chosen and edge.target in chosen:
                return Handoff("insufficient", "", (), 0, "conflicting_context")
        keys = tuple(sorted(chosen))
        text = "\n\n".join(json.dumps({"source": key, "revision": self._nodes[key].revision,
                                    "sha256": self._nodes[key].digest}, sort_keys=True)
                             + "\n" + self._nodes[key].text for key in keys)
        size = len(text.encode("utf-8"))
        if size > budget_bytes:
            return Handoff("insufficient", "", (), 0, "budget_exceeded")
        return Handoff("complete", text, keys, size, "dependency_closure")

    def retrieve(self, query: SphericalStamp, *, role: str, revisions: Mapping[str, str],
                 weights=None, limit=5):
        identifier(role)
        if type(limit) is not int or not 0 <= limit <= 100:
            raise ValueError("invalid limit")
        results = []
        for key, node in self._nodes.items():
            if role not in node.roles or revisions.get(key) != node.revision or node.stamp is None:
                continue
            values = query.compare(node.stamp)
            results.append({"source": key, "facets": values, "score": query.score(node.stamp, weights)})
        return sorted(results, key=lambda r: (-r["score"], r["source"]))[:limit]

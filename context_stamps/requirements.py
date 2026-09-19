"""Exact structured requirements; claim extraction belongs to the trusted application."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from typing import Mapping

from .security import bounded_text
from .selection import rank_candidates


def rank_candidates_safe(query, documents, scores, *, limit=10, policy="baseline", **experimental):
    """Preserve score ranking unless the caller explicitly opts into the legacy heuristic."""
    if policy == "experimental_coverage":
        return rank_candidates(query, documents, scores, limit=limit, **experimental)
    if policy != "baseline" or experimental:
        raise ValueError("unknown policy or experimental settings supplied to baseline")
    bounded_text(query, 16384)
    if len(documents) != len(scores) or len(scores) > 1000 or type(limit) is not int or not 0 <= limit <= 100:
        raise ValueError("invalid candidate count or limit")
    for document in documents:
        bounded_text(document)
    if not all(isinstance(s, (int, float)) and math.isfinite(s) for s in scores):
        raise ValueError("scores must be finite numbers")
    return sorted(range(len(scores)), key=lambda i: -scores[i])[:limit]


@dataclass(frozen=True)
class Requirement:
    subject: str
    attribute: str

    def __post_init__(self):
        for value in (self.subject, self.attribute):
            if not isinstance(value, str) or not value or len(value.encode("utf-8")) > 512:
                raise ValueError("subject and attribute must be nonempty bounded strings")
            if any(ord(c) < 32 for c in value):
                raise ValueError("control characters are not allowed")


@dataclass(frozen=True)
class Claim:
    source: str
    digest: str
    subject: str
    attribute: str
    value: str

    def __post_init__(self):
        Requirement(self.subject, self.attribute)
        Requirement(self.source, self.value)
        if (
            not isinstance(self.digest, str)
            or len(self.digest) != 64
            or any(c not in "0123456789abcdef" for c in self.digest)
        ):
            raise ValueError("claim digest must be a SHA-256 hex string")


@dataclass
class StructuredSelection:
    status: str
    text: str
    selected: list[str]
    units: int
    budget: int
    missing: list[dict]
    conflicts: list[dict]
    decisions: list[dict]

    def to_dict(self):
        return asdict(self)


def select_structured(memory, query, *, requirements, claims, revisions, budget_bytes=2048):
    """Find a minimum-byte complete packet for at most 12 exact requirements.

    Claims are trusted application metadata, bound to exact source digests. This
    function does not extract claims, verify their truth, or authorize source text.
    String values and identifiers compare exactly; callers must normalize units.
    All supplied requirements are hard constraints. Conflicts fail closed.
    """
    if not isinstance(revisions, Mapping):
        raise ValueError("an explicit current revision map is required")
    if type(budget_bytes) is not int or not 0 <= budget_bytes <= 1048576:
        raise ValueError("invalid byte budget")
    if not isinstance(requirements, (list, tuple)) or not 1 <= len(requirements) <= 12:
        raise ValueError("provide 1 to 12 structured requirements")
    if not all(isinstance(r, Requirement) for r in requirements) or len(set(requirements)) != len(
        requirements
    ):
        raise ValueError("requirements must be typed and unique")
    if (
        not isinstance(claims, (list, tuple))
        or len(claims) > 4096
        or not all(isinstance(c, Claim) for c in claims)
    ):
        raise ValueError("provide at most 4096 typed claims")
    bounded_text(query, 16384)
    rows = memory._rank(query, revisions)
    decisions, eligible = [], {}
    for row in rows:
        if row["freshness"] != "current":
            decisions.append({"source": row["source"], "reason": row["freshness"]})
        else:
            eligible[row["source"]] = row
    required = {r: i for i, r in enumerate(requirements)}
    values = {r: set() for r in requirements}
    coverage = {}
    for claim in claims:
        row = eligible.get(claim.source)
        if row is None:
            continue
        if claim.digest != row["digest"]:
            decisions.append({"source": claim.source, "reason": "claim_digest_mismatch"})
            continue
        key = Requirement(claim.subject, claim.attribute)
        if key in required:
            values[key].add(claim.value)
            coverage[claim.source] = coverage.get(claim.source, 0) | (1 << required[key])
    missing = [asdict(r) for r in requirements if not values[r]]
    conflicts = [{**asdict(r), "values": sorted(values[r])} for r in requirements if len(values[r]) > 1]

    def failure(reason):
        return StructuredSelection(
            "insufficient_evidence",
            "",
            [],
            0,
            budget_bytes,
            missing,
            conflicts,
            [*decisions, {"reason": reason}],
        )

    if missing or conflicts:
        return failure("missing_claims" if missing else "conflicting_claims")
    if len(coverage) > 128:
        raise ValueError("structured selection supports at most 128 eligible candidate sources")
    chunks = {}
    # Exact additive UTF-8 cost: charge each chunk plus a separator, then remove
    # the one separator not needed at the start. Keep one cheapest state/mask.
    states = {0: (0, ())}
    for source in sorted(coverage):
        row = eligible[source]
        chunks[source] = (
            json.dumps({"source": source, "sha256": row["digest"]}, ensure_ascii=False) + "\n" + row["text"]
        )
        cost = len(chunks[source].encode("utf-8")) + 2
        for mask, (total, selected) in list(states.items()):
            combined = mask | coverage[source]
            candidate = (total + cost, (*selected, source))
            if combined == mask or candidate[0] - 2 > budget_bytes:
                continue
            if combined not in states or candidate < states[combined]:
                states[combined] = candidate
    full = (1 << len(requirements)) - 1
    if full not in states:
        return failure("budget")
    selected = list(states[full][1])
    text = "\n\n".join(chunks[source] for source in selected)
    decisions.extend(
        {"source": source, "reason": "selected" if source in selected else "not_needed"}
        for source in coverage
    )
    return StructuredSelection(
        "current", text, selected, len(text.encode("utf-8")), budget_bytes, [], [], decisions
    )

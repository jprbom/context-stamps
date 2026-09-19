"""Exact-first routing and optional, independently validated compact exits.

A confidence policy measures agreement with a host's precise baseline, not truth.
It is bound to an application scope (domain, encoder, schema and snapshot policy).
Copyright (c) 2026 Prashant Jagtap. MIT License.
"""

import math
from dataclasses import dataclass

from .security import identifier


@dataclass(frozen=True)
class RoutingPolicy:
    scope: str
    limit: int
    minimum_score: float
    minimum_margin: float
    enabled: bool = False

    def __post_init__(self):
        identifier(self.scope)
        if type(self.limit) is not int or not 1 <= self.limit <= 100:
            raise ValueError("limit must be between one and 100")
        if not all(math.isfinite(v) and 0 <= v <= 1 for v in (self.minimum_score, self.minimum_margin)):
            raise ValueError("score and margin thresholds must be in [0, 1]")
        if type(self.enabled) is not bool:
            raise ValueError("enabled must be boolean")

    def accepts(self, score, margin, *, scope, limit):
        return (self.enabled and self.scope == scope and self.limit == limit
                and math.isfinite(score) and math.isfinite(margin)
                and self.minimum_score <= score <= 1
                and self.minimum_margin <= margin <= 1)


def error_upper_bound(errors, count, delta=0.05):
    """One-sided exact binomial upper bound, for a fixed policy on IID trials.

    This does not protect against distribution shift, repeated threshold tuning,
    correlated queries or a baseline which is itself wrong.
    """
    if (type(count) is not int or type(errors) is not int or not 0 <= errors <= count <= 10000
            or not 0 < delta < 1):
        raise ValueError("invalid bounded binomial counts or delta")
    if not count or errors == count:
        return 1.0
    if not errors:
        return -math.expm1(math.log(delta) / count)
    lo, hi = errors / count, 1.0
    coefficients = [math.lgamma(count + 1) - math.lgamma(i + 1) - math.lgamma(count - i + 1)
                    for i in range(errors + 1)]
    for _ in range(60):
        p = (lo + hi) / 2
        terms = [c + i * math.log(p) + (count - i) * math.log1p(-p)
                 for i, c in enumerate(coefficients)]
        largest = max(terms)
        cdf = math.exp(largest) * math.fsum(math.exp(v - largest) for v in terms)
        if cdf > delta:
            lo = p
        else:
            hi = p
    return hi


def fit_routing_policy(training, validation, *, scope, limit=10, max_error=0.05,
                       min_accepted=30, delta=0.05):
    """Select thresholds on training only; certify once on disjoint validation.

    Rows contain score (kth compact score), margin (kth minus k+1th), and
    correct (whole top-k agreement with a declared precise baseline).
    No validation retuning. Insufficient evidence disables the compact exit.
    """
    if not 0 < max_error < 1 or type(min_accepted) is not int or not 1 <= min_accepted <= 10000:
        raise ValueError("invalid calibration limits")
    RoutingPolicy(scope, limit, 0, 0)
    error_upper_bound(0, 0, delta)
    for rows in (training, validation):
        if not 1 <= len(rows) <= 10000:
            raise ValueError("one to 10000 calibration rows required")
        for row in rows:
            if (type(row["correct"]) is not bool or not all(
                    math.isfinite(row[k]) and 0 <= row[k] <= 1 for k in ("score", "margin"))):
                raise ValueError("invalid calibration row")
    selected, best_count = (1.0, 1.0), -1
    for score in (0.0, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95):
        for margin in (0.0, 1 / 256, 2 / 256, 4 / 256, 8 / 256, 16 / 256):
            accepted = [r for r in training if r["score"] >= score and r["margin"] >= margin]
            errors = sum(not r["correct"] for r in accepted)
            if len(accepted) >= min_accepted and errors / len(accepted) <= max_error:
                if len(accepted) > best_count:
                    selected, best_count = (score, margin), len(accepted)
    accepted = [r for r in validation if r["score"] >= selected[0] and r["margin"] >= selected[1]]
    errors = sum(not r["correct"] for r in accepted)
    upper = error_upper_bound(errors, len(accepted), delta)
    enabled = best_count >= 0 and len(accepted) >= min_accepted and upper <= max_error
    policy = RoutingPolicy(scope, limit, *selected, enabled)
    return policy, {"training_accepted": max(0, best_count), "validation_accepted": len(accepted),
                    "validation_errors": errors, "error_upper_bound": upper, "max_error": max_error,
                    "delta": delta, "enabled": enabled}


class ProgressiveRouter:
    """Host-owned callbacks receive only the current authorized candidate IDs.

    Callbacks return ranked (ID, score) pairs; compact scores must be in [0, 1].
    Precise results need not be probabilities. The default skips compact work.
    This enforces result membership, not authentication of host callbacks.
    """

    def search(self, *, eligible, precise, scope, limit=10, exact_key=None, compact=None, policy=None):
        identifier(scope)
        if type(limit) is not int or not 1 <= limit <= 100:
            raise ValueError("limit must be between one and 100")
        if isinstance(eligible, str) or len(eligible) > 1000000:
            raise ValueError("bounded explicit eligible IDs required")
        ids = tuple(eligible)
        for key in ids:
            identifier(key)
        allowed = set(ids)
        if len(allowed) != len(ids):
            raise ValueError("duplicate eligible ID")
        if exact_key is not None:
            identifier(exact_key)
            return {"route": "exact" if exact_key in allowed else "abstain",
                    "ids": [exact_key] if exact_key in allowed else [], "compact_attempted": False}
        if not ids:
            return {"route": "abstain", "ids": [], "compact_attempted": False}

        def checked(callback, count, bounded=False):
            rows = callback(ids, count)
            if not isinstance(rows, (list, tuple)) or len(rows) != min(count, len(ids)):
                raise ValueError("callback must return the requested number of ranked results")
            found, previous = set(), math.inf
            for key, score in rows:
                if key not in allowed or key in found or not math.isfinite(score) or score > previous:
                    raise ValueError("invalid, unauthorized, duplicate or unsorted callback result")
                if bounded and not 0 <= score <= 1:
                    raise ValueError("compact scores must be in [0, 1]")
                found.add(key)
                previous = score
            return rows

        attempted = bool(policy is not None and policy.enabled and policy.scope == scope
                         and policy.limit == limit and compact is not None and len(ids) > limit)
        if attempted:
            rows = checked(compact, limit + 1, True)
            score, margin = rows[limit - 1][1], rows[limit - 1][1] - rows[limit][1]
            if policy.accepts(score, margin, scope=scope, limit=limit):
                return {"route": "compact", "ids": [key for key, _ in rows[:limit]],
                        "compact_attempted": True}
        rows = checked(precise, limit)
        return {"route": "precise", "ids": [key for key, _ in rows], "compact_attempted": attempted}

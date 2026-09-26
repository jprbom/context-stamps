"""Bounded prefix acquisition and whole-trajectory risk accounting.

Copyright (c) 2026 Prashant Jagtap. MIT License.
This planner does not read evidence, authorize sources, or establish answer truth.
The initial learned target is retention of judged evidence within a fixed pool.
"""

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from functools import cached_property

from .context_state import canonical, digest
from .decisions.calibration import error_upper_bound
from .security import identifier, read_text

FEATURE_REVISION = "hybrid-prefix-33-v1"
TARGET_REVISION = "nonempty-judged-pool-retention-v1"
FEATURES = 33


def number(value, low, high):
    if type(value) not in (int, float) or not math.isfinite(value) or not low <= value <= high:
        raise ValueError("finite bounded numeric value required")


def fingerprint(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def candidate_order(rows):
    """Validate bounded six-feature rows and return stable hybrid-score order."""
    if type(rows) is not tuple or not 1 <= len(rows) <= 256:
        raise ValueError("one to 256 immutable candidate rows required")
    for row in rows:
        if type(row) is not tuple or len(row) != 6:
            raise ValueError("six retrieval features required")
        for v in row:
            number(v, -100000, 100000)
        for v in row[4:]:
            number(v, 0, 1)
    return tuple(sorted(range(len(rows)), key=lambda i: (-rows[i][2], i)))


def prefix_features(rows):
    """Use precomputed retrieval scores only; no qrels, teacher or query IDs.

    Each row: cosine, standardized cosine, hybrid, standardized BM25,
    reciprocal dense rank, reciprocal BM25 rank. Host must filter eligibility.
    Returns original row indices in hybrid order, prefix lengths and features.
    """
    order = candidate_order(rows)
    ordered = [rows[i] for i in order]
    n = len(rows)
    lengths = tuple(sorted({min(k, n) for k in (1, 2, 4, 8, 16, 32, 64, 128, 256)}))
    features = []
    for k in lengths:
        feature = [math.log1p(k) / math.log(257), k / n, n / 256]
        for j in range(6):
            prefix = [r[j] for r in ordered[:k]]
            mean = math.fsum(prefix) / k
            tail = math.fsum(r[j] for r in ordered[k:]) / (n - k) if k < n else 0.
            feature.extend((prefix[0], prefix[-1], mean,
                            math.sqrt(math.fsum((v - mean) ** 2 for v in prefix) / k), tail))
        features.append(tuple(feature))
    return order, lengths, tuple(features)


@dataclass(frozen=True)
class AcquisitionModel:
    """Portable linear or one-hidden-layer, three-head predictor; no pickle."""

    mean: tuple[float, ...]
    scale: tuple[float, ...]
    hidden_weights: tuple[tuple[float, ...], ...]
    hidden_bias: tuple[float, ...]
    output_weights: tuple[tuple[float, ...], ...]
    output_bias: tuple[float, ...]
    feature_revision: str = FEATURE_REVISION
    target_revision: str = TARGET_REVISION

    def __post_init__(self):
        if self.feature_revision != FEATURE_REVISION or self.target_revision != TARGET_REVISION:
            raise ValueError("unsupported feature or target family")
        if type(self.hidden_weights) is not tuple or len(self.hidden_weights) not in (0, 32):
            raise ValueError("supported bounded architecture required")
        width = len(self.hidden_weights)
        vectors = ((self.mean, FEATURES, -100000, 100000), (self.scale, FEATURES, 1e-6, 100000),
                   (self.hidden_bias, width, -100, 100), (self.output_bias, 3, -100, 100))
        for vector, size, low, high in vectors:
            if type(vector) is not tuple or len(vector) != size:
                raise ValueError("unaligned immutable model vector")
            for v in vector:
                number(v, low, high)
        if type(self.output_weights) is not tuple or len(self.output_weights) != 3:
            raise ValueError("three output heads required")
        for matrix, size in ((self.hidden_weights, FEATURES), (self.output_weights, width or FEATURES)):
            for vector in matrix:
                if type(vector) is not tuple or len(vector) != size:
                    raise ValueError("unaligned immutable model matrix")
                for v in vector:
                    number(v, -100, 100)

    @cached_property
    def revision(self):
        return fingerprint(asdict(self))

    def predict(self, features):
        if type(features) is not tuple or len(features) != FEATURES:
            raise ValueError("exact feature vector required")
        for value in features:
            number(value, -100000, 100000)
        values = [max(-8., min(8., (x - m) / s)) for x, m, s in zip(features, self.mean, self.scale)]
        if self.hidden_weights:
            values = [math.tanh(math.fsum(w * x for w, x in zip(row, values)) + b)
                      for row, b in zip(self.hidden_weights, self.hidden_bias)]
        logits = [max(-700., min(700., math.fsum(w * x for w, x in zip(row, values)) + b))
                  for row, b in zip(self.output_weights, self.output_bias)]
        # Scores for pool completeness, remaining judged gain, next-batch gain.
        # Sigmoid range alone does not make these calibrated probabilities.
        return tuple(1 / (1 + math.exp(-v)) for v in logits)

    def save(self, path):
        with open(path, "x", encoding="utf-8", newline="\n") as stream:
            stream.write(canonical({"model": asdict(self), "revision": self.revision}) + "\n")

    @classmethod
    def load(cls, path):
        def pairs(items):
            result = {}
            for key, value in items:
                if key in result:
                    raise ValueError("duplicate model key")
                result[key] = value
            return result
        payload = json.loads(read_text(path, 524288), object_pairs_hook=pairs)
        if type(payload) is not dict or set(payload) != {"model", "revision"}:
            raise ValueError("unsupported model file")
        raw = payload["model"]
        if type(raw) is not dict or set(raw) != set(cls.__dataclass_fields__):
            raise ValueError("unsupported model fields")
        for key in ("mean", "scale", "hidden_bias", "output_bias"):
            raw[key] = tuple(raw[key])
        for key in ("hidden_weights", "output_weights"):
            raw[key] = tuple(tuple(row) for row in raw[key])
        model = cls(**raw)
        if model.revision != payload["revision"]:
            raise ValueError("model content digest mismatch")
        return model


@dataclass(frozen=True)
class StopPolicy:
    model_revision: str
    threshold: float

    def __post_init__(self):
        digest(self.model_revision)
        number(self.threshold, 0, 1)

    @cached_property
    def revision(self):
        return fingerprint(asdict(self))

    def choose(self, scores):
        if type(scores) is not tuple or not 1 <= len(scores) <= 9:
            raise ValueError("bounded immutable score trajectory required")
        for score in scores:
            number(score, 0, 1)
        # Exhausting the pool is not an early stop or proof of sufficiency.
        return next((i for i, score in enumerate(scores[:-1]) if score >= self.threshold), len(scores) - 1)


@dataclass(frozen=True)
class Trajectory:
    cluster_id: str
    scores: tuple[float, ...]
    complete: tuple[bool, ...]

    def __post_init__(self):
        identifier(self.cluster_id)
        if (type(self.scores) is not tuple or not 1 <= len(self.scores) <= 9
                or type(self.complete) is not tuple or len(self.complete) != len(self.scores)
                or any(type(v) is not bool for v in self.complete)):
            raise ValueError("one bounded labelled trajectory per independent cluster required")
        for score in self.scores:
            number(score, 0, 1)
        if any(a and not b for a, b in zip(self.complete, self.complete[1:])):
            raise ValueError("pool-retention labels must be monotone")


@dataclass(frozen=True)
class AcquisitionRisk:
    policy_revision: str
    scope: str
    family_digest: str
    trajectory_digest: str
    clusters: int
    accepted: int
    errors: int
    family_size: int
    alpha: float
    upper_error: float | None

    def __post_init__(self):
        for value in (self.policy_revision, self.family_digest, self.trajectory_digest):
            digest(value)
        identifier(self.scope)
        if (any(type(v) is not int for v in (self.clusters, self.accepted, self.errors, self.family_size))
                or not 0 <= self.errors <= self.accepted <= self.clusters <= 10000
                or self.clusters == 0 or not 1 <= self.family_size <= 100):
            raise ValueError("bounded independent trajectory counts required")
        number(self.alpha, 1e-7, .5)
        expected = error_upper_bound(self.errors, self.accepted, self.alpha / self.family_size) if self.accepted else None
        if expected != self.upper_error:
            raise ValueError("risk bound does not match its counts")

    def permits(self, policy, *, scope, maximum_error):
        number(maximum_error, 0, 1)
        return (type(policy) is StopPolicy and policy.revision == self.policy_revision and scope == self.scope
                and self.upper_error is not None and self.upper_error <= maximum_error)


def assess_policy(policy, trajectories, *, scope, family, alpha=.05):
    """One observation per entire stopped trajectory, not per prefix.

    Family must contain all tested policy/scope pairs, frozen before calibration.
    Independence, representative sampling and unchanged future distributions are
    host assumptions, not properties established by the provided identifiers.
    """
    if type(policy) is not StopPolicy:
        raise ValueError("typed stop policy required")
    identifier(scope)
    if (type(family) is not tuple or not 1 <= len(family) <= 100
            or any(type(pair) is not tuple or len(pair) != 2 for pair in family)
            or len(set(family)) != len(family) or (scope, policy.revision) not in family):
        raise ValueError("complete unique frozen policy/scope family required")
    for domain, revision in family:
        identifier(domain)
        digest(revision)
    if (type(trajectories) is not tuple or not 1 <= len(trajectories) <= 10000
            or any(type(t) is not Trajectory for t in trajectories)
            or len({t.cluster_id for t in trajectories}) != len(trajectories)):
        raise ValueError("unique independent trajectory clusters required")
    number(alpha, 1e-7, .5)
    accepted = errors = 0
    for trajectory in trajectories:
        selected = policy.choose(trajectory.scores)
        if selected < len(trajectory.scores) - 1:
            accepted += 1
            errors += not trajectory.complete[selected]
    return AcquisitionRisk(policy.revision, scope, fingerprint(sorted(family)),
        fingerprint([asdict(t) for t in trajectories]), len(trajectories), accepted, errors,
        len(family), alpha, error_upper_bound(errors, accepted, alpha / len(family)) if accepted else None)


@dataclass(frozen=True)
class PrefixPlan:
    status: str
    indices: tuple[int, ...]
    score: float | None
    remaining_gain_score: float | None
    next_gain_score: float | None
    upper_error: float | None
    model_revision: str
    policy_revision: str


def plan_prefix(rows, model, policy, *, scope, risk=None, maximum_error=.05, max_items=256):
    """Return candidate indices only. Fail back to full pool without qualification.

    The caller must freshly enforce authorization, closure, complete serialized
    token budgets and downstream answer verification. No evidence is released.
    """
    if type(model) is not AcquisitionModel or type(policy) is not StopPolicy or model.revision != policy.model_revision:
        raise ValueError("matching immutable model and policy required")
    identifier(scope)
    number(maximum_error, 0, 1)
    if type(max_items) is not int or not 0 <= max_items <= 256:
        raise ValueError("bounded candidate budget required")
    if risk is not None and type(risk) is not AcquisitionRisk:
        raise ValueError("typed risk report required")
    if risk is None or not risk.permits(policy, scope=scope, maximum_error=maximum_error):
        order = candidate_order(rows)
        status = "unqualified_full_pool" if len(order) <= max_items else "budget_exhausted"
        return PrefixPlan(status, order[:max_items], None, None, None, None, model.revision, policy.revision)
    order, lengths, features = prefix_features(rows)
    for i, (length, feature) in enumerate(zip(lengths, features)):
        if length > max_items:
            return PrefixPlan("budget_exhausted", order[:max_items], None, None, None, None,
                              model.revision, policy.revision)
        scores = model.predict(feature)
        if i == len(lengths) - 1 or scores[0] >= policy.threshold:
            return PrefixPlan("early_stop" if i < len(lengths) - 1 else "pool_exhausted", order[:length],
                              *scores, risk.upper_error, model.revision, policy.revision)
    raise RuntimeError("unreachable acquisition state")

"""Held-out reliability bins with one-sided binomial error bounds.

Copyright (c) 2026 Prashant Jagtap. MIT License.
Labels, independent clusters and dataset separation come from the trusted host.
Bounds describe the sampled scope; they do not guarantee new-domain correctness.
"""

import hashlib
import math
from dataclasses import asdict, dataclass

from ..context_state import canonical, digest
from ..security import identifier


def error_upper_bound(errors, total, alpha=.05):
    """Exact one-sided Clopper-Pearson upper bound; implemented in log space."""
    if (type(errors) is not int or type(total) is not int or not 0 <= errors <= total <= 10000
            or total == 0 or type(alpha) not in (float, int) or not 1e-12 <= alpha < 1):
        raise ValueError("bounded binomial sample required")
    if errors == total:
        return 1.
    if errors == 0:
        return -math.expm1(math.log(alpha) / total)
    low, high = errors / total, 1.
    coefficients = [math.lgamma(total + 1) - math.lgamma(k + 1) - math.lgamma(total - k + 1)
                    for k in range(errors + 1)]
    for _ in range(60):
        p = min((low + high) / 2, math.nextafter(1., 0.))
        terms = [c + k * math.log(p) + (total - k) * math.log1p(-p) for k, c in enumerate(coefficients)]
        maximum = max(terms)
        log_cdf = maximum + math.log(sum(math.exp(v - maximum) for v in terms))
        if log_cdf > math.log(alpha):
            low = p
        else:
            high = p
    return high


@dataclass(frozen=True)
class CalibrationSample:
    cluster_id: str
    confidence: float
    correct: bool

    def __post_init__(self):
        identifier(self.cluster_id)
        if (type(self.confidence) not in (float, int) or not math.isfinite(self.confidence)
                or not 0 <= self.confidence <= 1 or type(self.correct) is not bool):
            raise ValueError("finite confidence and independent correctness label required")


@dataclass(frozen=True)
class CalibrationBin:
    lower: float
    upper: float
    count: int
    probability: float | None
    lower_correctness: float | None

    def __post_init__(self):
        if (any(type(v) not in (int, float) or not math.isfinite(v) for v in (self.lower, self.upper))
                or not 0 <= self.lower < self.upper <= 1
                or type(self.count) is not int or not 0 <= self.count <= 10000):
            raise ValueError("bounded calibration interval and count required")
        if self.count == 0:
            if self.probability is not None or self.lower_correctness is not None:
                raise ValueError("empty bin cannot assert reliability")
        elif (any(type(v) not in (int, float) or not math.isfinite(v)
                  for v in (self.probability, self.lower_correctness))
              or not 0 <= self.lower_correctness <= self.probability <= 1):
            raise ValueError("bounded probability and conservative correctness bound required")


@dataclass(frozen=True)
class Calibration:
    model_revision: str
    verifier_revision: str
    policy_revision: str
    scope: str
    schema_id: str
    sample_digest: str
    ece: float
    minimum_count: int
    bins: tuple[CalibrationBin, ...]
    alpha: float
    revision: str

    def __post_init__(self):
        for value in (self.model_revision, self.verifier_revision, self.policy_revision, self.scope):
            identifier(value)
        for value in (self.schema_id, self.sample_digest, self.revision):
            digest(value)
        if (type(self.ece) not in (int, float) or not math.isfinite(self.ece) or not 0 <= self.ece <= 1
                or type(self.alpha) not in (int, float) or not math.isfinite(self.alpha) or not 1e-9 <= self.alpha < 1
                or type(self.minimum_count) is not int or not 1 <= self.minimum_count <= 10000
                or type(self.bins) is not tuple or not 1 <= len(self.bins) <= 20
                or any(type(b) is not CalibrationBin for b in self.bins)
                or not 1 <= sum(b.count for b in self.bins) <= 10000):
            raise ValueError("bounded calibration certificate required")
        for i, item in enumerate(self.bins):
            if item.lower != i / len(self.bins) or item.upper != (i + 1) / len(self.bins):
                raise ValueError("fixed disjoint calibration intervals required")
        material = asdict(self)
        material.pop("revision")
        if hashlib.sha256(canonical(material).encode()).hexdigest() != self.revision:
            raise ValueError("calibration revision does not bind its content")

    def lookup(self, confidence, *, model_revision, scope, schema_id, verifier_revision, policy_revision):
        if ((model_revision, scope, schema_id, verifier_revision, policy_revision) !=
                (self.model_revision, self.scope, self.schema_id, self.verifier_revision, self.policy_revision)):
            return None
        if type(confidence) not in (int, float) or not math.isfinite(confidence) or not 0 <= confidence <= 1:
            return None
        for i, item in enumerate(self.bins):
            if item.lower <= confidence and (confidence < item.upper or i == len(self.bins) - 1 and confidence == 1):
                return item if item.count >= self.minimum_count else None
        return None


def fit_calibration(samples, *, model_revision, verifier_revision, policy_revision,
                    scope, schema_id, bins=10, minimum_count=30, alpha=.05):
    identifier(model_revision)
    identifier(verifier_revision)
    identifier(policy_revision)
    identifier(scope)
    digest(schema_id)
    if (type(samples) is not tuple or not 1 <= len(samples) <= 10000
            or any(type(s) is not CalibrationSample for s in samples)
            or len({s.cluster_id for s in samples}) != len(samples)):
        raise ValueError("one sample per independent cluster required")
    if (type(bins) is not int or not 1 <= bins <= 20 or type(minimum_count) is not int
            or not 1 <= minimum_count <= 10000):
        raise ValueError("bounded calibration binning required")
    if type(alpha) not in (int, float) or not math.isfinite(alpha) or not 1e-9 <= alpha < 1:
        raise ValueError("invalid confidence level")
    buckets = [[] for _ in range(bins)]
    for sample in samples:
        buckets[min(int(sample.confidence * bins), bins - 1)].append(sample)
    result, ece = [], 0.
    for i, bucket in enumerate(buckets):
        count = len(bucket)
        correct = sum(s.correct for s in bucket)
        probability = correct / count if count else None
        # Bonferroni correction supports adaptive choice among these fixed bins.
        lower = 1 - error_upper_bound(count - correct, count, alpha / bins) if count else None
        if count:
            ece += count / len(samples) * abs(probability - sum(s.confidence for s in bucket) / count)
        result.append(CalibrationBin(i / bins, (i + 1) / bins, count, probability, lower))
    sample_digest = hashlib.sha256(canonical([asdict(s) for s in samples]).encode()).hexdigest()
    payload = dict(model_revision=model_revision, verifier_revision=verifier_revision, policy_revision=policy_revision,
                   scope=scope, schema_id=schema_id, sample_digest=sample_digest,
                   ece=ece, minimum_count=minimum_count, bins=tuple(result), alpha=alpha)
    revision = hashlib.sha256(canonical({**payload, "bins": [asdict(b) for b in result]}).encode()).hexdigest()
    return Calibration(**payload, revision=revision)

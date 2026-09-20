"""Calibrated lexical/semantic score fusion for a precise retrieval backend."""

import hashlib
import json
import math
import random
from dataclasses import dataclass


def standardize_scores(values):
    """Population z-scores with a deterministic zero-variance result."""
    values = tuple(float(value) for value in values)
    if not 1 <= len(values) <= 1_000_000 or not all(math.isfinite(value) for value in values):
        raise ValueError("one to 1000000 finite scores required")
    mean = math.fsum(values) / len(values)
    variance = math.fsum((value - mean) ** 2 for value in values) / len(values)
    if variance == 0:
        return (0.0,) * len(values)
    scale = math.sqrt(variance)
    return tuple((value - mean) / scale for value in values)


@dataclass(frozen=True)
class HybridScoreProfile:
    semantic_weight: float = 0.75
    normalization: str = "population-z-v1"

    def __post_init__(self):
        if (type(self.semantic_weight) not in (int, float)
                or not math.isfinite(self.semantic_weight)
                or not 0 <= self.semantic_weight <= 1):
            raise ValueError("semantic weight must be in [0, 1]")
        if self.normalization != "population-z-v1":
            raise ValueError("unsupported normalization")

    @property
    def identity(self):
        payload = {"normalization": self.normalization, "semantic_weight": self.semantic_weight}
        return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()

    def fuse(self, semantic, lexical):
        semantic, lexical = standardize_scores(semantic), standardize_scores(lexical)
        if len(semantic) != len(lexical):
            raise ValueError("semantic and lexical scores must align")
        weight = self.semantic_weight
        return tuple(weight * sem + (1 - weight) * lex for sem, lex in zip(semantic, lexical))


@dataclass(frozen=True)
class ScopeCertificate:
    """Validation-bound permission to use a hybrid profile in one retrieval scope.

    The certificate contains no retrieval scores or source text.  Its digest binds the
    scope, profile and validation deltas so a different collection cannot silently
    inherit the decision.
    """

    scope: str
    profile_identity: str
    sample_count: int
    mean_gain: float
    lower_gain: float
    upper_gain: float
    enabled: bool
    validation_digest: str
    seed: int
    resamples: int

    @property
    def selected_profile(self):
        """Return the certified hybrid profile identity, or ``None`` for dense fallback."""
        return self.profile_identity if self.enabled else None


def certify_hybrid_scope(
    scope,
    profile,
    dense_metric,
    hybrid_metric,
    *,
    seed=20260920,
    resamples=5_000,
    minimum_samples=100,
    minimum_gain=0.0,
):
    """Enable a hybrid profile only when a paired bootstrap lower bound is positive.

    ``dense_metric`` and ``hybrid_metric`` must be per-query values from a disjoint
    validation set.  Evaluation results must not be reused to issue a certificate.
    """
    if not isinstance(scope, str) or not scope.strip() or len(scope) > 256:
        raise ValueError("a non-empty scope of at most 256 characters is required")
    if not isinstance(profile, HybridScoreProfile):
        raise TypeError("profile must be a HybridScoreProfile")
    dense = tuple(float(value) for value in dense_metric)
    hybrid = tuple(float(value) for value in hybrid_metric)
    if len(dense) != len(hybrid) or len(dense) < minimum_samples:
        raise ValueError("aligned validation metrics must meet minimum_samples")
    if not all(math.isfinite(value) for value in (*dense, *hybrid)):
        raise ValueError("validation metrics must be finite")
    if type(seed) is not int or type(resamples) is not int or not 1_000 <= resamples <= 100_000:
        raise ValueError("seed must be an integer and resamples must be in [1000, 100000]")
    if type(minimum_samples) is not int or minimum_samples < 2:
        raise ValueError("minimum_samples must be an integer of at least two")
    if type(minimum_gain) not in (int, float) or not math.isfinite(minimum_gain):
        raise ValueError("minimum_gain must be finite")

    deltas = tuple(hybrid_value - dense_value for dense_value, hybrid_value in zip(dense, hybrid))
    # Seeded reproducibility is required for statistical resampling; this is not a security RNG.
    generator = random.Random(seed)  # nosec B311
    size = len(deltas)
    means = sorted(
        math.fsum(deltas[generator.randrange(size)] for _ in range(size)) / size
        for _ in range(resamples)
    )
    lower = means[int(.025 * resamples)]
    upper = means[min(resamples - 1, int(.975 * resamples))]
    mean = math.fsum(deltas) / size
    digest_payload = {
        "dense": dense,
        "hybrid": hybrid,
        "minimum_gain": float(minimum_gain),
        "profile_identity": profile.identity,
        "scope": scope.strip(),
    }
    validation_digest = hashlib.sha256(
        json.dumps(digest_payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return ScopeCertificate(
        scope=scope.strip(),
        profile_identity=profile.identity,
        sample_count=size,
        mean_gain=mean,
        lower_gain=lower,
        upper_gain=upper,
        enabled=lower > minimum_gain,
        validation_digest=validation_digest,
        seed=seed,
        resamples=resamples,
    )

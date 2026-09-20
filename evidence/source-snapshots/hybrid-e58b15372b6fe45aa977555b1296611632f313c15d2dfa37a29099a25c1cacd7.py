"""Calibrated lexical/semantic score fusion for a precise retrieval backend."""

import hashlib
import json
import math
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


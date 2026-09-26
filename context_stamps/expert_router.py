"""Small cost-aware expert policy with bounded, selective coefficient quantization.

Copyright (c) 2026 Prashant Jagtap. MIT License.
A calibrated scope enables routing; predictions are not quality guarantees.
"""

import math
from dataclasses import dataclass

from .security import identifier


@dataclass(frozen=True)
class LinearExpertRouter:
    means: tuple[float, ...]
    scales: tuple[float, ...]
    weights: tuple[float, ...]
    intercept: float
    threshold: float
    approved_scopes: tuple[str, ...] = ()

    def __post_init__(self):
        n = len(self.weights)
        if (not 1 <= n <= 64 or len(self.means) != n or len(self.scales) != n
                or len(self.approved_scopes) > 128):
            raise ValueError("invalid bounded router")
        for field in ("means", "scales", "weights", "approved_scopes"):
            object.__setattr__(self, field, tuple(getattr(self, field)))
        if any(type(v) not in (int, float) or not math.isfinite(v) or abs(v) > 1e6
               for v in (*self.means, *self.scales, *self.weights, self.intercept, self.threshold)):
            raise ValueError("finite bounded coefficients required")
        if any(v <= 1e-12 for v in self.scales):
            raise ValueError("positive scales required")
        for scope in self.approved_scopes:
            identifier(scope)

    def predict(self, features):
        if len(features) != len(self.weights) or any(type(v) not in (float, int) or not math.isfinite(v) for v in features):
            raise ValueError("aligned finite features required")
        x = [(v - m) / s for v, m, s in zip(features, self.means, self.scales)]
        if any(abs(v) > 8 for v in x):
            return None, "out_of_training_range"
        # Quantize coefficients only. The input-dependent error bound retains
        # FP64 routing at close decisions; this is not transformer quantization.
        scale = max(max(abs(w) for w in self.weights), 1e-12) / 32767
        approx = [round(w / scale) * scale for w in self.weights]
        score = self.intercept + math.fsum(a * v for a, v in zip(approx, x))
        error = math.fsum(abs(w - a) * abs(v) for w, a, v in zip(self.weights, approx, x))
        error += 1e-12 * (1 + abs(score))
        if abs(score - self.threshold) <= error:
            return self.intercept + math.fsum(w * v for w, v in zip(self.weights, x)), "boundary_fp64"
        return score, "int16_coefficients"

    def choose(self, features, *, scope):
        identifier(scope)
        if scope not in self.approved_scopes:
            return "fusion", "scope_unqualified"
        score, reason = self.predict(features)
        if score is None:
            return "fusion", reason
        return ("hybrid" if score >= self.threshold else "fusion"), reason

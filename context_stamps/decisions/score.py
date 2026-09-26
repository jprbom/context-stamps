"""Bounded scores are not probabilities. Copyright 2026 Prashant Jagtap, MIT."""

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class Score:
    minimum: float
    maximum: float

    def __post_init__(self):
        if (any(type(v) not in (float, int) or not math.isfinite(v) or abs(v) > 1e12
                for v in (self.minimum, self.maximum)) or self.minimum >= self.maximum):
            raise ValueError("finite ordered score bounds required")

    def accepts(self, value):
        return type(value) in (int, float) and math.isfinite(value) and self.minimum <= value <= self.maximum

"""Explicit decision risk policy. Copyright 2026 Prashant Jagtap, MIT."""

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class DecisionPolicy:
    minimum_probability: float | None = None

    def __post_init__(self):
        p = self.minimum_probability
        if p is not None and (type(p) not in (int, float) or not math.isfinite(p) or not 0 <= p <= 1):
            raise ValueError("bounded minimum probability required")

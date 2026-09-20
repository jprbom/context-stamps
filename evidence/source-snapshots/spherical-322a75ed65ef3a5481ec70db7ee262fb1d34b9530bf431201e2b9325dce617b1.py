"""Spherical Context QR: bounded multi-view angular fingerprints.

Each view is a point on its own unit sphere. The bundle is a product of
spheres, not a recoverable encoding of the original text or a visual QR code.
Copyright (c) 2026 Prashant Jagtap. MIT License.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from typing import Mapping, Sequence

from stamps import Family, Stamp, similarity, stamp_vector


@dataclass(frozen=True)
class SphericalStamp:
    views: tuple[tuple[str, Stamp], ...]

    def __post_init__(self):
        object.__setattr__(self, "views", tuple(tuple(v) for v in self.views))
        if not 1 <= len(self.views) <= 8:
            raise ValueError("one to eight views required")
        names = []
        for name, stamp in self.views:
            if not isinstance(name, str) or not re.fullmatch(r"[a-z][a-z0-9_]{0,31}", name):
                raise ValueError("invalid view name")
            if not isinstance(stamp, Stamp):
                raise ValueError("each view requires a Stamp")
            names.append(name)
        if len(set(names)) != len(names):
            raise ValueError("duplicate view names")
        object.__setattr__(self, "views", tuple(sorted(self.views)))

    @classmethod
    def encode(cls, vectors: Mapping[str, Sequence[float]], families: Mapping[str, Family]):
        if set(vectors) != set(families) or not 1 <= len(families) <= 8:
            raise ValueError("vectors and families must have the same bounded views")
        result = []
        for name, family in families.items():
            if family.method != "gaussian-v1" or family.mean or family.planes:
                raise ValueError("spherical-v1 requires uncentered Gaussian angular projections")
            if len(vectors[name]) != family.dim:
                raise ValueError("vector dimension mismatch")
            values = tuple(float(x) for x in vectors[name])
            norm = math.hypot(*values)
            if not math.isfinite(norm) or norm == 0:
                raise ValueError("each view must have a finite nonzero direction")
            result.append((name, stamp_vector(tuple(x / norm for x in values), family)))
        return cls(tuple(result))

    @property
    def bits(self):
        return sum(stamp.bits for _, stamp in self.views)

    def compare(self, other: SphericalStamp) -> dict[str, float]:
        if tuple(n for n, _ in self.views) != tuple(n for n, _ in other.views):
            raise ValueError("view schema mismatch")
        return {name: similarity(a, b) for (name, a), (_, b) in zip(self.views, other.views)}

    def score(self, other: SphericalStamp, weights: Mapping[str, float] | None = None) -> float:
        values = self.compare(other)
        weights = dict.fromkeys(values, 1.0) if weights is None else weights
        if set(weights) != set(values) or not all(
            isinstance(w, (float, int)) and math.isfinite(w) and w >= 0 for w in weights.values()
        ) or not 0 < sum(weights.values()) < math.inf:
            raise ValueError("finite nonnegative weights with positive total required for every view")
        return math.fsum(values[k] * (weights[k] / sum(weights.values())) for k in values)

    def to_payload(self) -> str:
        """Portable payload; a QR renderer may encode these bytes if capacity permits."""
        return json.dumps({"format": "scqr1", "views": {n: str(s) for n, s in self.views}},
                          sort_keys=True, separators=(",", ":"))

    @classmethod
    def from_payload(cls, payload: str):
        if not isinstance(payload, str) or len(payload.encode("utf-8")) > 8192:
            raise ValueError("payload exceeds 8192 bytes")

        def unique(pairs):
            obj = {}
            for key, value in pairs:
                if key in obj:
                    raise ValueError("duplicate payload key")
                obj[key] = value
            return obj

        try:
            obj = json.loads(payload, object_pairs_hook=unique)
            if set(obj) != {"format", "views"} or obj["format"] != "scqr1":
                raise ValueError("unsupported payload")
            if not isinstance(obj["views"], dict):
                raise ValueError("views must be an object")
            return cls(tuple((n, Stamp.parse(s)) for n, s in obj["views"].items()))
        except (TypeError, KeyError, AttributeError, RecursionError) as exc:
            raise ValueError("invalid spherical payload") from exc

"""Small inspectable relevance model; optional NumPy fitting, stdlib inference."""

import json
import math
from dataclasses import asdict, dataclass

from .spherical import SphericalStamp


@dataclass(frozen=True)
class FacetModel:
    names: tuple[str, ...]
    coefficients: tuple[float, ...]
    intercept: float
    family_ids: tuple[str, ...]

    def __post_init__(self):
        for name in ("names", "coefficients", "family_ids"):
            object.__setattr__(self, name, tuple(getattr(self, name)))
        if not 1 <= len(self.names) <= 8 or len(set(self.names)) != len(self.names):
            raise ValueError("invalid model schema")
        if len(self.names) != len(self.coefficients) or len(self.names) != len(self.family_ids):
            raise ValueError("model dimensions mismatch")
        if not all(math.isfinite(x) for x in (*self.coefficients, self.intercept)):
            raise ValueError("finite parameters required")

    def score(self, query: SphericalStamp, candidate: SphericalStamp):
        if tuple(n for n, _ in query.views) != self.names or tuple(
            s.family_id for _, s in query.views
        ) != self.family_ids:
            raise ValueError("model encoder schema mismatch")
        features = query.compare(candidate)
        return self.intercept + math.fsum(w * features[n] for n, w in zip(self.names, self.coefficients))

    def to_json(self):
        return json.dumps({"format": "facet-ridge-v1", **asdict(self)}, sort_keys=True, allow_nan=False)

    @classmethod
    def from_json(cls, value):
        if not isinstance(value, str) or len(value.encode("utf-8")) > 16384:
            raise ValueError("model exceeds 16 KiB")
        data = json.loads(value)
        if data.pop("format", None) != "facet-ridge-v1":
            raise ValueError("unsupported model")
        return cls(**data)

    @classmethod
    def fit(cls, query_template, features, labels, *, penalty=1.0):
        import numpy as np

        x, y = np.asarray(features, dtype=float), np.asarray(labels, dtype=float)
        if (x.ndim != 2 or not 1 <= len(x) <= 100000 or x.shape[1] != len(query_template.views)
                or y.shape != (len(x),) or not np.isfinite(x).all() or not np.isfinite(y).all()
                or not math.isfinite(penalty) or penalty <= 0):
            raise ValueError("invalid training data or penalty")
        mean, scale = x.mean(axis=0), x.std(axis=0)
        scale[scale < 1e-12] = 1.0
        z, center = (x - mean) / scale, float(y.mean())
        w = np.linalg.solve(z.T @ z + penalty * np.eye(x.shape[1]), z.T @ (y - center)) / scale
        return cls(tuple(n for n, _ in query_template.views), tuple(float(v) for v in w),
                   float(center - mean @ w), tuple(s.family_id for _, s in query_template.views))

"""Explicit partial-facet queries without fabricated values for missing facets.

Document stamps retain their full schema and 32-byte format. Scores use only
observed query views. Mask/weight changes produce a different policy scope.
"""

import hashlib
import json
import math
from dataclasses import dataclass

from .security import identifier
from .spherical import SphericalStamp


@dataclass(frozen=True)
class FacetQuery:
    observed: SphericalStamp
    weights: tuple[tuple[str, float], ...] = ()

    def __post_init__(self):
        if not isinstance(self.observed, SphericalStamp):
            raise ValueError('observed query must be a nonempty spherical stamp')
        names = {name for name, _ in self.observed.views}
        weights = tuple(sorted(self.weights)) if self.weights else tuple((name, 1.0) for name in sorted(names))
        if (len(weights) != len(names) or {name for name, _ in weights} != names
                or not all(type(value) in (int, float) and math.isfinite(value) and value > 0
                           for _, value in weights)):
            raise ValueError('one positive finite weight per observed view required')
        total = sum(value for _, value in weights)
        if not math.isfinite(total):
            raise ValueError('weight total must be finite')
        normalized = tuple((name, value / total) for name, value in weights)
        if any(value == 0 for _, value in normalized):
            raise ValueError('weight normalization must not erase an observed view')
        object.__setattr__(self, 'weights', normalized)

    def policy_scope(self, application_scope):
        """Bind calibrated decisions to the active facets, families and weights.

        The host's scope must still identify the domain and candidate schema.
        Query values are intentionally absent: policies apply to query cohorts.
        """
        identifier(application_scope)
        payload = [application_scope, [(name, stamp.family_id, stamp.bits)
                                       for name, stamp in self.observed.views], self.weights]
        return 'facet-query:' + hashlib.sha256(json.dumps(payload, separators=(',', ':')).encode()).hexdigest()

    def compare(self, candidate):
        if not isinstance(candidate, SphericalStamp):
            raise ValueError('candidate must be a spherical stamp')
        values = dict(candidate.views)
        if not all(name in values for name, _ in self.observed.views):
            raise ValueError('candidate is missing an observed query view')
        subset = SphericalStamp(tuple((name, values[name]) for name, _ in self.observed.views))
        return self.observed.compare(subset)

    def score(self, candidate):
        agreement = self.compare(candidate)
        return math.fsum(weight * agreement[name] for name, weight in self.weights)

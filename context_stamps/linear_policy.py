"""Regularized local policy proposals from fully paired measured outcomes.

Copyright (c) 2026 Prashant Jagtap. MIT License.
This model predicts relative utility, not correctness probability. Promotion,
authorization, retention checks and rollback remain the host's responsibility.
"""

import math
from dataclasses import asdict, dataclass

from .experience import _hash, _id, _number
from .local_learning import revision


def _features(values, dimension):
    if type(values) is not tuple or len(values) != dimension:
        raise ValueError("fixed immutable feature vector required")
    for value in values:
        _number(value, -1, 1)


@dataclass(frozen=True)
class UtilityPair:
    cluster_id: str
    features: tuple[float, ...]
    baseline_correct: bool
    candidate_correct: bool
    baseline_cost: float
    candidate_cost: float

    def __post_init__(self):
        _id(self.cluster_id)
        if not 1 <= len(self.features) <= 64:
            raise ValueError("one to 64 features required")
        _features(self.features, len(self.features))
        if type(self.baseline_correct) is not bool or type(self.candidate_correct) is not bool:
            raise ValueError("both externally verified outcomes required")
        _number(self.baseline_cost, 0, 1e12)
        _number(self.candidate_cost, 0, 1e12)


@dataclass(frozen=True)
class LinearPolicy:
    binding: str
    baseline_action: str
    candidate_action: str
    coefficients: tuple[float, ...]
    intercept: float

    def __post_init__(self):
        _hash(self.binding)
        _id(self.baseline_action)
        _id(self.candidate_action)
        if self.baseline_action == self.candidate_action:
            raise ValueError("two distinct actions required")
        if type(self.coefficients) is not tuple or not 1 <= len(self.coefficients) <= 64:
            raise ValueError("bounded immutable coefficients required")
        for value in self.coefficients + (self.intercept,):
            _number(value, -1e9, 1e9)

    @property
    def revision(self):
        return revision(asdict(self))

    def choose(self, features, *, binding, allowed_actions):
        _features(features, len(self.coefficients))
        _hash(binding)
        if type(allowed_actions) is not tuple or not 1 <= len(allowed_actions) <= 16:
            raise ValueError("bounded host-authorized actions required")
        for action in allowed_actions:
            _id(action)
        if binding != self.binding:
            return None
        utility = self.intercept + math.fsum(a*b for a, b in zip(self.coefficients, features))
        action = self.candidate_action if utility > 0 else self.baseline_action
        if action in allowed_actions:
            return action
        return self.baseline_action if self.baseline_action in allowed_actions else None


def fit_linear_policy(rows, *, binding, baseline_action, candidate_action,
                      cost_cap, failure_penalty=20., ridge=10.):
    """Ridge least squares of paired quality gain minus normalized extra cost.

    Feature range [-1,1] and positive ridge bound numerical scale. The intercept
    is also regularized. No evaluation examples or imagined counterfactuals may
    enter this fit; a separate host gate must qualify the resulting proposal.
    """
    import numpy as np

    _hash(binding)
    _number(cost_cap, 1e-12, 1e12)
    _number(failure_penalty, 1e-9, 1e6)
    _number(ridge, 1e-6, 1e6)
    if (type(rows) is not tuple or not 2 <= len(rows) <= 100000
            or any(type(r) is not UtilityPair for r in rows)):
        raise ValueError("bounded paired training observations required")
    dimension = len(rows[0].features)
    if len({r.cluster_id for r in rows}) != len(rows):
        raise ValueError("duplicate training cluster")
    for row in rows:
        _features(row.features, dimension)
        if max(row.baseline_cost, row.candidate_cost) > cost_cap:
            raise ValueError("observed training cost exceeds declared cap")
    x = np.array([r.features + (1.,) for r in rows], dtype=np.float64)
    y = np.array([failure_penalty*(int(r.candidate_correct)-int(r.baseline_correct))
                  - (r.candidate_cost-r.baseline_cost)/cost_cap for r in rows])
    weights = np.linalg.solve(x.T @ x + ridge*np.eye(dimension+1), x.T @ y)
    return LinearPolicy(binding, baseline_action, candidate_action,
                        tuple(map(float, weights[:-1])), float(weights[-1]))

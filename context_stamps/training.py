"""Optional monotone pairwise training: optimize rankings, not pointwise labels."""

import math

from .facet_model import FacetModel


def fit_pairwise(template, differences, *, penalty=.01, epochs=400, learning_rate=.1):
    """Positive-minus-negative feature pairs; nonnegative weights resist inverse-view shortcuts.

    This is constrained logistic ranking, an established objective. It does not
    guarantee generalization or calibrate an activation probability.
    """
    import numpy as np

    x = np.asarray(differences, dtype=float)
    if x.ndim != 2 or not 1 <= len(x) <= 100000 or x.shape[1] != len(template.views):
        raise ValueError("invalid pairwise dimensions")
    if not np.isfinite(x).all() or np.max(np.abs(x)) > 1:
        raise ValueError("differences must be finite agreement differences in [-1,1]")
    if (type(epochs) is not int or not 1 <= epochs <= 10000 or not math.isfinite(penalty)
            or penalty < 0 or not math.isfinite(learning_rate) or not 0 < learning_rate <= 1):
        raise ValueError("invalid optimizer settings")
    weights = np.ones(x.shape[1], dtype=float)
    history = []
    for step in range(epochs + 1):
        margin = x @ weights
        loss = float(np.logaddexp(0, -margin).mean() + penalty * (weights @ weights) / 2)
        if step % 20 == 0 or step == epochs:
            history.append({"step": step, "loss": loss})
        if step == epochs:
            break
        probability = np.exp(-np.logaddexp(0, margin))
        gradient = -(x.T @ probability) / len(x) + penalty * weights
        weights = np.maximum(0, weights - learning_rate * gradient)
    model = FacetModel(tuple(n for n, _ in template.views), tuple(float(w) for w in weights), 0,
                       tuple(s.family_id for _, s in template.views))
    return model, history

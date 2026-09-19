"""Exact application constraints before approximate spherical activation."""

from .activation import activate
from .security import identifier


def activate_constrained(query, candidates, *, metadata, required, threshold, limit=5, model=None):
    """Apply trusted exact facets first; a hash collision cannot override them.

    Metadata must be bound to the current candidate by the application. This is
    exact schema enforcement, not automatic entity extraction or truth checking.
    """
    if len(candidates) != len(metadata) or len(candidates) > 1000:
        raise ValueError("metadata must align with the bounded candidate list")
    if not isinstance(required, dict) or not 1 <= len(required) <= 16:
        raise ValueError("one to 16 explicit constraints required")
    for name, value in required.items():
        identifier(name)
        identifier(value)
    eligible = []
    for i, fields in enumerate(metadata):
        if not isinstance(fields, dict):
            raise ValueError("candidate metadata must be a mapping")
        if all(fields.get(k) == v for k, v in required.items()):
            eligible.append(i)
    hits = activate(query, [candidates[i] for i in eligible], threshold=threshold, limit=limit, model=model)
    return [{**row, "index": eligible[row["index"]]} for row in hits]

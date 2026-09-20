"""Versionable typed relation maps and bounded diffusion features.

The exact edge list remains external evidence.  The vector is a lossy routing
feature derived by truncated personalized diffusion over declared edges.
"""

import hashlib
import json
import math
from dataclasses import dataclass

from .security import identifier


@dataclass(frozen=True, order=True)
class RelationEdge:
    source: str
    target: str
    kind: str
    weight: float = 1.0

    def __post_init__(self):
        identifier(self.source)
        identifier(self.target)
        identifier(self.kind)
        if (type(self.weight) not in (int, float) or not math.isfinite(self.weight)
                or not 0 < self.weight <= 1000):
            raise ValueError("edge weight must be finite and in (0, 1000]")


class RelationMap:
    """Immutable relation set with personalized-diffusion routing features."""

    def __init__(self, edges):
        values = tuple(sorted(RelationEdge(*edge) if not isinstance(edge, RelationEdge) else edge
                              for edge in edges))
        if not 1 <= len(values) <= 4096 or len(set(values)) != len(values):
            raise ValueError("one to 4096 unique edges required")
        self.edges = values

    @property
    def revision(self):
        payload = [(edge.source, edge.target, edge.kind, edge.weight) for edge in self.edges]
        return hashlib.sha256(json.dumps(payload, separators=(",", ":")).encode()).hexdigest()

    @staticmethod
    def _feature(vector, token, value):
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        position = int.from_bytes(digest[:4], "big") % len(vector)
        vector[position] += value if digest[4] & 1 else -value

    def encode(self, root, *, dim=64, hops=2, restart=0.35, decay=0.7):
        """Hash a bounded personalized diffusion into a normalized vector.

        Each step applies ``p[t+1] = restart*e[root] + (1-restart)*P^T p[t]``.
        Typed forward and reverse arcs contribute signed features, discounted by
        hop depth.  This is not an exact graph encoding or authorization check.
        """
        identifier(root)
        if (type(dim) is not int or not 8 <= dim <= 2048 or type(hops) is not int
                or not 1 <= hops <= 8 or not 0 < restart < 1 or not 0 < decay <= 1):
            raise ValueError("invalid diffusion parameters")
        adjacency = {}
        for edge in self.edges:
            adjacency.setdefault(edge.source, []).append((edge.target, edge.kind, edge.weight, "forward"))
            adjacency.setdefault(edge.target, []).append((edge.source, edge.kind, edge.weight, "reverse"))
        if root not in adjacency:
            raise ValueError("root is absent from relation map")
        vector = [0.0] * dim
        state = {root: 1.0}
        self._feature(vector, "root:" + root, 1.0)
        for depth in range(1, hops + 1):
            following = {root: restart}
            for source, mass in state.items():
                arcs = adjacency.get(source, ())
                total = math.fsum(arc[2] for arc in arcs)
                if not total:
                    following[root] = following.get(root, 0.0) + (1 - restart) * mass
                    continue
                for target, kind, weight, direction in arcs:
                    contribution = (1 - restart) * mass * weight / total
                    following[target] = following.get(target, 0.0) + contribution
                    token = f"h{depth}:{direction}:{kind}:{target}"
                    self._feature(vector, token, contribution * decay ** (depth - 1))
            state = following
        norm = math.sqrt(math.fsum(value * value for value in vector))
        if norm == 0:
            raise ValueError("relation features cancelled; increase dimension")
        return tuple(value / norm for value in vector)

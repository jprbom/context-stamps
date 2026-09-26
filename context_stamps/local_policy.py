"""Small Beta-smoothed context policy fitted from paired local outcomes.

Copyright (c) 2026 Prashant Jagtap. MIT License.
Proposals only: posterior means are not calibrated deployment probabilities.
Exact trusted context cells must never be replaced by unverified hash equality.
"""

from dataclasses import asdict, dataclass

from .experience import _hash, _id, _number
from .local_learning import revision


@dataclass(frozen=True)
class PolicyObservation:
    cluster_id: str
    cell: str
    action: str
    correct: bool
    cost: float

    def __post_init__(self):
        for value in (self.cluster_id, self.cell, self.action):
            _id(value)
        if type(self.correct) is not bool:
            raise ValueError("externally verified training outcome required")
        _number(self.cost, 0, 1e12)


@dataclass(frozen=True)
class CellPolicy:
    binding: str
    baseline_action: str
    choices: tuple[tuple[str, str], ...]

    def __post_init__(self):
        _hash(self.binding)
        _id(self.baseline_action)
        if (type(self.choices) is not tuple or len(self.choices) > 256
                or any(type(row) is not tuple or len(row) != 2 for row in self.choices)):
            raise ValueError("bounded immutable cell/action pairs required")
        for cell, action in self.choices:
            _id(cell)
            _id(action)
        if len({cell for cell, _ in self.choices}) != len(self.choices):
            raise ValueError("duplicate policy cell")

    @property
    def revision(self):
        return revision(asdict(self))

    def choose(self, cell, *, binding, allowed_actions):
        """Host must independently authorize every action and every source."""
        _id(cell)
        _hash(binding)
        if type(allowed_actions) is not tuple or len(allowed_actions) > 16:
            raise ValueError("bounded host-authorized action set required")
        for value in allowed_actions:
            _id(value)
        if binding != self.binding:
            return None
        proposed = dict(self.choices).get(cell, self.baseline_action)
        if proposed in allowed_actions:
            return proposed
        return self.baseline_action if self.baseline_action in allowed_actions else None


def fit_cell_policy(rows, *, binding, baseline_action, cost_cap, failure_penalty=20., minimum_tasks=20):
    """Beta(1,1) failure mean + normalized cost; unsupported cells keep baseline.

    Every task must have one measured observation for every registered action.
    This avoids treating missing counterfactuals as failures, successes or zeros.
    Cells/actions must be predefined using train-visible input features only.
    """
    _hash(binding)
    _id(baseline_action)
    _number(cost_cap, 1e-12, 1e12)
    _number(failure_penalty, 1e-9, 1e6)
    if type(minimum_tasks) is not int or not 1 <= minimum_tasks <= 10000:
        raise ValueError("positive bounded training support required")
    if (type(rows) is not tuple or not 1 <= len(rows) <= 100000
            or any(type(row) is not PolicyObservation for row in rows)):
        raise ValueError("bounded typed training observations required")
    actions = sorted({r.action for r in rows})
    cells = sorted({r.cell for r in rows})
    if not 2 <= len(actions) <= 16 or baseline_action not in actions or len(cells) > 256:
        raise ValueError("bounded cells and at least two actions including baseline required")
    tasks, buckets = {}, {}
    for row in rows:
        if row.cost > cost_cap:
            raise ValueError("training cost exceeds declared range")
        cell, observed = tasks.setdefault(row.cluster_id, (row.cell, set()))
        if cell != row.cell or row.action in observed:
            raise ValueError("task cell changed or repeated task/action observation")
        observed.add(row.action)
        count, errors, total = buckets.get((row.cell, row.action), (0, 0, 0.))
        buckets[(row.cell, row.action)] = count + 1, errors + int(not row.correct), total + row.cost
    if any(observed != set(actions) for _, observed in tasks.values()):
        raise ValueError("complete matched action outcomes required for every training task")
    choices = []
    for cell in cells:
        def utility(action):
            count, errors, total = buckets[(cell, action)]
            return failure_penalty * (errors + 1) / (count + 2) + total / count / cost_cap

        chosen = min(actions, key=lambda action: (utility(action), action != baseline_action, action))
        if buckets[(cell, baseline_action)][0] < minimum_tasks:
            chosen = baseline_action
        choices.append((cell, chosen))
    return CellPolicy(binding, baseline_action, tuple(choices))

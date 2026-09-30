"""Exact numerical tasks bound to explicit selectors and authorized table views.

Copyright (c) 2026 Prashant Jagtap. MIT License.
The host declares the task. This module does not interpret natural language,
verify perception, or turn a computed answer into an independent learning label.
"""

import hashlib
import json
import re
from dataclasses import asdict, dataclass, field
from decimal import Decimal, localcontext
from fractions import Fraction

from .context_state import CanonicalNode, ContextSnapshot, NodeRef, canonical, typed_tuple
from .security import bounded_text, identifier

REVISION = "numerical-task-v1"
NUMBER = re.compile(r"[+-]?(?:[0-9]{1,24}(?:\.[0-9]{1,24})?|\.[0-9]{1,24})(?:[eE][+-]?[0-9]{1,2})?")
OPERATIONS = frozenset(("lookup", "sum", "mean", "min", "max", "subtract", "absdiff", "ratio",
                        "percent_ratio", "greater", "less", "equal", "argmin", "argmax", "count"))


def _coordinates(values):
    if type(values) is not tuple or not 1 <= len(values) <= 16:
        raise ValueError("one to sixteen immutable named coordinates required")
    for pair in values:
        if type(pair) is not tuple or len(pair) != 2:
            raise ValueError("immutable coordinate pairs required")
        name, value = pair
        identifier(name)
        bounded_text(name, 64)
        bounded_text(value, 256)
        if not value.strip():
            raise ValueError("explicit nonempty coordinate value required")
    if len({name for name, _ in values}) != len(values):
        raise ValueError("duplicate coordinate name")
    return tuple(sorted(values))


def _bounded_fraction(value):
    if abs(value.numerator) > 10**48 or value.denominator > 10**48:
        raise ValueError("rational representation limit")
    return value


def _number(text):
    if type(text) is not str or not NUMBER.fullmatch(text):
        raise ValueError("bounded finite decimal text required; unit belongs in its own field")
    return _bounded_fraction(Fraction(text))


@dataclass(frozen=True)
class NumericalCell:
    key: str
    coordinates: tuple[tuple[str, str], ...]
    value: str | None
    unit: str = ""

    def __post_init__(self):
        identifier(self.key)
        object.__setattr__(self, "coordinates", _coordinates(self.coordinates))
        bounded_text(self.unit, 128)
        if self.value is not None:
            _number(self.value)


@dataclass(frozen=True)
class NumericalTable:
    cells: tuple[NumericalCell, ...]

    def __post_init__(self):
        typed_tuple(self.cells, NumericalCell, 128)
        if not self.cells or len({c.key for c in self.cells}) != len(self.cells):
            raise ValueError("one to 128 uniquely keyed numerical cells required")
        bounded_text(self.payload)

    @property
    def payload(self):
        return canonical(dict(schema=REVISION, cells=[asdict(c) for c in self.cells]))

    @classmethod
    def from_payload(cls, text):
        bounded_text(text)
        def unique(pairs):
            result = {}
            for key, value in pairs:
                if key in result:
                    raise ValueError("duplicate table field")
                result[key] = value
            return result
        def reject(_):
            raise ValueError("table JSON contains an unsupported numeric literal")
        try:
            data = json.loads(text, object_pairs_hook=unique, parse_int=reject, parse_float=reject, parse_constant=reject)
        except RecursionError:
            raise ValueError("table nesting limit") from None
        if (type(data) is not dict or set(data) != {"schema", "cells"} or data["schema"] != REVISION
                or type(data["cells"]) is not list or not 1 <= len(data["cells"]) <= 128):
            raise ValueError("explicit bounded numerical table schema required")
        cells = []
        for row in data["cells"]:
            if type(row) is not dict or set(row) != {"key", "coordinates", "value", "unit"}:
                raise ValueError("explicit numerical cell schema required")
            coordinates = row["coordinates"]
            if type(coordinates) is not list or any(type(p) is not list or len(p) != 2 for p in coordinates):
                raise ValueError("coordinate pairs required")
            cells.append(NumericalCell(row["key"], tuple(tuple(p) for p in coordinates), row["value"], row["unit"]))
        return cls(tuple(cells))


@dataclass(frozen=True)
class Selector:
    """Exact coordinate match. No substring, fuzzy or case-folded aliasing."""

    coordinates: tuple[tuple[str, str], ...]

    def __post_init__(self):
        object.__setattr__(self, "coordinates", _coordinates(self.coordinates))


@dataclass(frozen=True)
class NumericalTask:
    operation: str
    selectors: tuple[Selector, ...]
    result_coordinate: str | None = None

    def __post_init__(self):
        if type(self.operation) is not str or self.operation not in OPERATIONS:
            raise ValueError("supported declared numerical operation required")
        typed_tuple(self.selectors, Selector, 128)
        if not self.selectors:
            raise ValueError("explicit task population required; no implicit all rows")
        if self.operation == "lookup" and len(self.selectors) != 1:
            raise ValueError("lookup requires one entity selector")
        if self.operation in ("subtract", "absdiff", "ratio", "percent_ratio", "greater", "less", "equal") and len(self.selectors) != 2:
            raise ValueError("binary operation requires two ordered selectors")
        if self.operation in ("argmin", "argmax"):
            identifier(self.result_coordinate)
            bounded_text(self.result_coordinate, 64)
        elif self.result_coordinate is not None:
            raise ValueError("only extrema return a coordinate")


def numerical_view(source, table, *, key):
    """Host-reviewed projection; inherits exact source dependency and access scope.

    The host must construct the cells from the source. This helper establishes
    lineage, not semantic equivalence between supplied cells and source content.
    """
    if type(source) is not CanonicalNode or type(table) is not NumericalTable:
        raise ValueError("typed source node and table required")
    return CanonicalNode(key=key, revision=hashlib.sha256(table.payload.encode()).hexdigest(),
        tenant=source.tenant, text=table.payload,
        kind="MODEL_OUTPUT" if source.kind == "MODEL_OUTPUT" else "DERIVED_RESULT",
        temporal=source.temporal, roles=source.roles, provenance=REVISION,
        dependencies=(source.ref,), modality="table")


@dataclass(frozen=True)
class NumericalPacket:
    task: NumericalTask
    snapshot: ContextSnapshot
    seal: str

    def __post_init__(self):
        if (type(self.task) is not NumericalTask or type(self.snapshot) is not ContextSnapshot
                or len(self.snapshot.records) != 1):
            raise ValueError("declared task and one authorized table view required")

    @property
    def payload(self):
        return canonical(dict(schema=REVISION, task=asdict(self.task), source=asdict(self.snapshot.records[0].node.ref)))

    @property
    def digest(self):
        return hashlib.sha256((self.snapshot.state_revision+"\0"+self.payload).encode()).hexdigest()

    def is_current(self, state):
        return state.verify_binding(self.snapshot, self.payload, self.seal)


@dataclass(frozen=True)
class NumericalResult:
    status: str
    answer: str | None
    rational: tuple[int, int] | None
    unit: str | None
    selected_cells: tuple[str, ...]
    source: NodeRef
    source_kind: str
    packet_digest: str
    task_execution_verified: bool
    source_semantics_verified: bool = field(default=False, init=False)
    question_interpretation_verified: bool = field(default=False, init=False)


def prepare_numerical(state, snapshot, source, task):
    if (not all(callable(getattr(state, name, None)) for name in ("subset", "seal", "verify_binding"))
            or type(source) is not NodeRef or type(task) is not NumericalTask):
        raise ValueError("host-owned state, exact source reference and task required")
    selected = state.subset(snapshot, (source,))
    NumericalTable.from_payload(selected.records[0].node.text)
    packet = NumericalPacket(task, selected, "")
    return NumericalPacket(task, selected, state.seal(selected, packet.payload))


def execute_numerical(state, packet):
    """Execute the declared task only. Recheck authorization before returning.

    An absent selector is absent from this view, not proof of real-world absence.
    Host authorization must be checked again before any later consequential use.
    """
    if not callable(getattr(state, "verify_binding", None)) or type(packet) is not NumericalPacket:
        raise ValueError("host-owned state and prepared task packet required")
    node = packet.snapshot.records[0].node
    def result(status, answer=None, value=None, unit=None, cells=()):
        if not packet.is_current(state):
            status, answer, value, unit, cells = "unavailable_context", None, None, None, ()
        rational = (value.numerator, value.denominator) if type(value) is Fraction else None
        return NumericalResult(status, answer, rational, unit, cells, node.ref, node.kind, packet.digest,
                               status == "computed")
    if not packet.is_current(state):
        return result("unavailable_context")
    table = NumericalTable.from_payload(node.text)
    index = {}
    for position, cell in enumerate(table.cells):
        for coordinate in cell.coordinates:
            index.setdefault(coordinate, set()).add(position)
    selected = []
    for selector in packet.task.selectors:
        postings = [index.get(coordinate, set()) for coordinate in selector.coordinates]
        postings.sort(key=len)
        matches = postings[0].intersection(*postings[1:])
        if not matches:
            return result("selector_not_in_view")
        if len(matches) != 1:
            return result("ambiguous_selector")
        selected.append(table.cells[next(iter(matches))])
    keys = tuple(c.key for c in selected)
    if len(set(keys)) != len(keys):
        return result("duplicate_selected_cell")
    op = packet.task.operation
    if op == "count":
        return result("computed", str(len(keys)), Fraction(len(keys)), "", keys)
    if any(c.value is None for c in selected):
        return result("unreadable_value")
    if len({c.unit for c in selected}) != 1:
        return result("incompatible_units")
    values, unit = [_number(c.value) for c in selected], selected[0].unit
    if op in ("argmin", "argmax"):
        extreme = (min if op == "argmin" else max)(values)
        winners = [c for c, v in zip(selected, values) if v == extreme]
        if len(winners) != 1:
            return result("ambiguous_extremum")
        answer = dict(winners[0].coordinates).get(packet.task.result_coordinate)
        if answer is None:
            return result("missing_result_coordinate")
        return result("computed", answer=answer, cells=keys)
    if op in ("greater", "less", "equal"):
        comparison = {"greater": values[0] > values[1], "less": values[0] < values[1], "equal": values[0] == values[1]}[op]
        return result("computed", "Yes" if comparison else "No", cells=keys)
    try:
        if op == "lookup":
            value = values[0]
        elif op in ("sum", "mean"):
            value = Fraction(0)
            for item in values:
                value = _bounded_fraction(value+item)
            if op == "mean":
                value = _bounded_fraction(value/len(values))
        elif op in ("min", "max"):
            value = (min if op == "min" else max)(values)
        elif op in ("subtract", "absdiff"):
            value = values[0]-values[1]
            if op == "absdiff":
                value = abs(value)
        else:
            if not values[1]:
                return result("undefined_division")
            value = values[0]/values[1]*(100 if op == "percent_ratio" else 1)
            unit = "%" if op == "percent_ratio" else ""
        _bounded_fraction(value)
    except ValueError:
        return result("arithmetic_limit")
    with localcontext() as context:
        context.prec = 28
        answer = format(Decimal(value.numerator)/Decimal(value.denominator), "f")
    if "." in answer:
        answer = answer.rstrip("0").rstrip(".")
    return result("computed", answer, value, unit, keys)

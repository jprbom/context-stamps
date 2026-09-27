"""Bounded rational arithmetic over model-extracted chart cells.

Copyright (c) 2026 Prashant Jagtap. MIT License.
Exact arithmetic does not verify the extraction or the model's choice of cells.
This experiment module never executes model-provided Python or shell commands.
"""

import json
import math
import re
from dataclasses import dataclass
from decimal import Decimal, localcontext
from fractions import Fraction

NUMBER = r"[+-]?(?:\d{1,24}(?:\.\d{1,24})?|\.\d{1,24})(?:[eE][+-]?\d{1,2})?%?"
FIELDS = ("label", "series", "color", "unit")


def strict_json(text, maximum=32768):
    if type(text) is not str or len(text.encode("utf-8")) > maximum:
        raise ValueError("bounded JSON text required")
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate JSON field")
            result[key] = value
        return result
    def reject(_):
        raise ValueError("nonfinite JSON number")
    try:
        value = json.loads(text, object_pairs_hook=unique, parse_constant=reject)
    except RecursionError:
        raise ValueError("JSON nesting exceeds parser limit") from None
    pending, count = [(value, 0)], 0
    while pending:
        item, depth = pending.pop()
        count += 1
        if depth > 32 or count > 4096:
            raise ValueError("JSON structure limit")
        if type(item) is float and not math.isfinite(item):
            raise ValueError("nonfinite JSON number")
        if type(item) in (list, dict):
            pending.extend((child, depth+1) for child in (item.values() if type(item) is dict else item))
    return value


def chart_cells(table):
    if type(table) is not dict or set(table) != {"title", "cells", "warnings"}:
        raise ValueError("explicit table fields required")
    if type(table["title"]) is not str or len(table["title"]) > 256:
        raise ValueError("bounded chart title required")
    if type(table["warnings"]) is not list or len(table["warnings"]) > 8 or any(type(w) is not str or len(w) > 256 for w in table["warnings"]):
        raise ValueError("bounded extraction warnings required")
    if type(table["cells"]) is not list or not 1 <= len(table["cells"]) <= 128:
        raise ValueError("one to 128 chart cells required")
    result = {}
    for cell in table["cells"]:
        if type(cell) is not dict or set(cell) != {"id", "value", *FIELDS}:
            raise ValueError("explicit cell fields required")
        key = cell["id"]
        if type(key) is not str or not re.fullmatch(r"c\d{1,3}", key) or key in result:
            raise ValueError("unique bounded cell id required")
        if any(type(cell[f]) is not str or len(cell[f]) > 256 for f in FIELDS):
            raise ValueError("bounded cell descriptors required")
        if cell["value"] is not None and (type(cell["value"]) is not str or len(cell["value"]) > 80):
            raise ValueError("bounded numeric text or explicit unknown required")
        result[key] = cell
    return result


@dataclass(frozen=True)
class Quantity:
    value: Fraction
    unit: str = ""

    def __post_init__(self):
        if abs(self.value.numerator) > 10**48 or self.value.denominator > 10**48:
            raise ValueError("rational representation limit")


def number(text, unit=""):
    if type(text) is not str or not re.fullmatch(NUMBER, text):
        raise ValueError("supported numeric text required; no inferred missing value")
    unit = unit.strip().casefold()
    if unit in ("percent", "percentage"):
        unit = "%"
    if text.endswith("%"):
        if unit not in ("", "%"):
            raise ValueError("conflicting percentage units")
        text, unit = text[:-1], "%"
    return Quantity(Fraction(text), unit)


def execute_program(table, question, program):
    cells = chart_cells(table)
    if type(question) is not str or not question.strip() or len(question.encode()) > 16384:
        raise ValueError("bounded complete question required")
    constants = set(re.findall(r"(?<![\w.,])(" + NUMBER + r")(?![\w%]|\.\d|,\d)", question))
    used, literals, nodes = set(), set(), 0

    def cell(key):
        if type(key) is not str or key not in cells:
            raise ValueError("unknown cell reference")
        used.add(key)
        return cells[key]

    def numeric(key):
        item = cell(key)
        return number(item["value"], item["unit"])

    def same_units(values):
        if not values or any(type(v) is not Quantity for v in values) or len({v.unit for v in values}) != 1:
            raise ValueError("compatible observed units required")
        return values[0].unit

    def visit(expr, depth=0):
        nonlocal nodes
        nodes += 1
        if nodes > 64 or depth > 8 or type(expr) is not dict:
            raise ValueError("bounded expression tree required")
        if set(expr) == {"constant"}:
            text = expr["constant"]
            if type(text) is not str or text not in constants:
                raise ValueError("constant must occur as numeric text in the question")
            literals.add(text)
            return number(text)
        if set(expr) in ({"cell"}, {"cell", "field"}):
            item = cell(expr["cell"])
            if "field" not in expr:
                return number(item["value"], item["unit"])
            if expr["field"] not in FIELDS or not item[expr["field"]]:
                raise ValueError("known cell descriptor required")
            return item[expr["field"]]
        op = expr.get("op")
        if type(op) is not str:
            raise ValueError("declared arithmetic operation required")
        if "cells" in expr:
            names = expr["cells"]
            if (set(expr) not in ({"op", "cells"}, {"op", "cells", "field"})
                    or type(names) is not list or not 1 <= len(names) <= 128
                    or any(type(n) is not str for n in names) or len(set(names)) != len(names)):
                raise ValueError("unique bounded selected cells required")
            for name in names:
                cell(name)
            if op == "count" and "field" not in expr:
                return Quantity(Fraction(len(names)))
            values = [numeric(name) for name in names]
            if op in ("argmin", "argmax"):
                same_units(values)
                field = expr.get("field", "label")
                if field not in FIELDS:
                    raise ValueError("known result field required")
                extreme = (min if op == "argmin" else max)(v.value for v in values)
                winners = [name for name, v in zip(names, values) if v.value == extreme]
                if len(winners) != 1 or not cells[winners[0]][field]:
                    raise ValueError("ambiguous or unlabeled extremum")
                return cells[winners[0]][field]
            if "field" in expr:
                raise ValueError("descriptor field applies only to extrema")
        else:
            if set(expr) != {"op", "args"} or type(expr["args"]) is not list or not 1 <= len(expr["args"]) <= 128:
                raise ValueError("bounded argument list required")
            values = [visit(arg, depth+1) for arg in expr["args"]]
        if any(type(v) is not Quantity for v in values):
            raise ValueError("arithmetic requires numeric arguments")
        if op in ("sum", "mean", "min", "max"):
            unit = same_units(values)
            if op in ("min", "max"):
                return Quantity((min if op == "min" else max)(v.value for v in values), unit)
            total = Fraction(0)
            for value in values:
                total = Quantity(total + value.value, unit).value
            return Quantity(total / len(values) if op == "mean" else total, unit)
        if len(values) != 2:
            raise ValueError("binary operation requires two arguments")
        a, b = values
        if op in ("subtract", "absdiff", "greater", "less", "equal", "ratio", "percent_ratio"):
            unit = same_units(values)
            if op in ("greater", "less", "equal"):
                return {"greater": a.value > b.value, "less": a.value < b.value, "equal": a.value == b.value}[op]
            if op in ("subtract", "absdiff"):
                difference = a.value - b.value
                return Quantity(abs(difference) if op == "absdiff" else difference, unit)
            if not b.value:
                raise ValueError("undefined division")
            return Quantity(a.value / b.value * (100 if op == "percent_ratio" else 1), "%" if op == "percent_ratio" else "")
        if op == "multiply":
            if a.unit and b.unit:
                raise ValueError("one dimensionless multiplier required")
            return Quantity(a.value * b.value, a.unit or b.unit)
        raise ValueError("unsupported arithmetic operation")

    result = visit(program)
    rational, unit = None, None
    if type(result) is Quantity:
        rational, unit = [result.value.numerator, result.value.denominator], result.unit
        with localcontext() as context:
            context.prec = 28
            answer = format(Decimal(rational[0]) / Decimal(rational[1]), "f")
        if "." in answer:
            answer = answer.rstrip("0").rstrip(".")
    elif type(result) is bool:
        answer = "Yes" if result else "No"
    else:
        answer = result
    return dict(answer=answer, rational=rational, unit=unit, used_cells=sorted(used),
                question_constants=sorted(literals), expression_nodes=nodes,
                execution_verified=True, source_semantics_verified=False)

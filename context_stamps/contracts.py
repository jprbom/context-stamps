"""Deterministic rendering for bounded declarative configuration tasks.

This is not a general code generator. It never executes generated code.
"""

import re
from collections.abc import Mapping


def render_integer_assignments(values, *, allowed_names):
    if not isinstance(values, Mapping) or not 1 <= len(values) <= 32:
        raise ValueError("one to 32 integer assignments required")
    names = tuple(allowed_names)
    if len(set(names)) != len(names) or set(values) != set(names):
        raise ValueError("assignment fields do not match the declared contract")
    if any(not isinstance(name, str) or not re.fullmatch(r"[A-Z][A-Z0-9_]{0,63}", name) for name in names):
        raise ValueError("bounded uppercase identifiers required")
    if any(type(value) is not int or not -(2**31) <= value < 2**31 for value in values.values()):
        raise ValueError("signed 32-bit integer values required")
    return "\n".join(f"{name} = {values[name]}" for name in sorted(names)) + "\n"

"""Bounded categorical decisions. Copyright 2026 Prashant Jagtap, MIT."""

from dataclasses import dataclass

from ..context_state import typed_tuple
from ..security import identifier


@dataclass(frozen=True)
class Choice:
    options: tuple[str, ...]

    def __post_init__(self):
        typed_tuple(self.options, str, 64)
        if len(self.options) < 2:
            raise ValueError("two to 64 distinct choices required")
        for option in self.options:
            identifier(option)

    def accepts(self, value):
        return type(value) is str and value in self.options

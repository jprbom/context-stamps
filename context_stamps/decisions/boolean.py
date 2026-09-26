"""Strict Boolean decisions. Copyright 2026 Prashant Jagtap, MIT."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Boolean:
    def accepts(self, value):
        return type(value) is bool

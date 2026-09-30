"""Predeclared low-dimensional local routing family for chart training outcomes.

Copyright (c) 2026 Prashant Jagtap. MIT License.
Features never use answer labels or competing-arm outputs. No candidate activates.
"""

import re

from context_stamps.local_learning import revision
from context_stamps.local_policy import PolicyObservation, fit_cell_policy

ACTIONS = ("direct", "memory", "program")
PENALTIES = (2.0, 10.0, 50.0)
MINIMUM_TASKS = 8
COST_CAP = 32768.0


def input_cell(question):
    if type(question) is not str or not question.strip() or len(question.encode()) > 16384:
        raise ValueError("bounded input question required")
    text = question.casefold()
    if re.search(r"\b(ratio|percent|percentage|proportion)\b", text):
        return "relative"
    if re.search(r"\b(sum|total|average|mean|difference|more|less|fewer)\b", text):
        return "arithmetic"
    if re.search(r"\b(highest|lowest|largest|smallest|maximum|minimum|most|least)\b", text):
        return "extreme"
    return "other"


def fit(rows, *, binding, penalty):
    """Rows are independently scored matched outcomes with extraction costs included."""
    if penalty not in PENALTIES:
        raise ValueError("predeclared penalty required")
    observed = tuple(PolicyObservation(row["id"], input_cell(row["question"]), row["mode"],
        row["correct"], row["tokens"]) for row in rows)
    return fit_cell_policy(observed, binding=binding, baseline_action="direct", cost_cap=COST_CAP,
        failure_penalty=penalty, minimum_tasks=MINIMUM_TASKS)


def environment(reader_registration_digest):
    return revision(dict(experiment="chartqa-training-diagnostic-v1", reader=reader_registration_digest,
        actions=ACTIONS, cells=["relative", "arithmetic", "extreme", "other"], penalties=PENALTIES,
        minimum_tasks=MINIMUM_TASKS, cost_cap=COST_CAP))

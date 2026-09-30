"""Offline typed numerical workflow with authored observations; no model calls."""

import json

from context_stamps.context_state import AccessScope, CanonicalNode, ContextState, TemporalScope
from context_stamps.numerical_tasks import (
    NumericalCell,
    NumericalTable,
    NumericalTask,
    Selector,
    execute_numerical,
    numerical_view,
    prepare_numerical,
)


def demonstration():
    state = ContextState(tenant="workshop", policy_revision="p1", clock=lambda: 10)
    source = CanonicalNode(key="readings", revision="v1", tenant="workshop",
        text="Authored example: probe A reads 0.1 V; probe B reads 0.2 V.", kind="OBSERVATION",
        temporal=TemporalScope(1, 1), roles=("reader",), provenance="authored-example")
    table = NumericalTable((
        NumericalCell("a", (("probe", "A"),), "0.1", "V"),
        NumericalCell("b", (("probe", "B"),), "0.2", "V"),
    ))
    view = numerical_view(source, table, key="numerical-readings")
    state.put(source)
    state.put(view)
    scope = AccessScope("workshop", "local-user", "p1", ("reader",))
    snapshot = state.snapshot(scope, at=10, known_at=10)
    task = NumericalTask("sum", (Selector((("probe", "A"),)), Selector((("probe", "B"),))))
    packet = prepare_numerical(state, snapshot, view.ref, task)
    answer = execute_numerical(state, packet)
    absent = prepare_numerical(state, snapshot, view.ref,
        NumericalTask("lookup", (Selector((("probe", "Z"),)),)))
    missing = execute_numerical(state, absent)
    state.set_roles(source.key, ("private",))
    revoked = execute_numerical(state, packet)
    return dict(example="authored typed tasks; not model or device evaluation", model_calls=0,
        answer=answer.answer, exact_rational=answer.rational, unit=answer.unit,
        execution_verified=answer.task_execution_verified,
        source_semantics_verified=answer.source_semantics_verified,
        language_interpretation_verified=answer.question_interpretation_verified,
        missing_status=missing.status, missing_answer=missing.answer,
        revoked_status=revoked.status, revoked_answer=revoked.answer)


if __name__ == "__main__":
    print(json.dumps(demonstration(), sort_keys=True))

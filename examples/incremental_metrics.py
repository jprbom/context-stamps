"""Incremental metrics from fictional evaluation counts; no trained model.

Copyright (c) 2026 Prashant Jagtap. MIT License.
Changing false positives recomputes that input, precision and F1. Recall and an
unrelated result remain reusable. These cheap functions do not need an LLM.
"""

import json
from dataclasses import replace
from fractions import Fraction

from context_stamps.context_state import AccessScope, CanonicalNode, ContextClaim, ContextState, TemporalScope
from context_stamps.experience import ResourceUse
from context_stamps.incremental import ComputeAdapter, ComputeOutput, ComputeSpec, IncrementalExecutor


def count_node(name, value, revision="v1", supersedes=()):
    return CanonicalNode(name, revision, "research", f"Fictional {name} count: {value}.", "OBSERVATION",
        TemporalScope(0, 10), ("reader",), "fictional-confusion-counts",
        claims=(ContextClaim("count", str(value)),), supersedes=supersedes, validation_revision="fixture-v1")


def calculate(inputs):
    if inputs.spec.operation_revision == "count-v1":
        values = [int(c.value) for r in inputs.records for c in r.node.claims if c.name == "count"]
        if len(values) != 1 or not 0 <= values[0] <= 1000000:
            raise ValueError("one bounded count required")
        return Fraction(values[0])
    values = {p.binding.key: Fraction(p.text) for p in inputs.parents}
    if inputs.spec.operation_revision == "ratio-v1":
        request = json.loads(inputs.spec.request)
        numerator, other = values[request["numerator"]], values[request["other"]]
        return numerator / (numerator + other) if numerator + other else Fraction(0)
    precision, recall = values["precision"], values["recall"]
    return 2 * precision * recall / (precision + recall) if precision + recall else Fraction(0)


def compute(inputs):
    return ComputeOutput(str(calculate(inputs)), ResourceUse(0, 0, 0, 0, 0))


def verify(inputs, text):
    # Exact deterministic arithmetic control; not an independent model judge.
    return Fraction(text) == calculate(inputs)


ADAPTERS = tuple(ComputeAdapter(name, "fraction-v1", compute, verify)
                 for name in ("count-v1", "ratio-v1", "harmonic-v1"))
TARGETS = ("f1", "other")


def fixture():
    state = ContextState(tenant="research", policy_revision="p0", clock=lambda: 100)
    scope = AccessScope("research", "scientist", "p0", ("reader",))
    sources = {name: count_node(name, value) for name, value in (("tp", 40), ("fp", 10), ("fn", 5), ("other", 99))}
    for source in sources.values():
        state.put(source)
    graph = tuple(ComputeSpec(name, "count-v1", "fraction-v1", (source.ref,)) for name, source in sources.items()) + (
        ComputeSpec("precision", "ratio-v1", "fraction-v1", parents=("tp", "fp"),
                    request='{"numerator":"tp","other":"fp"}'),
        ComputeSpec("recall", "ratio-v1", "fraction-v1", parents=("tp", "fn"),
                    request='{"numerator":"tp","other":"fn"}'),
        ComputeSpec("f1", "harmonic-v1", "fraction-v1", parents=("precision", "recall")),
    )
    return state, scope, sources, graph


def main():
    state, scope, sources, graph = fixture()
    executor = IncrementalExecutor(state, ADAPTERS, authorize=lambda s, spec: s == scope)
    first = executor.run(graph, TARGETS, scope=scope, at=200, known_at=200)
    repeated = executor.run(graph, TARGETS, scope=scope, at=200, known_at=200)
    changed = count_node("fp", 20, "v2", (sources["fp"].ref,))
    state.put(changed)
    graph = tuple(replace(s, sources=(changed.ref,)) if s.key == "fp" else s for s in graph)
    updated = executor.run(graph, TARGETS, scope=scope, at=200, known_at=200)
    assert all(r.status == "complete" for r in (first, repeated, updated))
    assert {s.key for s in updated.steps if s.status == "computed"} == {"fp", "precision", "f1"}
    assert [r.text for r in first.results] == ["16/19", "99"]
    assert [r.text for r in updated.results] == ["16/21", "99"]
    assert all(executor.verify_receipt(r.receipt, r.text) for r in updated.results)
    state.set_roles("fp", ())
    denied = executor.run(graph, TARGETS, scope=scope, at=200, known_at=200)
    assert denied.status == "unavailable_context" and not denied.results
    print(json.dumps({"initial_f1": first.results[0].text, "updated_f1": updated.results[0].text,
        "first_computations": sum(s.status == "computed" for s in first.steps),
        "repeat_computations": sum(s.status == "computed" for s in repeated.steps),
        "changed_computations": [s.key for s in updated.steps if s.status == "computed"],
        "reused_after_change": [s.key for s in updated.steps if s.status == "reused"],
        "revoked_run": denied.status, "model_calls": 0,
        "scope": "fictional counts, exact rational arithmetic; no training or model-quality claim"}, indent=2))


if __name__ == "__main__":
    main()

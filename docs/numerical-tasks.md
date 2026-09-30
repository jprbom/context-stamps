# Numerical tasks with explicit evidence requirements

By Prashant Jagtap

A calculation can execute correctly and still answer the wrong question. In the
[recorded visual check](../evidence/chartqa-scalar-v2/README.md), the reader used
44 minus 11 to answer a question about a category absent from the chart. The
arithmetic kernel correctly returned 33; the task was unsupported.

`context_stamps.numerical_tasks` gives applications a separate typed route. The
host supplies the operation, required entity coordinates and explicit population.
The runtime binds this task to an authorized versioned table, resolves exact
selectors and executes bounded rational arithmetic. A missing entity cannot be
replaced by other available cells. No language model is required on this route.

Use it for instrument readings, tables, inventory comparisons, experiment metrics
and numerical steps in agent workflows where the application already knows the
task. Run the complete offline example:

```powershell
python examples/numerical_tasks.py
```

## Minimal use

```python
from context_stamps.numerical_tasks import (
    NumericalCell, NumericalTable, NumericalTask, Selector,
    numerical_view, prepare_numerical, execute_numerical,
)

# source is a host-owned CanonicalNode already inserted into state.
# scope is authenticated by the host; at/known_at specify its requested view.
table = NumericalTable((
    NumericalCell("a", (("probe", "A"),), "0.1", "V"),
    NumericalCell("b", (("probe", "B"),), "0.2", "V"),
))
view = numerical_view(source, table, key="numerical-readings")
state.put(view)
snapshot = state.snapshot(scope, at=at, known_at=known_at)
task = NumericalTask("sum", (
    Selector((("probe", "A"),)),
    Selector((("probe", "B"),)),
))
packet = prepare_numerical(state, snapshot, view.ref, task)
result = execute_numerical(state, packet)
assert result.answer == "0.3"
assert result.rational == (3, 10)
```

The example assumes a trusted in-memory `ContextState`. A durable `ContextStore`
also works; its insertion calls additionally require the host scope and unique
mutation ID. The source must remain available, authorized and valid in the
requested temporal view. Recheck `packet.is_current(state)` before later use.
The packet is not an authorization lock over an external action.

Selectors match exact named coordinates. A label that occurs in multiple series
needs a series coordinate; the runtime does not guess. Coordinates are intersected
through an exact inverted index. There is no semantic hash match or fuzzy alias.
The table supports at most 128 cells, each with up to 16 named coordinates, within
the existing 64 KiB node bound. This is a bounded local numerical API, not a
large-scale analytical database.

## Operations and failure states

Supported operations are lookup, count, sum, mean, minimum, maximum, subtraction,
absolute difference, ratio, percentage ratio, comparisons and labelled extrema.
Use their API names `lookup`, `count`, `sum`, `mean`, `min`, `max`, `subtract`,
`absdiff`, `ratio`, `percent_ratio`, `greater`, `less`, `equal`, `argmin`, `argmax`.
For labelled extrema, supply `result_coordinate`, such as `"probe"`.

Population selectors are always explicit. For subtraction and ratios their order
matters. Selecting a subset does not establish a maximum over omitted records.
An application must supply a complete population when its task requires one.

| Result status | Meaning |
|---|---|
| `computed` | The declared task executed on its exact selected cells |
| `selector_not_in_view` | A requested entity was absent from the authorized table view |
| `ambiguous_selector` | More than one cell matched; add identifying coordinates |
| `duplicate_selected_cell` | Different selectors would double-count the same cell |
| `unreadable_value` | A required value is unknown; zero is not substituted |
| `incompatible_units` | Arithmetic needs matching units; no implicit scale conversion |
| `undefined_division` | The denominator is zero |
| `ambiguous_extremum` | A labelled minimum/maximum is tied |
| `missing_result_coordinate` | The winning cell lacks the requested output coordinate |
| `arithmetic_limit` | Exact intermediate/result representation exceeded its bound |
| `unavailable_context` | Source access, revision or packet binding is no longer valid |

Values use decimal strings, with units supplied separately. Arithmetic uses exact
rational values within bounded numerator/denominator sizes. The display answer
rounds recurring decimals to 28 significant digits; the returned rational pair
remains exact. A value of `25` with unit `%` stays on that scale. There is no
automatic currency, time-zone, measurement-unit or percentage conversion.

## What the result establishes

`task_execution_verified` is true only for a computed declared task.
`source_semantics_verified` and `question_interpretation_verified` remain false.
The host still needs independent evidence that the table reflects the source and
that the declared task expresses the user's intent. A model-generated task does
not acquire semantic verification merely because the host passes it to this API.

The table projection inherits the source dependency and initial access scope. A
projection of `MODEL_OUTPUT` remains `MODEL_OUTPUT`; other projections are
unvalidated `DERIVED_RESULT`. The library does not claim that supplied cells were
correctly extracted. Revoking a source invalidates the view and its task packet,
including through another durable-store writer and after restart.

Consequently, a successful calculation is not an independent positive label for
local model training. Only separately checked application outcomes can support
that label. A missing selector means absent from this view, not proof of absence
in the world. This distinction prevents an incomplete extraction from becoming
false negative knowledge.

The initial validation has 14 boundary tests, including the retained missing-
category scenario, unit scales, unreadable values, ties, exact decimals, altered
tasks/scopes and durable revocation. These are API tests. They do not establish a
ChartQA gain, a learned model improvement or target-edge efficiency.

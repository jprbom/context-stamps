# Declared numerical task checks

By Prashant Jagtap. Engineering evidence; no model or device benchmark.

The [numerical task API](../../docs/numerical-tasks.md) binds a host-declared
operation and exact entity selectors to an authorized table view. It refuses
missing or ambiguous selections before calculating. The runtime does not infer
that the host's task represents a natural-language request or that an extracted
table represents its original source correctly.

The complete local core suite passes **447 tests without skips**. Fourteen new
tests cover exact rational values, the absent-category failure, ambiguous series,
coordinate aliasing, units, zero/unknown values, explicit populations, ties,
scope/task changes, expiry, malformed data, and durable restart with a second
writer revoking the source. The source distribution and wheel both build; Bandit
and Ruff pass. The wheel was also installed into an isolated target using a
Python 3.12 environment and its example reproduced the recorded output with
zero model calls. An earlier install attempt used a CPU environment without
`pip` and stopped before installing anything.

The authored example returns 0.3 V as exact rational 3/10, refuses an absent probe,
and refuses the old packet after source revocation. It makes zero model calls.
The absent probe result means absent from this view, not a verified claim of
real-world absence. Neither source semantics nor language interpretation becomes
verified through calculation.

```powershell
python -m unittest discover -s tests -v
python examples/numerical_tasks.py
python experiments/verify_numerical.py
```

The manifest binds the recorded core log, example output and current reviewed
sources. The replay runs the repository example, never arbitrary archived code.
These checks do not revise the earlier visual scores or qualify local autonomous
learning. The separate ChartQA training reader still uses the frozen v3 model
protocol so its failures remain measurable; it does not silently use this typed
API as if it had understood a benchmark question.

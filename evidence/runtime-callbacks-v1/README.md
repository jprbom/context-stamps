# Runtime callback regression evidence

By Prashant Jagtap

The previous runtime held its shared state lock while trusted external callbacks
ran. An invalidation from another thread could therefore wait for retrieval,
tokenization, verification or model computation to finish. Exact pre-change source
bytes remain in `engineering-sources-v1/runtime-callbacks-initial.json.gz`.

The revised runtime snapshots state, invokes callbacks outside that lock and
rechecks context epoch and receipt validity before accepting output. Concurrent
exact requests share a bounded pending computation; unrelated computations can
progress independently. Each returned result still needs verification.

Eleven regression tests use events to assert ordering rather than timing a
claimed speedup. They cover routing, retrieval, token counting, context and result
verification, computation, cached result verification, revocation of waiting
requests, receipt expiry, exception cleanup, same-identity recursion and capacity
exhaustion. Full-suite logs and source digests are in the manifest. Tests run on
CPU. The Python 3.12 environment skips 16 optional dependency tests; the Python
3.13 environment runs all 339 tests without skips.

This evidence does not establish production throughput, hard callback
cancellation, embedded-device performance, model accuracy or automatic local
weight improvement. A callback already given evidence cannot have that evidence
withdrawn. Host adapters need execution deadlines and request admission limits.

```bash
python -m unittest discover -s tests -p test_runtime_callbacks.py -v
python experiments/verify_runtime_callbacks.py
```

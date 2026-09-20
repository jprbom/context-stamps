# Partial-facet queries and selective context reuse

Private v0.3.3 candidate. By Prashant Jagtap. The document stamp remains 256 bits / 32 bytes. The latest model training remains the three-seed ITQ study; this update changes query handling and cache invalidation, not model weights.

## Query only what is known

`FacetQuery` accepts a nonempty `SphericalStamp` containing only observed query views. It compares those views against a full document stamp and optionally accepts positive per-view weights. Missing values are not filled with fabricated content, zeros or arbitrary labels. A candidate missing an observed view, or using an incompatible family, is rejected.

```python
from context_stamps import FacetQuery, Family, HashingEncoder, SphericalStamp, Stamp256Codec

encoder = HashingEncoder(64)
families = {name: Family(encoder.identity, 64, 64, 17 + i)
            for i, name in enumerate(("content", "entity", "intent", "task"))}
codec = Stamp256Codec(families)
document = codec.encode({name: encoder.encode(value) for name, value in {
    "content": "Update timeout", "entity": "worker_alpha", "intent": "implement", "task": "timeout",
}.items()})
query = FacetQuery(SphericalStamp.encode(
    {"content": encoder.encode("Update timeout")}, {"content": families["content"]},
))
assert set(query.compare(document)) == {"content"}
assert len(codec.pack(document)) == 32
scope = query.policy_scope("my-project:" + codec.schema.identity)
```

Use this scope when calibrating and applying a `RoutingPolicy`. It includes the application scope, observed facet names, encoder families, bit widths and normalized weights. Query values are excluded because calibration applies to a cohort, not an exact query. Changing the observed mask or weights creates a new scope. The host must still identify the actual domain and candidate schema, keep validation independent and supply authorization.

The score is a ranking utility, not a probability. Fewer observed views can make candidates ambiguous; a compatible scope does not prove confidence. `ProgressiveRouter` still defaults to the precise backend. Exact required entity/task constraints must remain host checks; omitting a facet does not waive them. This API does not extract or learn facets automatically.

Run `python examples/partial_facets.py` for the complete offline example.

## Invalidate affected receipts, preserve unrelated work

Each cached packet records its complete declared dependency closure. `ContextSession` maintains a reverse index from a source to every receipt containing it.

- `put(node)` invalidates receipts containing that node, including same-revision content changes and role changes.
- `link(source, target, ...)` invalidates receipts containing the source endpoint. This also covers a newly added dependency whose target was absent from the old packet.
- Expiry and LRU eviction remove receipt entries from the reverse index.
- Unrelated receipts remain valid. A lock serializes mutations and reads.

The original v0.3.2 implementation cleared every receipt on every successful mutation. Its exact source is retained in `evidence/source-snapshots` and loaded as the comparison baseline in the replay. A packet cannot remain cached across a change to one of its declared dependencies. Undeclared dependencies remain the host's responsibility. Remote transactions and invalidation of already delivered plaintext are not supplied by this API.

## Measured changing-context replay

The replay uses actual text from 20 source files with declared independent dependency pairs. There are 20 rounds; each round after the first changes one dependency at the same revision and revalidates its relationship. The larger 100-pair case repeats source text under synthetic IDs. It is not a real coding-task benchmark or 100 independent projects.

| Pairs | Method | Reused / handoffs | Transfer payload bytes | Handoff ms | Mutation ms |
|---:|---|---:|---:|---:|---:|
| 10 | Exact graph | 0 / 200 | 1,534,795 | 3.1526 | 0.2623 |
| 10 | Global cache | 0 / 200 | 1,541,195 | 4.3760 | 0.3809 |
| 10 | Selective cache | 171 / 200 | 229,928 | 1.4239 | 0.2576 |
| 100 | Exact graph | 0 / 2,000 | 15,326,055 | 40.2343 | 0.5958 |
| 100 | Global cache | 0 / 2,000 | 15,390,055 | 57.6539 | 1.3370 |
| 100 | Selective cache | 1,881 / 2,000 | 977,056 | 24.5561 | 0.4908 |

![Changing-context comparison](assets/selective-invalidation.png)

All **6,600 packets** match their corresponding uncached control by SHA-256. **152 receipt checks** confirm affected receipts are unavailable and unrelated selective-cache receipts remain available. Setup timings are recorded separately in `setups.json`; the table is warm-operation timing, not total deployment cost. Method order is randomized per round. Additional receipt-correctness checks occur outside the timed handoff loop.

Transfer values count first packet delivery plus a 32-byte receipt, then 32 bytes per reuse. They assume a populated shared resolver and exclude transport headers/authentication. Both methods resolve identical evidence, so these results establish no reduction in model input tokens. Local sub-millisecond packet times do not establish end-to-end agent/network latency or universal superiority over other cache implementations.

[Protocol](../evidence/mutation-reuse-v1/protocol.json) · [records](../evidence/mutation-reuse-v1/results.json) · [summary](../evidence/mutation-reuse-v1/summary.json) · [source bindings](../evidence/mutation-reuse-v1/manifest.json).

## Verification

```bash
python -m unittest discover -s tests -v
python experiments/verify_mutation_reuse.py
python examples/progressive_context.py
python examples/partial_facets.py
# Re-execution regenerates the evidence directory; use a separate checkout.
python experiments/run_mutation_reuse.py
```

The 102-test suite includes 4,000 comparisons with an uncached graph over 500 mutation steps, covering cycles, stale edges, new dependencies and denied roles. The replay verifier independently checks saved packet equality, expected cache hits, receipt outcomes and aggregate accounting. Timings are observations, not exactly reproducible assertions.

## Remaining gaps

The standalone 32-byte representation still trails dense retrieval. The current public-data confidence policy is disabled; precise fallback protects ranking quality. Query-conditioned learned facets, naturally occurring unseen agent tasks, distributed persistence, multi-tenant authentication, device energy and changes to internal neural attention remain open. No public release or new model-quality claim follows from these local cache improvements.

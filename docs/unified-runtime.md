# Use the context runtime

By Prashant Jagtap

`ContextRuntime` composes registered retrieval experts, versioned dependency closure, context budgets, an application verifier and exact computation reuse. The core works offline with the standard library. GPU reranking and the trained expert router are optional.

```bash
git clone https://github.com/jprbom/context-stamps.git
cd context-stamps
python -m pip install -e .
python examples/unified_context.py
```

## Prepare context and verify an outcome

```python
import hashlib
from context_stamps import (
    ContextExpert, ContextNode, ContextRuntime, RuntimeBudget, Verification,
)

runtime = ContextRuntime(
    tenant="lab", principal="researcher", role="reader", policy="policy-v1",
)
runtime.put(ContextNode("config", "batch_size=32", "v1", frozenset({"reader"})))
runtime.put(ContextNode("metric", "validation_loss=0.24", "v1", frozenset({"reader"})))
runtime.link("metric", "config", "depends_on", provenance="experiment-manifest")

# Replace this small adapter with dense, hybrid or a calibrated reranker.
# It receives only the supplied eligible IDs that are current and permitted.
expert = ContextExpert("baseline", lambda request: ["metric"], estimated_ms=1)
packet = runtime.prepare_context(
    "Review this experiment", eligible=["config", "metric"],
    experts=[expert], baseline="baseline", scope="lab-profile-v1", limit=1,
    verifier=lambda p: Verification({"config", "metric"} <= set(p.sources)),
    budget=RuntimeBudget(bytes=2048, milliseconds=1000, iterations=2),
)
assert packet.status == "complete"

settings = dict(
    request_digest=hashlib.sha256(b"extract-batch-size:deterministic-v1").hexdigest(),
    model="exact-parser-v1", prompt="batch-size-v1", tool="parser",
    verifier_revision="v1", compute=lambda text: "32",
    verify=lambda value, text: value == "32" and "batch_size=32" in text,
)
assert not runtime.run_verified(packet, **settings)["reused"]
assert runtime.run_verified(packet, **settings)["reused"]
runtime.invalidate("config")
assert runtime.resolve(packet.receipt).status == "insufficient"
```

The verifier defines application success. In a research workflow it can check required datasets and checkpoint revisions; in coding it can check source/test bindings. The package does not infer scientific truth or execute untrusted code to check a claim. A missing-evidence verdict such as `Verification(False, ("config",))` requests another exact source within the same eligible set. The loop stops on completion, repeated state, missing permissions, exhausted bytes/tokens, deadline or iteration count. This is bounded evidence recovery, not self-modifying model weights.

`record_outcome` retains at most 128 host-provided outcome records without source text. Outcomes do not automatically train or promote a policy.

## Budgets and expert selection

Pass the intended reader's exact tokenizer as `token_counter` when creating the runtime, then use `RuntimeBudget(tokens=...)`. The counter sees the entire serialized evidence packet, including source/revision/digest headers. A byte limit is always separate. Missing tokenizers cause an error rather than an implicit word-count estimate. Reserve room for system instructions, the user query, tools and output separately in the reader adapter.

Register the known-quality method as `baseline`. An optional `choose(request)` callback may propose another registered expert, but the runtime accepts it only for an explicitly approved scope. A scope should bind the domain, model revisions, corpus snapshot policy, facets, candidate recipe and score calibration. Approval is supplied by the trusted host after evaluation; the runtime cannot authenticate a claim that a policy is qualified.

A latency estimate never authorizes a quality downgrade. If the selected qualified expert cannot fit, the result is `insufficient`. Deadline checks run before and after callbacks; a Python callback cannot be forcibly cancelled here. Network/process adapters must enforce their own timeout and cancellation.

## Concurrent callbacks and revocation

Retrieval, routing, token counting, application verification and model computation run outside the runtime's shared state lock. A slow callback therefore does not prevent another thread from invalidating evidence. Context changes during a callback reject its result. Receipt expiry is checked again before releasing a prepared packet or verified computation. Revocation cannot withdraw text already delivered to a trusted callback or interrupt that callback; it prevents acceptance and reuse of its obsolete result.

Concurrent requests with the same complete computation identity and context epoch share one pending computation. Each caller still runs its own verifier. Requests with different identities can progress independently. Pass `max_inflight=16` to `ContextRuntime` to set the maximum number of distinct outstanding computations (default 16, allowed 1–256). Further uncached identities return `insufficient` at that limit; exact cached results remain available subject to verification. The host must separately bound incoming requests, waiting threads and adapter execution time.

Invalidation wakes waiting requests immediately. An adapter exception releases its pending slot and wakes waiters; a waiting request may then retry the computation. Recursive computation of the same identity on the same thread raises `ValueError` instead of deadlocking. Callbacks remain trusted, pure/idempotent host code. This is local concurrency control, not a distributed serving or GPU scheduling system, and no production throughput improvement is inferred from the regression tests.

## Optional accelerated reranking

Install the controller extra in a working CUDA environment. Load a reviewed, pinned BERT cross-encoder and tokenizer with safetensors, `trust_remote_code=False` and a local revision. Then:

```python
from context_stamps.efficient_reranker import EfficientReranker

scorer = EfficientReranker(tokenizer, model, precision="autocast_bf16")
scores = scorer.score(query_text, authorized_candidate_texts)
```

The default keeps FP32 weights and uses BF16 matrix arithmetic, matching the recorded reference precision. It groups pairs by length and reuses passage token IDs under exact content digests. It retains the candidate set and 512-token pair limit. The cache has bounded numeric storage and entry count; Python bookkeeping is additional. Cold-cache and warm-cache measurements must be reported separately.

Full FP16/BF16 weight conversion is available for experiments but changes scores and is not the approved reference-preserving profile. The previous small-student int8 experiment did not improve CPU latency. Choosing a lower bit width without a suitable operator and numerical gate is not an optimization claim.

The tiny trained expert router uses int16-rounded coefficients with an input-dependent rounding bound and FP64 fallback near the decision boundary. It retains the original coefficients and uses floating-point accumulation: this is a decision-fidelity experiment, not an integer-kernel speed or total-memory reduction. It does not quantize the transformer. Its current cheap-route calibration failed; the published router therefore approves no scopes. Use it for reproduction, not an automatic quality-saving claim.

## Host responsibilities

- Use one runtime and reranker cache per trusted isolation boundary. A stamp or receipt never grants permission.
- Supply current source revisions and explicit relationships. After changing a dependency, revalidate the relationship before reusing its result.
- Bind every generation setting and external input to `request_digest`. Bind changes in model, prompt, tools, policy and verifier to their identities.
- Cache only pure/idempotent computations with complete inputs. Verify every returned result, including reuse.
- Exact computation reuse supports at most 128 source bindings. Larger closures fail closed in `run_verified`.
- Treat source text as data. Register trusted callbacks explicitly; the runtime never executes instructions found in retrieved text.
- Test target workloads and hardware. The bounded local implementation does not establish service-level latency, universal model gains or comprehensive security.

See [measured runtime results](runtime-v1-results.md) and the [remaining research gates](unified-context-roadmap.md).

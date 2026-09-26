"""Run offline: python examples/unified_context.py."""

import hashlib

from context_stamps import ContextExpert, ContextNode, ContextRuntime, RuntimeBudget, Verification

runtime = ContextRuntime(tenant="research-lab", principal="researcher", role="reader", policy="review-v1")
runtime.put(ContextNode("config", "batch_size=32", "v1", frozenset({"reader"})))
runtime.put(ContextNode("metrics", "validation_loss=0.24", "v1", frozenset({"reader"})))
runtime.link("metrics", "config", "depends_on", provenance="experiment-manifest")

# Replace with a real retriever. It receives only eligible, current source IDs.
expert = ContextExpert("baseline", lambda request: ["metrics"], estimated_ms=1)
context = runtime.prepare_context("Review this experiment", eligible=["config", "metrics"],
    experts=[expert], baseline="baseline", scope="lab-v1", limit=1,
    verifier=lambda packet: Verification({"config", "metrics"} <= set(packet.sources)),
    budget=RuntimeBudget(bytes=2048, milliseconds=1000, iterations=2))
assert context.status == "complete"
print(context.text)

# Host-owned, deterministic example. For a model, include all generation settings
# in request_digest and use an application-specific verifier.
options = dict(request_digest=hashlib.sha256(b"extract-batch-size:deterministic-v1").hexdigest(),
    model="exact-parser-v1", prompt="batch-size-v1", tool="parser", verifier_revision="v1",
    compute=lambda text: "32", verify=lambda result, text: result == "32" and "batch_size=32" in text)
assert not runtime.run_verified(context, **options)["reused"]
assert runtime.run_verified(context, **options)["reused"]
runtime.invalidate("config")
assert runtime.resolve(context.receipt).status == "insufficient"
print("Verified reuse succeeded; dependency invalidation revoked the receipt.")

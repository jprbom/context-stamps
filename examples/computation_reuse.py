"""Run offline: python examples/computation_reuse.py."""

import hashlib
from dataclasses import replace

from context_stamps.computation import ComputationCache, ComputationIdentity, exact_decimal

digest = hashlib.sha256(b"batch_size=32").hexdigest()
binding = ComputationIdentity(
    tenant="lab", principal="researcher", model="model-revision", prompt="prompt-v1",
    tool="decimal-v1", policy="policy-v1", request=digest,
    sources=(("training-config", "v1", digest),),
)
cache = ComputationCache()
cache.put(binding, exact_decimal("multiply", "32", "4"))
assert cache.get(binding) == "128"
assert cache.get(replace(binding, model="new-model-revision")) is None
assert cache.get(replace(binding, policy="revoked-policy-v2")) is None
print("Repeated computation reused; model and policy changes invalidate it.")

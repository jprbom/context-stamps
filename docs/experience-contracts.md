# Reference-only experience contracts

By Prashant Jagtap

`context_stamps.experience` supplies strict, immutable records for the learning/audit plane. It has no mandatory dependencies. Schema 1 remains byte-compatible for existing measured records; schema 2 adds dispatch/reconciliation and unknown token/call usage. Persistence is provided by [AuditStore](durable-audit.md) and owned-worker execution by [ManagedExecutor](managed-execution.md). These contracts do not grant evidence access or train a model.

```python
from hashlib import sha256
from context_stamps.experience import (
    Action, Authority, Episode, EvidenceRef, ResourceUse,
    TransitionOutcome, TransitionPlan, decode_record, encode_record,
    validate_transition,
)

# Fictional example. Put actual payloads in authorized host storage.
digest = sha256(b"batch_size=32").hexdigest()
source = EvidenceRef("lab", "config", "v1", digest, "observation")
episode = Episode(
    "episode-1", "research", "project-a", "template-a", "development",
    "environment-v1", 7, Authority("lab", "researcher", "policy-v1"),
    "exact-parser-v1", "verifier-v1", 1000,
)
action = Action("read-config", "tool-v1", source, "pure", digest)
plan = TransitionPlan("episode-1", 0, (source,), "belief-v1", action, None, None, 1001)
outcome = TransitionOutcome(
    "episode-1", 0, action, (source,), "succeeded", True, "verifier-v1", None,
    ResourceUse(input_tokens=0, output_tokens=0, model_calls=0, tool_calls=1, wall_ms=1.2),
    (), 1002,
)
validate_transition(episode, plan, outcome)
assert decode_record(encode_record(outcome)) == outcome
```

| Record | Purpose |
|---|---|
| `Episode` | Task family, project/template, split, environment, seed, trusted identity and model/verifier revisions |
| `TransitionPlan` | Observed references, belief revision, proposed action and separately typed prediction/uncertainty |
| `TransitionOutcome` | Actual action, observed consequences, verified/failed/abstained status, costs and reward components |
| `EpisodeEnd` | Terminal status and end timestamp |
| `DispatchClaim` | Exclusive proposal attempt, exact plan/action binding and deadline |
| `ActionReconciliation` | Verified terminal provider effect status, preserving the original outcome |
| `EvidenceRef` | Tenant, source, revision, content hash, and observation versus prediction kind |

Serialization rejects unknown fields, duplicate JSON keys, unsupported versions, unbounded identifiers/arrays, nonfinite costs and invalid success records. Transition validation rejects foreign tenants, changed action/verifier bindings and backwards timestamps. Numeric resource fields reject Boolean values. Unmeasured tokens, calls, RAM, VRAM and energy are `None`, not zero. Wall time is a measured host duration; adapter usage claims retain their host trust boundary.

The five unsuccessful states—failed, abstained, timed_out, cancelled and unknown—remain valid records with explicit reason codes. Keep these examples when building a training export. Do not silently drop them or label a timeout as a correct answer.

Timestamps are supplied by the trusted host. They do not prove that a prediction existed before its consequence. The [transactional audit journal](durable-audit.md) now enforces plan-before-outcome order within the log and records its own commit sequence/time; the host execution path must still commit before calling an external action. A predicted payload remains in separately authorized storage; its reference cannot appear in the observations field. A `verified=True` label is an assertion from the host verifier, not a new independent proof of truth.

Use opaque IDs, never credential strings or raw private text. Strict structure cannot detect a secret disguised as a valid identifier. The audit store requires current host authorization callbacks and checks exact event/action bindings. Evidence access, retention policy, key custody and permitted exports still need host integration. Do not expose these constructors as an unauthenticated service.

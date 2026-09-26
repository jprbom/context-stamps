"""Offline fictional experiment approval using temporal evidence and typed decisions.

Copyright (c) 2026 Prashant Jagtap. MIT License.
The exact provider demonstrates the interface; it is not a trained model.
"""

import json

from context_stamps.context_compiler import CompileBudget, ContextCompiler, ContextTask, ModelProfile
from context_stamps.context_state import (
    AccessScope,
    CanonicalNode,
    ContextClaim,
    ContextState,
    EvidenceRequirement,
    TemporalScope,
)
from context_stamps.decisions import Boolean, Proposal, Question, decide_batch


def main():
    state = ContextState(tenant="research", policy_revision="access-v1", clock=lambda: 1000)
    scope = AccessScope("research", "scientist", "access-v1", ("researcher",))
    policy = CanonicalNode(
        key="experiment-policy", revision="v1", tenant="research",
        text="The reviewed experiment permits a maximum batch size of 128.",
        kind="POLICY", temporal=TemporalScope(valid_from=0, observed_at=900),
        roles=("researcher",), provenance="approved-experiment-manifest",
        claims=(ContextClaim("batch_limit", "128"),), validation_revision="policy-review-v1",
    )
    state.put(policy)
    requirement = EvidenceRequirement("batch_limit", kinds=("POLICY",))
    task = ContextTask("check-batch", "Can the experiment use batch size 64?", (requirement,))
    context = ContextCompiler(state).compile(
        task, scope=scope, at=1000, known_at=1000,
        profile=ModelProfile("exact-demo-v1"), budget=CompileBudget(bytes=4096, milliseconds=5000),
    )
    question = Question("allowed", task.question, Boolean(), (requirement,), "experiment-batch-v1")

    def provider(questions, text):
        payload = json.loads(text)
        limit = next(c["value"] for r in payload["evidence"] for c in r["claims"] if c["name"] == "batch_limit")
        return (Proposal(questions[0].question_id, 64 <= int(limit)),)

    def verify(q, value, packet):
        # An application-owned deterministic check, separate from provider output.
        limits = {int(c.value) for r in packet.snapshot.records for c in r.node.claims if c.name == "batch_limit"}
        return len(limits) == 1 and value is (64 <= limits.pop())

    args = dict(context=context, state=state, provider=provider, verify=verify, verifier_revision="batch-check-v1")
    decision = decide_batch((question,), **args)[0]
    if decision.abstained or decision.result is not True:
        raise RuntimeError("expected verified fixture decision")
    print(json.dumps(decision.to_dict(), indent=2))
    state.set_roles(policy.key, ())
    revoked = decide_batch((question,), **args)[0]
    if not revoked.abstained or revoked.reason != "state_changed":
        raise RuntimeError("revoked evidence must not be reused")
    print("After revocation:", revoked.reason)


if __name__ == "__main__":
    main()

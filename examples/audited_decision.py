"""Complete offline context -> plan -> typed decision -> durable receipt example.

Copyright (c) 2026 Prashant Jagtap. MIT License.
Fictional records, deterministic provider, temporary private storage; no model.
"""

import hashlib
import json
import secrets
import tempfile
import time
from dataclasses import asdict
from pathlib import Path

from context_stamps.audit import AuditStore
from context_stamps.context_compiler import (
    CompileBudget,
    ContextCompiler,
    ContextTask,
    ModelProfile,
    compiled_material,
)
from context_stamps.context_state import (
    AccessScope,
    CanonicalNode,
    ContextClaim,
    ContextState,
    EvidenceRequirement,
    TemporalScope,
    canonical,
)
from context_stamps.decisions import Boolean, Proposal, Question, decide_batch
from context_stamps.experience import (
    Action,
    Authority,
    Episode,
    EpisodeEnd,
    EvidenceRef,
    ResourceUse,
    TransitionOutcome,
    TransitionPlan,
    references,
)


def main():
    # A real host supplies its authenticated principal, durable key management,
    # evidence storage and an independently retained latest journal checkpoint.
    actor = Authority("lab", "researcher", "access-v1")
    now = lambda: time.time_ns() // 1000000  # noqa: E731
    state = ContextState(tenant="lab", policy_revision="access-v1")
    scope = AccessScope("lab", "researcher", "access-v1", ("reader",))
    source = CanonicalNode("policy", "v1", "lab", "Reviewed experiment batch size must not exceed 128.",
        "POLICY", TemporalScope(0, now()), ("reader",), "reviewed-manifest",
        claims=(ContextClaim("batch_limit", "128"),), validation_revision="review-v1")
    state.put(source)
    required = EvidenceRequirement("batch_limit", kinds=("POLICY",))
    task = ContextTask("approve-run", "May this run use batch size 64?", (required,))
    context = ContextCompiler(state).compile(task, scope=scope, at=now(), known_at=now(),
        profile=ModelProfile("exact-demo-v1"), budget=CompileBudget(bytes=8192, milliseconds=5000))
    if context.status != "complete":
        raise RuntimeError("fixture context was not compiled")
    question = Question("allowed", task.question, Boolean(), (required,), "batch-policy-v1")
    revoked = set()

    def authorize(authority, operation, record):
        return authority == actor and (record is None or all(ref.source not in revoked for ref in references(record)))

    with tempfile.TemporaryDirectory(prefix="context-audit-example-") as directory:
        root = Path(directory)
        artifacts = root / "artifacts"
        artifacts.mkdir()

        def put_artifact(name, data):
            # Fixed example names; not a generic path resolver or public file service.
            raw = canonical(data).encode("utf-8")
            (artifacts / (name + ".json")).write_bytes(raw)
            return EvidenceRef("lab", name, "v1", hashlib.sha256(raw).hexdigest(), "observation")

        observed = put_artifact("policy", asdict(source))
        request = put_artifact("request", {"packet": compiled_material(context), "binding": context.binding,
                                           "question_schema": question.schema_id})
        episode = Episode("experiment-1", "batch-policy", "fictional-lab", "approval-v1", "development",
                          "offline-v1", 7, actor, "exact-demo-v1", "limit-check-v1", now())
        action = Action("typed-policy-check", "v1", request, "pure", request.digest)
        plan = TransitionPlan(episode.episode_id, 0, (observed,), "compiled-v1", action, None, None, now())
        signing_key = secrets.token_bytes(32)
        journal = root / "audit.sqlite"
        with AuditStore(journal, tenant="lab", signing_key=signing_key, authorize=authorize, create=True) as audit:
            audit.append(episode, authority=actor)
            audit.append(plan, authority=actor)  # Transaction commits before provider execution.
            calls = []

            def provider(questions, text):
                calls.append(1)
                value = next(c["value"] for row in json.loads(text)["evidence"] for c in row["claims"]
                             if c["name"] == "batch_limit")
                return (Proposal(questions[0].question_id, 64 <= int(value)),)

            def verify(q, result, packet):
                limits = {int(c.value) for row in packet.snapshot.records for c in row.node.claims if c.name == "batch_limit"}
                return len(limits) == 1 and result is (64 <= limits.pop())

            start = time.perf_counter()
            decisions = decide_batch((question,), context=context, state=state, provider=provider, verify=verify,
                                     verifier_revision="limit-check-v1")
            elapsed = (time.perf_counter() - start) * 1000
            decision = decisions[0]
            if decision.abstained or decision.result is not True or len(calls) != 1:
                raise RuntimeError("fixture decision failed; the committed plan remains unresolved")
            output = put_artifact("decision", decision.to_dict())
            outcome = TransitionOutcome(episode.episode_id, 0, action, (output,), "succeeded", True,
                "limit-check-v1", None, ResourceUse(0, 0, 0, 1, elapsed), (), now())
            # Token/model-call counts are zero because this provider is a pure Python function.
            audit.append(outcome, authority=actor)
            audit.append(EpisodeEnd(episode.episode_id, "succeeded", now()), authority=actor)
            checkpoint = audit.checkpoint(authority=actor)

        with AuditStore(journal, tenant="lab", signing_key=signing_key, authorize=authorize, checkpoint=checkpoint) as audit:
            audit.verify(authority=actor, checkpoint=checkpoint)
            history = audit.history(episode.episode_id, authority=actor)
            verified_refs = set()
            for entry in history:
                for ref in references(entry.record):
                    raw = (artifacts / (ref.source + ".json")).read_bytes()
                    if hashlib.sha256(raw).hexdigest() != ref.digest:
                        raise RuntimeError("external evidence changed")
                    verified_refs.add(ref)
            revoked.add("policy")
            try:
                audit.history(episode.episode_id, authority=actor)
            except PermissionError:
                revocation_blocks_history = True
            else:
                raise RuntimeError("revocation should hide the complete history")
        summary = dict(verified_decision=decision.result, committed_events=checkpoint.sequence,
            external_references_verified=len(verified_refs), reopened_and_verified=True,
            revocation_blocks_history=revocation_blocks_history, model_calls=0,
            scope="fictional deterministic demonstration; no model-quality or exactly-once execution claim")
        print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

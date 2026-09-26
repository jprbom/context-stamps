"""Offline compiled context -> managed worker -> typed decision -> durable replay.

Copyright (c) 2026 Prashant Jagtap. MIT License.
Fictional policy and deterministic functions. No GPU, model or external service.
"""

import hashlib
import json
import secrets
import tempfile
import time
from pathlib import Path

from context_stamps.audit import AuditStore, DispatchBlockedError
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
from context_stamps.execution import ExecutionAdapter, ManagedExecutor, WorkerResult
from context_stamps.experience import (
    Action,
    Authority,
    Episode,
    EpisodeEnd,
    EvidenceRef,
    ResourceUse,
    TransitionPlan,
    references,
)


def policy_provider(payload, idempotency_key, deadline_ms):
    data = json.loads(payload)
    limit = next(int(c["value"]) for row in data["context"]["evidence"]
                 for c in row["claims"] if c["name"] == "batch_limit")
    answer = {"question_id": "allowed", "value": data["requested_batch"] <= limit}
    return WorkerResult(canonical(answer).encode(), ResourceUse(0, 0, 0, 1, 0.))


def policy_verifier(payload, output):
    request, answer = json.loads(payload), json.loads(output)
    limits = {int(c["value"]) for row in request["context"]["evidence"] if row["kind"] == "POLICY"
              for c in row["claims"] if c["name"] == "batch_limit"}
    return (set(answer) == {"question_id", "value"} and answer["question_id"] == "allowed"
            and type(answer["value"]) is bool and len(limits) == 1
            and answer["value"] is (request["requested_batch"] <= limits.pop()))


def main():
    now = lambda: time.time_ns() // 1000000  # noqa: E731
    actor = Authority("lab", "researcher", "access-v1")
    scope = AccessScope("lab", "researcher", "access-v1", ("reader",))
    state = ContextState(tenant="lab", policy_revision="access-v1")
    source = CanonicalNode("policy", "v1", "lab", "Reviewed experiment batch size must not exceed 128.",
        "POLICY", TemporalScope(0, now()), ("reader",), "reviewed-manifest",
        claims=(ContextClaim("batch_limit", "128"),), validation_revision="review-v1")
    state.put(source)
    required = EvidenceRequirement("batch_limit", kinds=("POLICY",))
    task = ContextTask("approve-run", "May this run use batch size 64?", (required,))
    context = ContextCompiler(state).compile(task, scope=scope, at=now(), known_at=now(),
        profile=ModelProfile("exact-demo-v1"), budget=CompileBudget(bytes=8192, milliseconds=10000))
    if context.status != "complete":
        raise RuntimeError("fixture compilation failed")
    question = Question("allowed", task.question, Boolean(), (required,), "batch-policy-v1")
    with tempfile.TemporaryDirectory(prefix="managed-decision-example-") as directory:
        root = Path(directory)
        refs = {}

        def publish(raw):
            fingerprint = hashlib.sha256(raw).hexdigest()
            (root / (fingerprint + ".json")).write_bytes(raw)
            ref = EvidenceRef("lab", fingerprint, "v1", fingerprint, "observation")
            refs[fingerprint] = ref
            return ref

        def resolve(ref):
            # Closed example map: never join a model-supplied path to a directory.
            if refs.get(ref.source) != ref:
                raise PermissionError("unknown artifact")
            return (root / (ref.digest + ".json")).read_bytes()

        def authorize(authority, operation, record):
            return authority == actor and (record is None or all(refs.get(r.source) == r for r in references(record)))

        request = publish(canonical({"context": json.loads(context.text), "requested_batch": 64,
            "compiled_material": compiled_material(context), "binding": context.binding,
            "question_schema": question.schema_id}).encode())
        episode = Episode("managed-1", "batch-policy", "fictional-lab", "approval-v1", "development",
                          "offline-v1", 7, actor, "exact-demo-v1", "limit-check-v1", now())
        action = Action("typed-policy-check", "v1", request, "pure", request.digest)
        plan = TransitionPlan(episode.episode_id, 0, (request,), "compiled-v1", action, None, None, now())

        def current(proposal):
            return (proposal == plan and time.monotonic() < context.deadline and state.is_current(context.snapshot)
                    and state.verify_binding(context.snapshot, compiled_material(context), context.binding))

        key = secrets.token_bytes(32)
        with AuditStore(root / "audit.sqlite", tenant="lab", signing_key=key, authorize=authorize, create=True) as audit:
            audit.append(episode, authority=actor)
            audit.append(plan, authority=actor)
            executor = ManagedExecutor(audit,
                adapters=(ExecutionAdapter(action.tool, action.tool_revision, "pure", "limit-check-v1",
                                           policy_provider, policy_verifier),),
                resolve=resolve, publish=publish, is_current=current)
            deadline = now() + max(1, int((context.deadline - time.monotonic()) * 1000))
            result = executor.run(plan, authority=actor, deadline_ms=deadline)
            if result.outcome.record.status != "succeeded":
                raise RuntimeError("managed fixture failed")
            answer = json.loads(resolve(result.outcome.record.observations[0]))
            # The final typed API still checks its compiled binding, current state
            # and requirements. Its provider is now an immediate in-memory reply.
            decisions = decide_batch((question,), context=context, state=state,
                provider=lambda questions, text: (Proposal(answer["question_id"], answer["value"]),),
                verify=lambda q, value, packet: policy_verifier(resolve(request), canonical(answer).encode()),
                verifier_revision="limit-check-v1")
            if decisions[0].abstained or decisions[0].result is not True:
                raise RuntimeError("typed decision was not accepted")
            try:
                executor.run(plan, authority=actor, deadline_ms=deadline)
            except DispatchBlockedError:
                duplicate_blocked = True
            else:
                raise RuntimeError("duplicate proposal was dispatched")
            audit.append(EpisodeEnd(episode.episode_id, "succeeded", now()), authority=actor)
            checkpoint = audit.checkpoint(authority=actor)
            wall_ms = result.outcome.record.cost.wall_ms
        with AuditStore(root / "audit.sqlite", tenant="lab", signing_key=key,
                        authorize=authorize, checkpoint=checkpoint) as reopened:
            reopened.verify(authority=actor)
            events = reopened.history(episode.episode_id, authority=actor)
            checked_refs = {ref for entry in events for ref in references(entry.record)}
            if any(hashlib.sha256(resolve(ref)).hexdigest() != ref.digest for ref in checked_refs):
                raise RuntimeError("artifact changed")
        print(json.dumps(dict(verified_decision=decisions[0].result, committed_events=len(events),
            duplicate_blocked=duplicate_blocked, reopened_and_verified=True,
            external_references_verified=len(checked_refs), model_calls=0,
            managed_wall_ms=wall_ms, scope="offline deterministic fixture; includes Python worker startup"), indent=2))


if __name__ == "__main__":
    main()

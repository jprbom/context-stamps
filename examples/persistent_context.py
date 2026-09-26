"""Durable context -> bounded working set -> compiler -> managed decision -> audit.

Copyright (c) 2026 Prashant Jagtap. MIT License.
Fictional experiment metadata and exact Python functions; no trained model.
"""

import hashlib
import json
import secrets
import tempfile
import time
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
from context_stamps.state_store import ContextStore
from context_stamps.working_set import ContextWorkingSet


def provider(payload, key, deadline_ms):
    evidence = json.loads(payload)["packet"]["evidence"]
    values = {c["name"]: int(c["value"]) for row in evidence for c in row["claims"]}
    return WorkerResult(canonical({"allowed": values["batch"] <= values["limit"]}).encode(), ResourceUse(0, 0, 0, 1, 0.))


def verify(payload, output):
    evidence, result = json.loads(payload)["packet"]["evidence"], json.loads(output)
    limits = {int(c["value"]) for r in evidence if r["kind"] == "POLICY" for c in r["claims"] if c["name"] == "limit"}
    batches = {int(c["value"]) for r in evidence if r["kind"] == "OBSERVATION" for c in r["claims"] if c["name"] == "batch"}
    return (set(result) == {"allowed"} and type(result["allowed"]) is bool and len(limits) == len(batches) == 1
            and result["allowed"] is (batches.pop() <= limits.pop()))


def main():
    now = lambda: time.time_ns() // 1000000  # noqa: E731
    scope = AccessScope("lab", "researcher", "p1", ("reader",))
    actor = Authority("lab", "researcher", "p1")
    state_key, audit_key = secrets.token_bytes(32), secrets.token_bytes(32)
    authorize_state = lambda s, op, subject: s == scope  # noqa: E731
    # This local example supplies a fictional authenticated scope. A deployed
    # host must validate identities and restrict write/authority-assignment rights.
    with tempfile.TemporaryDirectory(prefix="persistent-context-example-") as directory:
        root = Path(directory)
        policy = CanonicalNode("policy", "v1", "lab", "Reviewed batch limit is 128.", "POLICY",
            TemporalScope(0, now()), ("reader",), "reviewed-config", claims=(ContextClaim("limit", "128"),),
            validation_revision="review-v1")
        run = CanonicalNode("run", "v1", "lab", "Observed batch size is 64.", "OBSERVATION",
            TemporalScope(0, now()), ("reader",), "run-manifest", claims=(ContextClaim("batch", "64"),),
            dependencies=(policy.ref,), validation_revision="run-check-v1")
        with ContextStore(root / "state.sqlite", tenant="lab", signing_key=state_key, authorize=authorize_state,
                          create=True, policy_revision="p1") as store:
            store.put(policy, scope=scope, mutation_id="ingest-policy")
            store.put(run, scope=scope, mutation_id="ingest-run")
            before = store.snapshot(scope, at=now(), known_at=now())
            checkpoint = store.checkpoint(scope=scope, verify=True)

        with ContextStore(root / "state.sqlite", tenant="lab", signing_key=state_key, authorize=authorize_state,
                          checkpoint=checkpoint) as store, ContextWorkingSet(store, scope=scope, max_nodes=2) as memory:
            if not store.is_current(before):
                raise RuntimeError("snapshot did not survive restart")
            memory.pin(policy.ref, at=now(), known_at=now())
            ticket = memory.prefetch((run.ref,), at=now(), known_at=now())
            wait_until = time.perf_counter() + 5
            while time.perf_counter() < wait_until:
                prefetched = memory.prefetch_result(ticket)
                if prefetched.status != "pending":
                    break
                time.sleep(.005)
            if prefetched.status != "ready" or memory.stats()["active_nodes"] != 1:
                raise RuntimeError("prefetch must remain outside the active set")
            memory.activate((run.ref,), at=now(), known_at=now())
            required = (EvidenceRequirement("limit", kinds=("POLICY",)), EvidenceRequirement("batch", kinds=("OBSERVATION",)))
            task = ContextTask("approve", "Does the recorded batch respect the reviewed limit?", required)
            packet = ContextCompiler(memory).compile(task, scope=scope, at=now(), known_at=now(),
                profile=ModelProfile("exact-v1"), budget=CompileBudget(bytes=8192, milliseconds=10000))
            if packet.status != "complete":
                raise RuntimeError("fixture did not compile")
            refs = {}

            def publish(raw):
                fingerprint = hashlib.sha256(raw).hexdigest()
                (root / (fingerprint + ".json")).write_bytes(raw)
                ref = EvidenceRef("lab", fingerprint, "v1", fingerprint, "observation")
                refs[fingerprint] = ref
                return ref

            def resolve(ref):
                if refs.get(ref.source) != ref:
                    raise PermissionError("unknown artifact")
                return (root / (ref.digest + ".json")).read_bytes()

            def authorize_audit(a, op, record):
                if a != actor or record is not None and any(refs.get(r.source) != r for r in references(record)):
                    return False
                # Artifact access follows current access to its underlying sources.
                store.inventory(scope, at=now(), known_at=now(), roots=(run.ref,))
                return True

            request = publish(canonical({"packet": json.loads(packet.text), "binding": packet.binding,
                                         "compiled_material": compiled_material(packet)}).encode())
            episode = Episode("run-approval", "batch-policy", "fictional-lab", "approval-v1", "development",
                              "offline-v1", 7, actor, "exact-v1", "verify-v1", now())
            action = Action("batch-policy", "v1", request, "pure", request.digest)
            plan = TransitionPlan(episode.episode_id, 0, (request,), "compiled-v1", action, None, None, now())
            with AuditStore(root / "audit.sqlite", tenant="lab", signing_key=audit_key,
                            authorize=authorize_audit, create=True) as audit:
                audit.append(episode, authority=actor)
                audit.append(plan, authority=actor)
                executor = ManagedExecutor(audit, adapters=(ExecutionAdapter("batch-policy", "v1", "pure", "verify-v1", provider, verify),),
                    resolve=resolve, publish=publish,
                    is_current=lambda p: p == plan and memory.verify_binding(packet.snapshot, compiled_material(packet), packet.binding))
                deadline = now() + max(1, int((packet.deadline - time.monotonic()) * 1000))
                outcome = executor.run(plan, authority=actor, deadline_ms=deadline).outcome.record
                if outcome.status != "succeeded":
                    raise RuntimeError("managed decision failed")
                output = resolve(outcome.observations[0])
                question = Question("allowed", task.question, Boolean(), required, "batch-policy-v1")
                decision = decide_batch((question,), context=packet, state=memory,
                    provider=lambda qs, text: (Proposal("allowed", json.loads(output)["allowed"]),),
                    verify=lambda q, value, context: value is json.loads(output)["allowed"] and verify(resolve(request), output),
                    verifier_revision="verify-v1")[0]
                if decision.abstained or decision.result is not True:
                    raise RuntimeError("typed result not verified")
                audit.append(EpisodeEnd(episode.episode_id, "succeeded", now()), authority=actor)
                audit_checkpoint = audit.verify(authority=actor)
            stats = memory.stats()
            memory.activate((), at=now(), known_at=now())  # Pinned policy stays active.
            evicted = memory.collect(idle_seconds=0)
            retained = len(store.snapshot(scope, at=now(), known_at=now()).records)
            store.set_roles("policy", (), scope=scope, mutation_id="revoke-policy")
            revoked = not memory.is_current(packet.snapshot)
        print(json.dumps(dict(state_reopened=True, verified_decision=True, audit_events=audit_checkpoint.sequence,
            prefetched_before_demand=prefetched.status == "ready", working_stats=stats,
            collected_working_copies=evicted, retained_source_versions=retained, revocation_invalidates_packet=revoked,
            model_calls=0, scope="fictional deterministic integration; no model-quality claim"), indent=2))


if __name__ == "__main__":
    main()

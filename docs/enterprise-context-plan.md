# Enterprise Context Intelligence Runtime

By Prashant Jagtap

This is the active implementation objective, superseding the earlier general-learning programme. The runtime owns versioned enterprise context; interchangeable models consume compiled views. A 256-bit stamp is a routing/control reference. Detailed evidence, permissions, relationships and computation inputs remain externally available and independently verifiable.

**Target:** given a task, select fresh, authorized, sufficient context within a declared economic budget, execute an appropriate decision/tool/model, and provide a receipt explaining the evidence and computation used.

“Smallest,” “causally sufficient” and “economically optimal” are research objectives, not current guarantees. Exact minimality can be proved only for a bounded candidate set with a declared cost function and requirements. Declared dependency closure does not establish causal discovery. Learned sufficiency probabilities require calibration and an explicit scope.

## Seven planes and current evidence

| Plane | Starting implementation | Work remaining |
|---|---|---|
| Ingestion | File observations, source identities and explicit facet compilation | Canonical modality adapters, typed provenance and verified extraction |
| Context state | `ContextNode`, graph, versions, roles and relation bindings | Bitemporal history, epistemic types, negative knowledge, lifecycle and persistent authority |
| Context intelligence | Experimental rankers, exact requirements, failed cheap-route learner | Calibrated sufficiency/value/conflict/uncertainty and context-node attention |
| Context compiler | Structured minimum-byte selection and runtime dependency closure | Unified authorization-first compiler, model profiles, faithful compression, actual tokenizer budgets and adaptive acquisition |
| Decision/routing | Expert registration and verifier-controlled abstention | First-class Choice, Boolean and Score decisions, batch API and calibration certificates |
| Model execution | Explicit host-owned callbacks and local reader pilots | Shared reader/VLM/tool contract, hard deadlines, side-effect execution semantics and measured portability |
| Learning/audit | Experiment manifests, exact reuse and bounded in-memory outcomes | Durable reference-only events, prospective labels, policy candidate evaluation and immutable receipts |

The package remains at v0.5.0. This plan does not publish a 1.0 release or change existing evidence. Use the current controller-v2/runtime-v1 reports for numbers; the revised objective contains an older retrieval table that must not silently replace those results.

## Implementation sequence

### 1. Canonical state and typed decisions

Extend state through an optional interface so existing users retain the small standard-library core. Every canonical node binds tenant/source/revision/content hash, modality, provenance, explicit epistemic type, validity interval, observation/transaction time, policy scope and lifecycle. Keep contradictory versions instead of overwriting them.

Temporal queries take both an effective time and a knowledge cutoff. A later correction must not appear in a replay of what the system knew earlier. `effective_at` and `event_time` are distinct from ingestion/transaction time. Supersession links bind exact versions and are acyclic. Historical evidence remains subject to current authorization.

Knowledge kinds: FACT, OBSERVATION, CLAIM, INFERENCE, POLICY, ASSUMPTION, PREDICTION, USER_ASSERTION, MODEL_OUTPUT and DERIVED_RESULT. A model output cannot satisfy a verified-fact or policy requirement without a separate, versioned validation result. Typed status alone is not proof of truth.

Absence states: UNKNOWN, KNOWN_ABSENT, NOT_APPLICABLE, REDACTED, UNAUTHORIZED and STALE. Negative assertions require scope, provenance, validity and expiry. An absent record is not evidence of known absence. Public responses must not reveal the existence of unauthorized records; detailed denial reasons belong in a restricted audit view.

Add `decisions/schema.py`, `choice.py`, `boolean.py`, `score.py`, `calibration.py`, `abstention.py` and `batch.py`. Bind every result to a question, typed result, evidence/stamp IDs, state revision, model/verifier version and abstention reason. Probability and calibration error are nullable unless a validated calibration scope exists. A numeric score is not automatically a probability. Revalidate authority and state before publishing batched decisions.

Proof: typed round trips, invalid-input rejection, old-policy queries, delayed corrections, conflicting sources, expired negative assertions, no hidden-record disclosure, and mutation/revocation races. These are engineering checks, not general decision-accuracy evidence.

### 2. Context compiler, value and adaptive acquisition

Expose a model-aware compile request containing task, trusted principal, time query, graph snapshot, exact requirements, tokenizer/formatter revisions, confidence policy and resource limits. Implement the ten compiler stages explicitly:

1. Filter authorization before ranking or forwarding evidence to any model.
2. Reject stale or temporally inapplicable nodes.
3. Detect conflicts and record the resolution rule or abstention.
4. Include required dependency closure.
5. Rank eligible additions by measured/estimated task value and marginal cost.
6. Remove redundant content without dropping provenance or required dependencies.
7. Apply optional semantic compression only with traceable source spans and verification; preserve raw-evidence fallback.
8. Format for a pinned model adapter.
9. Count the complete serialized packet with the actual tokenizer, including provenance and instructions.
10. Check exact requirements and/or a calibrated sufficiency evaluator; retrieve more, stop or abstain.

Value combines relevance, information gain, dependency contribution, temporal applicability and source trust against token, latency and risk costs. Authorization and hard policy rules are constraints, never soft scores that high relevance can outweigh. Guard zero denominators and explicitly record coefficient units. Fit costs on development data and compare against simple fixed rules.

Adaptive acquisition ends at a sufficient verified packet, an uneconomic marginal improvement, no progress, or an exhausted token/time/call budget. Predicted sufficiency cannot grant permission, promote an assumption to fact, or declare missing evidence present. Freeze thresholds before prospective evaluation.

Proof: bounded exact-minimum control on small instances; full-context, exact-metadata, graph and retrieval controls at equal information; actual total token/call/time accounting; failure and calibration curves. For general tasks, report measured bounded optimization rather than claiming a globally minimal context.

### 3. Working sets, lifecycle and speculation

Implement cold storage, warm materialized context and the active compiled window with explicit page-in, page-out, pin, fault, prefetch and eviction operations. Lifecycle states are HOT, WARM, COLD, SUPERSEDED, INVALID and ARCHIVED. Track byte/token capacity and per-principal quotas; apply backpressure.

Separate reasoning eviction from compliance retention. Removing a node from the working set must not delete a retained source or orphan an active computation. Retention policy can prevent physical deletion. Prefetched evidence remains outside the active packet until demanded and independently authorized.

Proof: source mutation during prefetch, revocation, pinned policy evidence, invalid dependencies, retained archives, bounded memory under repeated requests, and cancellation. Measure prefetch usefulness and net end-to-end latency after counting wasted work at repeat ratios 0%, 10% and 50%.

### 4. Delta computation and durable audit

Extend selective packet invalidation into a versioned computation DAG. Recompute changed inputs and affected active descendants; retain unrelated exact results. Bind receipts to result hash, full input/dependency identities, model/prompt/generation settings, policy/tool/verifier versions and timestamps. Semantic stamp equality alone cannot validate computation reuse.

The new experience schema draft remains useful for this plane: observations, proposed actions, predictions and measured outcomes have separate types. It is not yet a durable ledger. Complete its validation tests, then add transactional storage, restart replay, negative outcomes, permitted exports and retention/deletion. Predictions must be recorded before action execution; never backfill them from observed results.

Separate pure computation reuse from effectful action execution. Idempotency/reconciliation needs provider-side status evidence; a local cache cannot guarantee exactly-once side effects. Move external callbacks outside shared locks and recheck the immutable snapshot/authority at commit. Use cancellable workers and one propagated deadline.

Proof: a branched dependency graph where only affected computations rerun; same-content/new-version changes; policy/tool/model updates; retry/restart races; failures retained; no stale output release; measured whole-request overhead.

### 5. Learned context intelligence

Use verified workflow trajectories to train context-node attention and separate heads for relevance, value, decision, sufficiency, missing context, uncertainty, conflict, staleness and next-context prediction. Start with frozen representations and strong exact/rule/linear/MLP controls. Compare sparse attention and recurrence only when they improve held-out outcomes at matched compute.

The suggested 50M–500M context model and 0.5B–3B decision model ranges are experiment candidates, not minimum architecture sizes or guaranteed RTX fits. Profile smaller candidates first; parameter count excludes activations, encoder, cache and optimizer memory. Do not modify foundation-model attention before an external context layer earns promotion.

The existing cheap-route learner failed calibration and stays disabled. The existing selected student trails hybrid on four of five collections. Train new candidates from licensed training-only data, retain failed trials, and keep the current qualified routes as regression controls.

Proof: untouched project/time/template partitions, calibration/coverage, zero-repeat and changing-state tasks, label provenance, matched token/latency budgets, quantization parity, measured device speed, and rollback. A lower prediction loss alone does not qualify a policy.

### 6. Multimodal adapters and local readers

Text, image, audio, video, table and code adapters emit the same canonical node contract. Blobs stay external with hashes, timestamps and encoder/OCR/extraction lineage. Video uses a temporal event/scene graph. Keep model-specific vector/stamp family compatibility explicit; a 32-byte wire format does not imply interchangeable semantic codes from unrelated encoders.

Evaluate three reader classes: small decision/context models, general quantized language models, and vision-language models. Select pinned licensed checkpoints after a VRAM/RAM feasibility check; run one resident GPU workload at a time. Keep data preparation, ledger work, evidence replay and routine tests on CPU. Load RTX work only for explicit GPU verification, training or benchmarking.

Proof: adapter equivalence, malformed and missing media, changing timestamps, permission changes, separate model/tokenizer/prompt records, and end-task metrics. Do not infer image/audio/video quality gains from successful adapter calls.

### 7. Prospective benchmark and release evidence

Register licenses, revisions, task IDs, partitions, contamination risks and access policy before running evaluation. The current five retrieval collections are development/regression material. Add untouched licensed retrieval scopes before algorithm selection.

Required benchmark families: BEIR retrieval; RULER; LongBench v2; LongMemEval; LoCoMo; BEAM. Preserve each benchmark's native scoring and report any local subset or modified protocol separately. The large-context suites may exceed local full-context capacity; classify unsupported lengths rather than truncate silently or compare unequal baselines. Use a typed-decision external provider only as an optional matched control after its access, cost and privacy conditions are reviewed.

For every experiment record per-task success, false acceptance, abstention, evidence correctness, tokens, calls, cold/warm whole-request latency, p50/p95/p99, peak RAM/VRAM and observed concurrency limits. Report energy only with reliable telemetry. Include confidence intervals clustered by original task, all failures, and controls for exact metadata, graph, dense/hybrid retrieval, full context, no reuse and fixed experts.

Release proof requires reproducible CPU/RTX profiles, source/data/model hashes, security/isolation tests, independent reruns, a claim-to-evidence map, source/model/data notices and owner attribution. Public documentation must distinguish implemented APIs, measured local results and unproven research targets. No universal-superiority or AGI claim follows from this programme.

## Immediate work

The research worktree is isolated from the public main branch. CPU baseline reproduction is complete after correcting a path error in the new runner; the failed attempt remains recorded. The reference-only experience contracts now have ten tests. All 155 tests execute without skips in both the CPU-capable environment and a newly isolated CUDA environment; the latter also passes a CUDA calculation check. A short training-capacity probe uses only the existing public training partitions and selects batch 128 as an initial throughput setting for both existing controller variants. It does not qualify a new model or demonstrate sustained serving capacity. Next: implement canonical temporal/epistemic state and typed decisions, then the integrated compiler and durable audit path before training a new context model.

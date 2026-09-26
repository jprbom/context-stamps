# Temporal evidence, context compilation and typed decisions

By Prashant Jagtap · Research branch implementation

The runtime can now select an authorized temporal view, compile explicitly required evidence, and return verified `Boolean`, `Choice` or `Score` decisions in one provider batch. It is a standard-library implementation. It does not require a GPU or a language model.

This is useful when an application already knows its required evidence: experiment approval, configuration checks, version-aware coding tools or policy-based workflow routing. The host supplies authenticated roles, validated claims and an outcome verifier. The runtime does not infer that a statement is true because it is labelled `FACT`.

## Run the complete example

```bash
git clone --branch research/enterprise-context https://github.com/jprbom/context-stamps.git
cd context-stamps
python -m pip install -e .
python examples/enterprise_decision.py
```

The [complete example](../examples/enterprise_decision.py) checks whether a fictional experiment may use batch size 64 under a reviewed limit of 128. Its deterministic provider returns `True`; the host checks the result against the compiled evidence. The receipt names the exact evidence revision and verifier. Revoking access makes the previous packet unusable. `probability` remains `null` because no calibration experiment was performed.

```python
from context_stamps.context_compiler import (
    CompileBudget, ContextCompiler, ContextTask, ModelProfile,
)
from context_stamps.context_state import EvidenceRequirement
from context_stamps.decisions import Boolean, Choice, Score, Question, decide_batch

# `state` and `scope` come from the trusted host; see the complete example.
required = EvidenceRequirement("batch_limit", kinds=("POLICY",))
task = ContextTask("check-batch", "Can this run use batch size 64?", (required,))
packet = ContextCompiler(state).compile(
    task, scope=scope, at=1000, known_at=1000,
    profile=ModelProfile("my-provider-v1"),
    budget=CompileBudget(bytes=4096, milliseconds=5000),
)
question = Question("allowed", task.question, Boolean(), (required,), "batch-policy-v1")
decisions = decide_batch(
    (question,), context=packet, state=state,
    provider=my_batch_provider, verify=my_outcome_verifier,
    verifier_revision="experiment-check-v1",
)
```

`Choice(("approve", "review", "reject"))` accepts only those strings. `Score(0, 100)` accepts a finite number in that interval, never a Boolean. Providers return a tuple of `Proposal` records with exactly the requested question IDs. The maximum batch is 32 questions. The provider can be a local model, remote model or deterministic function; none is bundled or implicitly contacted.

## Temporal and knowledge semantics

All times use integer UTC epoch milliseconds. A query supplies `at` (when the evidence applies) and `known_at` (what the runtime had observed and ingested by then). Validity is half-open: `max(valid_from, effective_at) <= at < valid_until`. Omitted `valid_until` means no declared expiry. Transaction time is assigned at ingestion; observation time comes from the source. Delayed corrections therefore cannot rewrite a replay of what was known earlier.

Exact `supersedes` references retire predecessors from the replacement's effective start. Expiry or invalidation of a replacement does not reactivate the old evidence. Existing versions cannot be overwritten; a changed payload requires a new revision. Current authorization applies even to historical queries. The first version initializes a source's ACL; later ingestion does not grant new access. Use the explicit host method `set_roles` to change it.

The ten epistemic kinds include `FACT`, `POLICY`, `OBSERVATION`, `ASSUMPTION` and `MODEL_OUTPUT`. Verified requirements accept only explicitly permitted kinds with a validation revision. Model outputs and assumptions cannot satisfy a verified-fact requirement. Validation revisions are host assertions, not cryptographic proof of semantic truth.

`KNOWN_ABSENT` and `NOT_APPLICABLE` require validation and a finite validity interval. They satisfy only a requirement explicitly permitting that status. `UNKNOWN`, `REDACTED`, `UNAUTHORIZED` and `STALE` never silently become `False` or evidence of absence. A denied dependency also hides every dependent node from the view, including its hidden identifiers.

## Compiler behavior and cost

The compiler filters authorization, temporal validity and dependency availability before selection. Different accepted values for the same required claim cause abstention by default. A versioned `ConflictPolicy` may prioritize authority, source reliability, effective time, observation time or kind. Equal-priority contradictions remain unresolved. Both source versions remain in state, and each resolution is inspectable.

For at most 16 candidate roots, the compiler enumerates dependency-closed selections. `optimal=True` means the complete bounded search found the cheapest eligible selection under the declared serialized cost. It does not mean universal semantic or economic optimality. Larger candidate pools use deterministic coverage per marginal serialized cost and report `optimal=False`. Search, byte and elapsed-time limits are explicit.

By default the cost is UTF-8 bytes, including provenance and dependency metadata. To enforce a token budget, supply a real tokenizer callback and pinned tokenizer identity in `ModelProfile`, then set `CompileBudget(tokens=...)`. The counter must match the formatter and actual model. A character count is not a tokenizer. The budget covers the compiled text; the host must account separately for batch questions, system instructions, chat templates, tool schemas and output reservations unless its formatter includes them.

Custom formatters require an explicit evidence-preservation verifier. The default JSON keeps source text labelled as untrusted data. HMAC bindings detect changes to the packet, task, model/formatter/tokenizer identifiers or deadline before provider execution. The host remains responsible for prompt handling; labels and signatures cannot make arbitrary source text safe instructions.

## Calibration and abstention

Raw provider confidence is never relabelled as probability. `fit_calibration` uses independently labelled, held-out clusters to fit fixed reliability bins. Each bin records empirical correctness and a one-sided exact binomial lower correctness bound with a correction across bins. A `DecisionPolicy(minimum_probability=...)` gates on that conservative bound. An empirical probability of 1.0 is not certainty.

Certificates bind model, question text/type/requirements, declared task scope, verifier and authorization-policy revision. Changed bindings, undersampled bins and missing confidence prevent a probability-based acceptance. The host must establish dataset independence, cluster identity, calibration/test separation and validity under distribution shift. These statistical checks have unit tests; no new learned decision model or real-world reliability certificate is claimed.

## Local engineering evidence

The [recorded full suite](../evidence/enterprise-state-v1/local-cpu/manifest.json) contains **182 passing tests, zero skips**, including 27 new temporal/compiler/decision tests. The new tests cover independent small-instance cost enumeration, conflicting policies, delayed corrections, finite negative knowledge, transitive revocation, malformed adapters, packet alteration, calibration scope and concurrent revocation during a provider call.

The same run compares full authorized serialization with compilation on fictional exact-claim fixtures, with one relevant record and 3–255 irrelevant records. There are 20 measured sequential trials per size, following one warmup per method. Both paths use the same serializer and temporal view.

| Records | Full serialization median, ms | Compiler median, ms | Full p95, ms | Compiler p95, ms | Serialized byte reduction |
|---:|---:|---:|---:|---:|---:|
| 4 | 0.044 | 0.104 | 0.052 | 0.173 | 72.2% |
| 16 | 0.130 | 0.175 | 0.245 | 0.236 | 92.8% |
| 64 | 0.458 | 0.405 | 0.608 | 0.520 | 98.2% |
| 256 | 1.901 | 1.443 | 2.589 | 2.382 | 99.5% |

Compilation adds overhead at the two smallest sizes. These timings exclude ingestion, tokenization, model inference, networking and concurrent load. The byte reductions follow from these deliberately simple fixtures; they are not token savings or measured enterprise productivity. This experiment does not compare semantic retrieval quality or train a model.

```powershell
# CPU preparation/tests; the CUDA environment includes Torch for optional tests.
$env:CUDA_VISIBLE_DEVICES=''
$env:OMP_NUM_THREADS='2'
$env:MKL_NUM_THREADS='2'
python experiments/enterprise_state_validation.py --out evidence/local-independent-run
python experiments/verify_enterprise_state.py
```

Use a new output directory for each run. `verify_enterprise_state.py` checks the published measurements and hashes; it does not substitute for an independent rerun. Training uses the RTX separately, following the [training-capacity measurements](enterprise-foundations.md). This compiler/state phase does not benefit from GPU execution.

## Remaining boundaries

The state and integrity key are in-process. There is no durable ledger, remote authentication server, distributed transaction protocol or persistent lifecycle manager yet. An `AccessScope` must come from the trusted application, never unchecked client input. The host must allow-list model destinations and prevent unauthorized external transmission.

Permission and state changes during a callback invalidate its result. Callbacks execute outside the state lock, but cannot yet be forcibly cancelled. A deadline rejects late output; it does not stop ongoing work or undo a side effect. No external side effects should be authorized from a decision without a fresh host check.

Claim coverage is deterministic and explicit. Learned sufficiency, adaptive acquisition, semantic conflict detection/compression, sparse context-node attention, durable computation DAGs and canonical media extraction remain in the [implementation programme](enterprise-context-plan.md). The media type/digest fields do not establish image, voice or video model performance.

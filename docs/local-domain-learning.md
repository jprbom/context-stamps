# Local learning for small domain models

By Prashant Jagtap

The primary aim is a small, locally deployed domain system that completes demanding work reliably within its device's memory, latency, energy and cost limits. Context Stamps should help that system learn from verified local experience. The seven-plane enterprise runtime is the mechanism for this aim. A larger language model, more retrieved text or a lower token count alone is not the success criterion.

“Domain AGI” expresses an ambition. The testable target is competence across an explicitly bounded domain: unfamiliar combinations of tasks, changing evidence, planning, tool use, correction, uncertainty and recovery. We have not established that capability or general autonomous self-improvement.

## Three levels of adaptation

| Level | What can change locally | Required evidence | Current position |
|---|---|---|---|
| Verified memory | Facts, temporal versions, dependencies, working sets and reusable computation | Provenance, current permissions, independent validation and invalidation | Existing state, compiler and reuse interfaces; application integration remains |
| Context policy | Which evidence to acquire, which approved expert to use, when to stop or abstain | Separate training and prospective paired evaluation; quality and total resource checks | New small statistical policy, persistent evaluation registry and rollback API; simulated demonstration only |
| Model parameters | Small task heads, low-rank adapters or a domain SLM | Licensed training data, held-out task families, retention tests, quantized inference parity and device measurements | Prior experimental context heads exist; no automatically improving SLM weights are qualified |

Learning stays in the local environment. The new modules have no network client, telemetry, model loader, shell execution or remote training dependency. A host application still controls its own providers, permissions, storage and adapters. This library does not make an application offline merely because its learning registry is local.

## What is implemented

`context_stamps.local_policy.fit_cell_policy` fits a small conditional policy from matched, externally checked local outcomes. For each predefined context cell and approved action, it keeps counts, errors and costs. A Beta(1,1) estimate smooths the failure rate, and a dimensioned cost penalty discourages expensive actions. Under-supported and unseen cells retain the baseline. The exported policy contains only bounded cell/action choices and an environment binding; it can run without a tensor framework or GPU.

These estimates propose candidates. They are not calibrated confidence in a future answer. Context cells must be defined from input information available before answering, never from an answer key. Every training task must include observations for every compared action. Partial logs cannot stand in for unobserved counterfactual outcomes. Logged observational data need a separate causal/off-policy method; it is not supported by this fitter.

`context_stamps.local_learning.LocalLearningRegistry` persists a local sequence of frozen candidates and evaluation plans in SQLite. It:

1. Reserves fresh task-cluster IDs before evaluation. Training, earlier evaluation and abandoned-round IDs cannot be reused as new evaluation cases.
2. Requires every registered task, including failures and abstentions. Partial results cannot activate a candidate.
3. Keeps an experiment error budget across restarts and abandoned rounds. At round `r`, the available budget is `alpha / (r * (r + 1))`; four statistical checks share it equally. The sum over rounds is at most `alpha` within this one registry and scope.
4. Bounds new failures on tasks the baseline solved, overall candidate failure, and deadline exceedances using one-sided exact binomial bounds. It requires a positive lower bound on paired resource savings after the declared learning cost is amortized.
5. Rejects missing measurements, sampled memory overruns, safety violations and observations outside the preregistered cost range. Outliers are retained as a failed gate, not clipped away.
6. Activates only a data artifact's digest. It neither installs nor executes a proposed model or program. A trusted host maps that digest to an already reviewed policy and independently authorizes actions and evidence.
7. Supports explicit rollback, cancels pending activation and prevents the revoked candidate from being reactivated in that registry.

The registry binds the scope, base policy, verifier, runtime/model versions and device profile through a host-supplied digest. A different binding returns no applicable policy. Permission checks still run at the time of use. Returning to a baseline does not authorize an action under an obsolete policy.

## Statistical interpretation

Let `D` be baseline cost minus candidate cost on the same task. Both costs must fall within the registered range `[0, C]`. For `n` independent paired tasks and per-check error budget `a`, the implemented lower bound is:

```text
mean(D) - C * sqrt(2 * log(1/a) / n) - update_cost / amortization_tasks
```

This is a conservative Hoeffding bound for differences in `[-C, C]`. It is not an empirical-Bernstein confidence sequence. The whole sample size is fixed before the round; inspecting partial results cannot earn early activation. A small perfect sample will usually remain inconclusive. A zero observed error is not a zero error bound.

The new-failure check bounds the probability of `baseline correct AND candidate incorrect`. It conservatively bounds the increase in error without allowing successes elsewhere to hide new harms. The absolute-failure check also prevents an equally bad baseline and candidate from qualifying merely through parity. The latency check is an absolute deadline exceedance limit, not proof of a relative p95 speedup. Memory checks cover observed runs, not all future inputs.

The statistical interpretation assumes IID task clusters for the binomial checks and independent bounded paired costs for the savings check; candidates, thresholds and task selection must be frozen before labels are exposed. Repeated prompts from one source are one cluster, not independent samples. Mixed domains require separately registered scopes and an additional programme-wide error budget. Temporal dependence, poisoning, distribution shift and unreported experiments can invalidate the interpretation. Creating a fresh database to reset the error budget is not valid methodology.

SQLite transactions enforce ordinary restart and competing-writer behavior. They cannot attest that a host supplied honest labels or complete records. The file is not encrypted, authenticated or resistant to an administrator modifying or restoring it. The host must protect the file, retain its experiment lineage, and connect promotion events to the existing durable audit system before a managed enterprise deployment. Automatic drift detection, signed policy export, multi-device coordination and model-weight training are not implemented here.

The design draws on [safe policy improvement with a baseline](https://proceedings.mlr.press/v97/laroche19a.html). It uses a simpler fixed-sample gate, not that paper's MDP algorithm. [Time-uniform confidence sequences](https://arxiv.org/abs/1810.08240) motivate a future more sample-efficient sequential gate. [Conformal risk control](https://arxiv.org/abs/2208.02814) provides another research direction, but its assumptions must be checked before adapting it to changing local streams. These are established statistical tools; their presence does not establish project novelty.

## Run the complete local example

From a reviewed clone with the core installed:

```bash
python -m pip install -e .
python examples/local_learning.py
python -m unittest discover -s tests -p test_local_learning.py -v
python experiments/verify_local_learning.py
```

The [complete example](../examples/local_learning.py) fits a two-cell policy on 60 simulated tasks, evaluates a learned candidate and a deliberately harmful shortcut on 600 new fixture IDs each, then rolls back. All wall-time and memory values in this fixture are explicitly simulated; the resource unit is `fixture_units`. The fitted policy chooses compact context for the direct cell and full context for the dependency cell. The harmful shortcut is rejected. The [retained records](../evidence/local-learning-v1/manifest.json) contain every fixture observation, frozen plan, result and source fingerprint. This is an engineering demonstration, not a public benchmark or evidence of SLM improvement.

To connect a real local model, keep the same sequence and replace fixture outcomes with complete measured pipelines. Use the same local model revision and settings for baseline and candidate; randomize execution order; record cold and warm runs separately. Count context preparation, selection, inference, tools and verification. Report training time/energy separately and charge it to the relevant deployment objective. Unknown energy or memory remains unknown and cannot pass a gate that requires it. A policy artifact must not contain training text or private evidence.

## Research beyond a conventional language-model loop

| Method to investigate | Concrete role in the domain system | Decisive comparison |
|---|---|---|
| Bayesian state estimation | Track uncertainty over validated domain state and missing observations; update sufficient statistics locally | Fixed rules and ordinary temporal retrieval at equal information |
| Value of information | Acquire an extra observation only when expected reduction in verified decision loss justifies its cost | Fixed-k and full-context policies including acquisition overhead |
| Task-aware rate–distortion | Allocate representation capacity to facts that change domain decisions; preserve exact references externally | Fixed eight-facet allocation plus precise resolver; measure harmful collisions and missed dependencies |
| Structured domain models | Combine exact calculations, constraints and state transitions with a small reader | Same reader alone and same tools without learned context control |
| Sparse local experts | Select an approved numeric tool, symbolic rule, small head or SLM by task state | Best fixed expert, rule routing and equally informed learned routing |
| Change-point monitoring | Quarantine a learned policy when verified error/resource behavior shifts | Static policy, scheduled retraining and deliberate abrupt/gradual shifts |
| Selective local adapters | Update small task heads or low-rank weights only for persistent verified failure modes | Context-only adaptation first; retention and quantization tests after every update |

These are proposed experiments, not implemented capabilities or claims of first invention. The 256-bit spherical stamp continues to route to external evidence and relationship maps. More compact representations cannot preserve unlimited information. A local policy must never infer authorization or truth from a hash collision.

## Qualification order and RTX workflow

The first [local coding controls](terminal-local-pilot.md) show why this order matters: the 1.5B model passes 0/2 selected tasks, and the 7B coding reference passes 1/2 while exhausting both step budgets. One agent writes a correct artifact but fails to repair its own check; the other output uses placeholder counts without consulting the source. These observations identify action, evidence-use and recovery gaps. They are not training data or proof of local adaptation.

Start with software maintenance and operational diagnostics, where bounded tool outputs and executable verifiers are practical. This is the initial evaluation domain, not a claim that present models already handle its full scope. Add other domains only after their evidence, verifiers and risk requirements are defined.

1. **Freeze a domain contract.** Declare tasks, exclusions, harmful errors, abstention behavior, allowed tools and device limits. Keep task-family/project holdouts separate from fitting, tuning and calibration. Public benchmark tasks marked evaluation-only must never enter training.
2. **Establish controls.** Measure a 0.5–3B local model with full context, ordinary retrieval and the runtime. Add a stronger 7–14B local reference where memory permits. All use comparable evidence and tools; matching a larger model on a selected task set is a domain-specific result.
3. **Fit the cheap policy on CPU.** Start with verified paired outcomes and the small statistical model above. Compare against an ordinary rule table. Reject unsupported cells rather than extrapolating confidence.
4. **Train on RTX only when useful.** Fit context heads or small adapters on the independent training partition; freeze weights before calibration. Use the [existing RTX training runbook](controller-methodology-v2.md) for context-ranker experiments. It is not a completed domain-SLM fine-tuning pipeline. Keep data preparation, registries and scoring on CPU; record actual GPU work and do not artificially load the GPU.
5. **Run a prospective local cycle.** Register the candidate and fresh paired evaluation once. Record verified task success, new failures, abstention, complete input/output tokens, p50/p95 latency, RAM/VRAM, measured energy, learning cost and break-even deployment volume. Failed or inconclusive candidates retain the active policy.
6. **Test adaptation and retention.** Evaluate new workflow variants, dependencies outside selected context, contradictory/stale evidence, permission revocation, missing tools, poisoned feedback, repeated tasks, zero-repeat streams, crashes and abrupt/gradual distribution shift. Include older domain tasks after each update.
7. **Qualify the target device.** Repeat the frozen comparison on the intended edge CPU/GPU/NPU with realistic thermal and battery limits. An RTX laptop result is not embedded-device evidence. Offline operation, bounded storage and recovery are acceptance criteria.

Automatic improvement means this cycle can run locally when the host has authorized its data, limits and candidate types. It does not mean every update must be accepted, every model can learn from its own answers, or quality must increase on every possible task.

# Monitoring locally adapted policies

By Prashant Jagtap

Local improvement needs both a promotion gate and a way to stop using a candidate that deteriorates. `LocalRegressionMonitor` adds the second part. It monitors externally verified task failures and missed deadlines on CPU, persists alarms across restarts, and can request rollback through `LocalLearningRegistry`. It never trains, loads, activates or executes a model. The existing local training experiments still have no qualified model-weight update.

Use it with a reviewed local context policy, a small task head or an adapter whose complete pipeline has already passed separate adaptation and retention checks. The monitor operates on outcomes, so it does not depend on a particular reader architecture. A model's confidence or self-evaluation is not an independent correctness label.

## Local lifecycle

1. Fit a candidate from authorized, independently checked training experience.
2. Freeze it and run the protected paired evaluation in [local domain learning](local-domain-learning.md). Keep the current revision if the candidate fails or remains inconclusive.
3. Start a monitoring epoch for the qualified revision. Reserve each task cluster and its input-defined cohort **before execution**.
4. Record complete verified results. Unknown correctness/resources, a reported safety violation or a measured RAM overrun produces an immediate alarm. A model abstention is a failed task if the domain contract does not count abstention as successful completion.
5. After a statistical or operational alarm, stop dispatch and call `enforce`. The registry returns to its anchor and revokes that candidate. Collect subsequent authorized evidence for another candidate; do not train on a protected evaluation partition.

The host must implement dispatch, independent verification, deadline termination, privacy controls and scheduling. No background service is installed. The library makes no network calls. A surrounding application still needs its own offline/network policy.

## Complete offline demonstration

```bash
python -m pip install -e .
python examples/local_adaptation_cycle.py
python -m unittest discover -s tests -p test_local_monitor.py -v
python experiments/verify_local_monitor.py
```

The [example](../examples/local_adaptation_cycle.py) fits a two-cell CPU policy, evaluates separate adaptation and retention fixtures, observes a simulated deterioration, restarts the monitor and applies rollback. All labels and resource values are simulated. The [retained evidence](../evidence/local-monitor-v1/README.md) records that distinction. This is a control-flow and statistical implementation check, not a new SLM benchmark.

For real integration, these values come from the host's frozen deployment and verifier:

```python
from context_stamps.local_monitor import (
    LocalRegressionMonitor, MonitoredOutcome, MonitorLimits,
)

monitor = LocalRegressionMonitor(
    "domain-monitor.sqlite", scope="maintenance", binding=environment_digest,
    limits=MonitorLimits(
        maximum_failure=0.10, maximum_deadline_rate=0.05,
        deadline_ms=1000, maximum_ram_bytes=2 * 1024**3,
    ),
    alpha=0.01, cohorts=("adaptation", "retention"),
)
# Check that the registry revision has passed qualification before starting.
plan = monitor.start(qualified_policy_digest)
# Persist/recover this plan. It contains no prompt or training text.
monitor.reserve(plan, cluster_id=task_cluster_id, cohort="retention")
# Execute the approved pipeline, measure it, then verify the outcome locally.
status = monitor.observe(plan, MonitoredOutcome(
    cluster_id=task_cluster_id, binding=environment_digest,
    candidate_revision=qualified_policy_digest,
    evidence_revision=raw_verified_record_digest,
    correct=verified_success, wall_ms=complete_pipeline_ms,
    peak_ram_bytes=measured_peak_ram,
))
if status["alarm"]:
    monitor.enforce(plan, learning_registry)
monitor.close()
```

The plan and digests identify records; they do not authenticate labels or establish source truth. Do not substitute guessed measurements. The example RAM ceiling is configurable; sampled process RSS alone may miss child processes, model-server residency or GPU memory. VRAM, energy and learning-cost enforcement still belong to the host and promotion protocol.

## Statistical construction

For each registered cohort, the monitor maintains separate Bernoulli processes for task failure and deadline exceedance. For a declared acceptable conditional rate `p`, each alternative rate `q > p` contributes the likelihood ratio `q/p` on a failure and `(1-q)/(1-p)` on a success. The four alternative rates are frozen at `p + (1-p) * {1/8, 1/4, 1/2, 3/4}`.

To remain sensitive after a long quiet period, the implementation mixes over **every possible start position**. Start `s` receives prior weight `1/(s*(s+1))`; weights sum to one. At observation `t`, the unstarted processes retain total weight `1/(t+1)`. Four log-space accumulators per metric update all those starts in constant arithmetic space. Storage for unique task reservations and receipts still grows with observations.

If the conditional failure probability given the past is at most `p`, each likelihood product is a nonnegative supermartingale. Its weighted mixture therefore is too. Ville's inequality bounds the chance of ever crossing `1/a` by `a`. Epoch `r` receives `alpha/(r*(r+1))`, divided equally over two metrics and all registered cohorts. The sum of the statistical false-alarm budgets is at most `alpha` for this monitor lineage, including repeated observation and cleanly ended epochs. This uses established [time-uniform martingale methods](https://arxiv.org/abs/1808.03204); the particular engineering integration is not a claim to invent those methods.

This bound requires the **conditional null** throughout the stream, honest complete labels, prospective cohort assignment and a frozen deployment/measurement contract. It is stronger than a claim about an overall historical average and is not justified merely by checking that average. Correlated outcomes may satisfy the conditional null, but arbitrary temporal dependence, changing verifiers or selective reporting can invalidate it. Finite arithmetic is checked against a separate rational oracle; this is not a formal numerical proof for every platform.

The statistic is not a probability that a model is wrong, a calibrated answer-confidence score or proof that a shift has occurred. The monitor detects evidence against absolute error/deadline limits; it does not compare against simultaneous baseline answers, estimate a change point, or prove a newly trained policy is better. There is no guaranteed detection time for small or gradual changes. Rare harms can occur before an alarm. Immediate operational alarms are separate from the statistical false-alarm bound.

## Recovery and limits

- One pending cluster per monitor is intentional. After a crash, dispatch remains blocked until the host recovers its actual result or explicitly records it as unknown. A stalled verifier needs a host watchdog; elapsed time is not inferred by this library.
- Reservation IDs cannot be reused, even in a later epoch. Assign one outcome to a repeated source/task cluster according to the frozen host contract; unique strings alone do not make repeated tasks independent.
- Limits, environment binding, cohort names and lifetime risk budget cannot change when reopening the file. A candidate cannot restart its history in the same monitor. At most 128 epochs and 100,000 reservations are supported; exhaustion stops new work. The host must not reset the database or rename the candidate to evade those limits.
- Recover the latest frozen plan with `current_plan()` after a restart. On startup, inspect `status`, reconcile a pending task and retry `enforce` for a durable alarm. Monitor and learning registries are separate databases, so this is an idempotent recovery protocol, not an atomic cross-database transaction.
- Rollback compares the expected active digest inside the learning registry's SQLite write transaction. An old alarm cannot revoke a newer revision. The host must still serialize policy selection and dispatch; rollback cannot cancel work already executing or reverse its external effects.
- A new candidate's evaluation remains separate from live monitoring. Passing a stream without an alarm cannot earn promotion. Retention tasks must actually exercise older skills; cohort labels do not supply those tasks automatically.
- Local SQLite files need host access controls, encryption where required and durable audit integration. The files are not tamper-resistant. Poisoning-resistant verification, multi-device coordination, automatic weight training and target-edge qualification remain unfinished.

For an RTX setup, leave this monitor, the registry and small policy fitting on CPU. Run one useful GPU training or inference workload at a time. Train an adapter only after context-only adaptation and a simple fixed rule have been compared on the same verified cases. Retain the immutable base, evaluate older skills and quantify whole-pipeline resource cost before any local promotion.

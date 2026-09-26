"""Persistent local regression alarms from prospectively reserved outcomes.

Copyright (c) 2026 Prashant Jagtap. MIT License.
This monitors verified failures and missed deadlines, not model confidence.
The trusted host owns labels, reservations, authorization and dispatch. No
model execution, training, telemetry or background service is provided here.
"""

import json
import math
import sqlite3
from dataclasses import asdict, dataclass

from .experience import _hash, _id, _number
from .local_learning import LocalLearningRegistry, _json
from .security import prepare_database


def bernoulli_log_e(failures, trials, limit):
    """Equal mixture of four fixed Bernoulli likelihood-ratio processes.

    Under P(failure | past) <= limit, each product is a nonnegative
    supermartingale. The mixture has initial value one. Crossing 1/alpha
    has probability at most alpha under that conditional null (Ville).
    This is an alarm statistic, not a probability of correctness or drift.
    """
    if (type(trials) is not int or not 0 <= trials <= 100000
            or type(failures) is not int or not 0 <= failures <= trials):
        raise ValueError("bounded integer counts required")
    _number(limit, 1e-6, .5)
    values = []
    for fraction in (.125, .25, .5, .75):
        alternative = limit + (1 - limit) * fraction
        values.append(failures * math.log(alternative / limit)
                      + (trials - failures) * math.log((1 - alternative) / (1 - limit)))
    largest = max(values)
    return largest + math.log(math.fsum(math.exp(v - largest) for v in values) / len(values))


def _log_add(left, right):
    if left is None:
        return right
    top = max(left, right)
    return top + math.log1p(math.exp(min(left, right) - top))


def update_restart_mixture(log_sums, trials, failed, limit):
    """Constant-memory mixture over all deterministic candidate start times.

    Start s has prior mass 1/(s*(s+1)); unstarted processes retain total mass
    1/(t+1). This avoids resetting risk after a quiet period. It can improve
    late-change sensitivity, but does not guarantee a finite detection delay.
    Returns four log weighted sums and the combined log e-value.
    """
    if (type(trials) is not int or not 1 <= trials <= 100000 or type(failed) is not bool
            or type(log_sums) is not tuple or len(log_sums) != 4):
        raise ValueError("bounded sequential state required")
    for value in log_sums:
        if trials == 1:
            if value is not None:
                raise ValueError("initial state must be empty")
        elif type(value) not in (float, int) or not math.isfinite(value):
            raise ValueError("finite previous mixture state required")
    _number(limit, 1e-6, .5)
    weight = -math.log(trials) - math.log(trials + 1)
    updated = []
    for prior, fraction in zip(log_sums, (.125, .25, .5, .75)):
        alternative = limit + (1 - limit) * fraction
        increment = math.log(alternative / limit) if failed else math.log((1 - alternative) / (1 - limit))
        updated.append(_log_add(prior, weight) + increment)
    top = max(updated)
    started = top + math.log(math.fsum(math.exp(v - top) for v in updated) / 4)
    return tuple(updated), _log_add(started, -math.log(trials + 1))


@dataclass(frozen=True)
class MonitorLimits:
    maximum_failure: float = .1
    maximum_deadline_rate: float = .05
    deadline_ms: float = 1000.
    maximum_ram_bytes: int = 2 * 1024**3

    def __post_init__(self):
        _number(self.maximum_failure, 1e-6, .5)
        _number(self.maximum_deadline_rate, 1e-6, .5)
        _number(self.deadline_ms, 1e-9, 3600000)
        if type(self.maximum_ram_bytes) is not int or not 1 <= self.maximum_ram_bytes <= 2**50:
            raise ValueError("positive bounded RAM ceiling required")


@dataclass(frozen=True)
class MonitoredOutcome:
    cluster_id: str
    binding: str
    candidate_revision: str
    evidence_revision: str
    correct: bool | None
    wall_ms: float | None
    peak_ram_bytes: int | None
    safety_violation: bool = False

    def __post_init__(self):
        _id(self.cluster_id)
        for value in (self.binding, self.candidate_revision, self.evidence_revision):
            _hash(value)
        if self.correct is not None and type(self.correct) is not bool:
            raise ValueError("externally verified Boolean correctness or unknown required")
        if type(self.safety_violation) is not bool:
            raise ValueError("Boolean safety label required")
        if self.wall_ms is not None:
            _number(self.wall_ms, 0, 1e15)
        if (self.peak_ram_bytes is not None
                and (type(self.peak_ram_bytes) is not int or not 0 <= self.peak_ram_bytes <= 2**53 - 1)):
            raise ValueError("bounded measured RAM or unknown required")


class LocalRegressionMonitor:
    """One environment, fixed limits, bounded local monitoring history.

    Reserve one task cluster before execution; record it before dispatching
    another. Unknown outcomes alarm immediately. Statistical alarms spend
    alpha/(r*(r+1)) per epoch, split over failure/deadline and every cohort.
    A restart cannot erase a crossing or recover spent error budget. SQLite
    state is neither encrypted nor resistant to an administrator restoring it.
    """

    def __init__(self, path, *, scope, binding, limits, alpha=.01,
                 cohorts=("adaptation", "retention")):
        _id(scope)
        _hash(binding)
        _number(alpha, 1e-6, .25)
        if (type(limits) is not MonitorLimits or type(cohorts) is not tuple
                or not 1 <= len(cohorts) <= 8 or len(set(cohorts)) != len(cohorts)):
            raise ValueError("typed limits and one to eight distinct cohorts required")
        for name in cohorts:
            _id(name)
        prepare_database(path)
        self._db = sqlite3.connect(path, timeout=5, isolation_level=None)
        self._db.execute("PRAGMA journal_mode=DELETE")
        self._db.execute("PRAGMA synchronous=FULL")
        self._db.execute("CREATE TABLE IF NOT EXISTS monitor_config (id INTEGER PRIMARY KEY CHECK(id=1), value TEXT)")
        self._db.execute("CREATE TABLE IF NOT EXISTS epochs (id INTEGER PRIMARY KEY, candidate TEXT UNIQUE, plan TEXT, state TEXT)")
        self._db.execute("CREATE TABLE IF NOT EXISTS outcomes (cluster TEXT PRIMARY KEY, epoch INTEGER, cohort TEXT, result TEXT)")
        self._config = dict(schema=1, scope=scope, binding=binding, limits=asdict(limits),
                            alpha=alpha, cohorts=sorted(cohorts))
        try:
            self._db.execute("BEGIN IMMEDIATE")
            self._db.execute("INSERT OR IGNORE INTO monitor_config VALUES (1, ?)", (_json(self._config),))
            if json.loads(self._db.execute("SELECT value FROM monitor_config WHERE id=1").fetchone()[0]) != self._config:
                raise ValueError("monitor binding, limits, cohorts or risk budget changed")
            self._db.execute("COMMIT")
        except BaseException:
            self._db.execute("ROLLBACK")
            self._db.close()
            raise

    def close(self):
        self._db.close()

    def current_plan(self):
        """Recover the latest frozen plan after a restart, including an alarm."""
        row = self._db.execute("SELECT plan FROM epochs ORDER BY id DESC LIMIT 1").fetchone()
        return json.loads(row[0]) if row else None

    def start(self, candidate):
        """Freeze a previously qualified revision before observing live labels.

        The host must check promotion and serialize dispatch with this call.
        Starting a monitor does not itself qualify or activate the candidate.
        """
        _hash(candidate)
        self._db.execute("BEGIN IMMEDIATE")
        try:
            last = self._db.execute("SELECT state FROM epochs ORDER BY id DESC LIMIT 1").fetchone()
            if last and not json.loads(last[0])["closed"]:
                raise ValueError("finish the current monitoring epoch first")
            if self._db.execute("SELECT 1 FROM epochs WHERE candidate=?", (candidate,)).fetchone():
                raise ValueError("candidate already monitored; do not reset its history")
            epoch = self._db.execute("SELECT COUNT(*) FROM epochs").fetchone()[0] + 1
            if epoch > 128:
                raise ValueError("monitor epoch capacity reached")
            plan = dict(self._config, epoch=epoch, candidate=candidate,
                        per_check_alpha=self._config["alpha"] / (epoch * (epoch + 1) * 2 * len(self._config["cohorts"])))
            state = dict(closed=False, reasons=[], pending=None,
                         counts={c: dict(trials=0, failures=0, late=0,
                                         failures_log_sums=[None]*4, late_log_sums=[None]*4,
                                         failures_log_e=0., late_log_e=0.) for c in self._config["cohorts"]})
            self._db.execute("INSERT INTO epochs VALUES (?, ?, ?, ?)", (epoch, candidate, _json(plan), _json(state)))
            self._db.execute("COMMIT")
            return plan
        except BaseException:
            self._db.execute("ROLLBACK")
            raise

    def _state(self, plan):
        if type(plan) is not dict or type(plan.get("epoch")) is not int:
            raise ValueError("registered monitor plan required")
        row = self._db.execute("SELECT plan, state FROM epochs WHERE id=?", (plan["epoch"],)).fetchone()
        if row is None or json.loads(row[0]) != plan:
            raise ValueError("unknown or modified monitoring plan")
        return json.loads(row[1])

    def status(self, plan):
        state = self._state(plan)
        return dict(state, alarm=bool(state["reasons"]),
                    dispatch_allowed=not state["closed"] and state["pending"] is None)

    def reserve(self, plan, *, cluster_id, cohort):
        """Reserve an opaque unique cluster and its input-defined cohort first."""
        _id(cluster_id)
        _id(cohort)
        self._db.execute("BEGIN IMMEDIATE")
        try:
            state = self._state(plan)
            if state["closed"] or state["pending"] is not None or cohort not in state["counts"]:
                raise ValueError("open monitor, no pending outcome and registered cohort required")
            if self._db.execute("SELECT 1 FROM outcomes WHERE cluster=?", (cluster_id,)).fetchone():
                raise ValueError("cluster already reserved; repeats cannot manufacture new evidence")
            if self._db.execute("SELECT COUNT(*) FROM outcomes").fetchone()[0] >= 100000:
                raise ValueError("monitor observation capacity reached; stop dispatch")
            state["pending"] = cluster_id
            self._db.execute("INSERT INTO outcomes VALUES (?, ?, ?, NULL)", (cluster_id, plan["epoch"], cohort))
            self._db.execute("UPDATE epochs SET state=? WHERE id=?", (_json(state), plan["epoch"]))
            self._db.execute("COMMIT")
        except BaseException:
            self._db.execute("ROLLBACK")
            raise

    def observe(self, plan, outcome):
        if type(outcome) is not MonitoredOutcome:
            raise ValueError("typed measured outcome required")
        self._db.execute("BEGIN IMMEDIATE")
        try:
            state = self._state(plan)
            if (state["closed"] or state["pending"] != outcome.cluster_id
                    or outcome.binding != plan["binding"] or outcome.candidate_revision != plan["candidate"]):
                raise ValueError("outcome must match the reserved task, binding and active epoch")
            cohort = self._db.execute("SELECT cohort FROM outcomes WHERE cluster=?", (outcome.cluster_id,)).fetchone()[0]
            limits = MonitorLimits(**plan["limits"])
            if outcome.safety_violation:
                state["reasons"].append("safety_violation")
            if outcome.correct is None or outcome.wall_ms is None or outcome.peak_ram_bytes is None:
                state["reasons"].append("unknown_outcome_or_resource")
            if outcome.peak_ram_bytes is not None and outcome.peak_ram_bytes > limits.maximum_ram_bytes:
                state["reasons"].append("memory_ceiling")
            if outcome.correct is not None and outcome.wall_ms is not None:
                count = state["counts"][cohort]
                count["trials"] += 1
                count["failures"] += int(not outcome.correct)
                count["late"] += int(outcome.wall_ms > limits.deadline_ms)
                for key, limit in (("failures", limits.maximum_failure), ("late", limits.maximum_deadline_rate)):
                    failed = not outcome.correct if key == "failures" else outcome.wall_ms > limits.deadline_ms
                    sums, log_e = update_restart_mixture(tuple(count[key+"_log_sums"]), count["trials"], failed, limit)
                    count[key+"_log_sums"], count[key+"_log_e"] = sums, log_e
                    if log_e >= -math.log(plan["per_check_alpha"]):
                        state["reasons"].append(f"{cohort}:{key}_alarm")
            state["pending"] = None
            state["closed"] = bool(state["reasons"])
            self._db.execute("UPDATE outcomes SET result=? WHERE cluster=?", (_json(asdict(outcome)), outcome.cluster_id))
            self._db.execute("UPDATE epochs SET state=? WHERE id=?", (_json(state), plan["epoch"]))
            self._db.execute("COMMIT")
            return self.status(plan)
        except BaseException:
            if self._db.in_transaction:
                self._db.execute("ROLLBACK")
            raise

    def end(self, plan):
        """End a clean epoch; unresolved work must first be recorded as unknown."""
        self._db.execute("BEGIN IMMEDIATE")
        try:
            state = self._state(plan)
            if state["pending"] is not None:
                raise ValueError("resolve or explicitly record the unknown pending outcome first")
            state["closed"] = True
            self._db.execute("UPDATE epochs SET state=? WHERE id=?", (_json(state), plan["epoch"]))
            self._db.execute("COMMIT")
        except BaseException:
            self._db.execute("ROLLBACK")
            raise

    def enforce(self, plan, registry):
        """Idempotently request rollback after a durable alarm.

        The databases are separate: retry this at startup before dispatch.
        Compare-and-swap prevents a delayed alarm revoking a newer candidate.
        This cannot cancel work already running or undo its external effects.
        """
        if type(registry) is not LocalLearningRegistry:
            raise ValueError("typed local learning registry required")
        if not self.status(plan)["alarm"] or registry.active_revision(binding=plan["binding"]) is None:
            return False
        return registry.rollback(reason="local_monitor_alarm", expected_active=plan["candidate"])

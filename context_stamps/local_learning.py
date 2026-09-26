"""Local candidate promotion with fixed-sample, alpha-spending checks.

Copyright (c) 2026 Prashant Jagtap. MIT License.
The trusted host supplies independent task clusters, complete measurements and
external outcome labels. This module neither runs models nor authenticates data.
Its SQLite registry is local experiment state, not a tamper-resistant audit log.
"""

import hashlib
import json
import math
import sqlite3
from dataclasses import asdict, dataclass

from .decisions.calibration import error_upper_bound
from .experience import _hash, _id, _number
from .security import prepare_database


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def revision(value):
    """Content identity, not evidence of trust or authorization."""
    return hashlib.sha256(_json(value).encode()).hexdigest()


def _clusters(values, maximum=10000):
    if type(values) is not tuple or len(values) > maximum:
        raise ValueError("bounded immutable cluster IDs required")
    for value in values:
        _id(value)
    if len(set(values)) != len(values):
        raise ValueError("duplicate cluster; repeated trials are not independent tasks")


@dataclass(frozen=True)
class LearningLimits:
    """Freeze units, ceilings and amortization before exposing evaluation labels."""

    metric: str
    cost_cap: float
    minimum_saving: float
    update_cost: float
    amortization_tasks: int
    maximum_new_failure: float = .01
    maximum_failure: float = .1
    deadline_ms: float = 1000.
    maximum_deadline_rate: float = .05
    maximum_ram_bytes: int = 2 * 1024**3

    def __post_init__(self):
        if self.metric not in ("total_tokens", "wall_ms", "energy_joules", "usd", "fixture_units"):
            raise ValueError("explicit supported cost unit required")
        _number(self.cost_cap, 1e-12, 1e12)
        _number(self.minimum_saving, 0, self.cost_cap)
        _number(self.update_cost, 0, 1e15)
        _number(self.deadline_ms, 1e-9, 3600000)
        for value in (self.maximum_new_failure, self.maximum_failure, self.maximum_deadline_rate):
            _number(value, 0, .5)
        if (type(self.amortization_tasks) is not int or not 1 <= self.amortization_tasks <= 1000000
                or type(self.maximum_ram_bytes) is not int or not 1 <= self.maximum_ram_bytes <= 2**50):
            raise ValueError("finite deployment horizon and memory ceiling required")


@dataclass(frozen=True)
class PairedOutcome:
    """One independent task cluster; failures/abstentions count as unsuccessful.

    Cost includes the entire deployed pipeline. Unknown measurements stay None.
    evidence_revision binds an external raw paired record, never prompt text.
    """

    cluster_id: str
    evidence_revision: str
    baseline_correct: bool
    candidate_correct: bool
    baseline_cost: float | None
    candidate_cost: float | None
    candidate_wall_ms: float | None
    candidate_peak_ram_bytes: int | None
    safety_violation: bool = False

    def __post_init__(self):
        _id(self.cluster_id)
        _hash(self.evidence_revision)
        if any(type(v) is not bool for v in
               (self.baseline_correct, self.candidate_correct, self.safety_violation)):
            raise ValueError("verified outcome and safety labels must be Boolean")
        for value in (self.baseline_cost, self.candidate_cost, self.candidate_wall_ms):
            if value is not None:
                _number(value, 0, 1e15)
        if (self.candidate_peak_ram_bytes is not None
                and (type(self.candidate_peak_ram_bytes) is not int
                     or not 0 <= self.candidate_peak_ram_bytes <= 2**53 - 1)):
            raise ValueError("measured memory must be a nonnegative integer or unknown")


@dataclass(frozen=True)
class EvaluationCohort:
    """A host-declared, disjoint task population frozen before labels are read."""

    name: str
    clusters: tuple[str, ...]

    def __post_init__(self):
        _id(self.name)
        _clusters(self.clusters)
        if not self.clusters:
            raise ValueError("an evaluation cohort cannot be empty")


def _cohort_partition(cohorts, clusters):
    if (type(cohorts) is not tuple or not 1 <= len(cohorts) <= 8
            or any(type(c) is not EvaluationCohort for c in cohorts)
            or len({c.name for c in cohorts}) != len(cohorts)):
        raise ValueError("one to eight uniquely named typed cohorts required")
    combined = tuple(key for cohort in cohorts for key in cohort.clusters)
    _clusters(combined)
    if set(combined) != set(clusters):
        raise ValueError("cohorts must partition all and only the registered evaluation clusters")


def assess_candidate(rows, limits, *, alpha):
    """Four simultaneous one-sided checks at alpha/4, fixed sample size.

    New-failure probability bounds net accuracy loss conservatively without
    offsetting harms by gains elsewhere. Cost uses bounded paired differences.
    Requires IID task clusters for the binomial limits, independent bounded cost
    differences, frozen policies and no within-round optional stopping.
    """
    if (type(rows) is not tuple or not 1 <= len(rows) <= 10000
            or any(type(row) is not PairedOutcome for row in rows)
            or type(limits) is not LearningLimits):
        raise ValueError("typed bounded paired observations and limits required")
    _clusters(tuple(row.cluster_id for row in rows))
    _number(alpha, 4e-12, .25)
    failures = []
    if any(row.safety_violation for row in rows):
        failures.append("safety_violation")
    if any(None in (r.baseline_cost, r.candidate_cost, r.candidate_wall_ms,
                    r.candidate_peak_ram_bytes) for r in rows):
        failures.append("missing_measurements")
    if any(r.candidate_peak_ram_bytes is not None and r.candidate_peak_ram_bytes > limits.maximum_ram_bytes
           for r in rows):
        failures.append("memory_ceiling")
    if any(v is not None and v > limits.cost_cap for r in rows for v in (r.baseline_cost, r.candidate_cost)):
        failures.append("cost_out_of_registered_range")
    if limits.metric == "wall_ms" and any(r.candidate_cost != r.candidate_wall_ms for r in rows):
        failures.append("inconsistent_wall_measurement")
    result = {"eligible": False, "reasons": failures, "tasks": len(rows), "alpha": alpha,
              "limits_revision": revision(asdict(limits)),
              "observations_revision": revision([asdict(r) for r in sorted(rows, key=lambda r: r.cluster_id)])}
    if failures:
        return result
    level = alpha / 4
    n = len(rows)
    new_failures = sum(r.baseline_correct and not r.candidate_correct for r in rows)
    errors = sum(not r.candidate_correct for r in rows)
    late = sum(r.candidate_wall_ms > limits.deadline_ms for r in rows)
    harm_upper = error_upper_bound(new_failures, n, level)
    failure_upper = error_upper_bound(errors, n, level)
    deadline_upper = error_upper_bound(late, n, level)
    savings = math.fsum(r.baseline_cost - r.candidate_cost for r in rows) / n
    # Differences are in [-cap, cap]; no clipping of inconvenient observations.
    radius = limits.cost_cap * math.sqrt(2 * math.log(1 / level) / n)
    net_lower = savings - radius - limits.update_cost / limits.amortization_tasks
    if harm_upper > limits.maximum_new_failure:
        failures.append("new_failure_bound")
    if failure_upper > limits.maximum_failure:
        failures.append("absolute_failure_bound")
    if deadline_upper > limits.maximum_deadline_rate:
        failures.append("deadline_bound")
    if net_lower <= limits.minimum_saving:
        failures.append("net_saving_bound")
    result.update(eligible=not failures, new_failures=new_failures, candidate_failures=errors,
                  deadline_exceedances=late, new_failure_upper=harm_upper,
                  failure_upper=failure_upper, deadline_upper=deadline_upper,
                  mean_saving=savings, net_saving_lower=net_lower)
    return result


def assess_cohorts(rows, limits, cohorts, *, alpha):
    """Separate harm/error/deadline checks per cohort; one pooled savings check.

    The round error budget is divided equally over 3*K+1 tests. Union bounding
    does not require independence between cohorts, but each binomial population
    still needs independent identically distributed task clusters. Retention
    need not save resources itself; it must meet quality and deadline limits.
    """
    if (type(rows) is not tuple or not rows or len(rows) > 10000
            or any(type(row) is not PairedOutcome for row in rows)):
        raise ValueError("typed bounded paired observations required")
    _clusters(tuple(row.cluster_id for row in rows))
    _cohort_partition(cohorts, tuple(row.cluster_id for row in rows))
    count = 3 * len(cohorts) + 1
    _number(alpha, count * 1e-12, .25)
    level = alpha / count
    # Reuse the reviewed primitive bounds at the explicit per-check level.
    # Only the checks declared below can influence this composed decision.
    overall = assess_candidate(rows, limits, alpha=4 * level)
    quality = {"new_failure_bound", "absolute_failure_bound", "deadline_bound"}
    overall["reasons"] = [r for r in overall["reasons"] if r not in quality]
    overall["eligible"] = not overall["reasons"]
    overall["applied_statistical_checks"] = ["net_saving_bound"]
    lookup = {r.cluster_id: r for r in rows}
    reports, failures = {}, list(overall["reasons"])
    for cohort in sorted(cohorts, key=lambda c: c.name):
        report = assess_candidate(tuple(lookup[key] for key in sorted(cohort.clusters)), limits, alpha=4 * level)
        report["reasons"] = [r for r in report["reasons"] if r != "net_saving_bound"]
        report["eligible"] = not report["reasons"]
        report["applied_statistical_checks"] = sorted(quality)
        reports[cohort.name] = report
        failures.extend(f"cohort:{cohort.name}:{reason}" for reason in report["reasons"])
    return dict(eligible=not failures, reasons=failures, tasks=len(rows), alpha=alpha,
                per_check_alpha=level, statistical_checks=count, overall=overall, cohorts=reports,
                cohort_partition_revision=revision([{ "name": c.name, "clusters": sorted(c.clusters)}
                                                   for c in sorted(cohorts, key=lambda c: c.name)]))


class LocalLearningRegistry:
    """One scope, one pending trial, bounded lifetime, local-only SQLite state.

    Register before running a fresh holdout. Every round spends alpha/(r(r+1)),
    even if abandoned. Restarts preserve spent rounds and reserved cluster IDs.
    Host binds model/runtime/policy/verifier/device revisions into binding; it
    alone maps an active digest to an already reviewed data-only policy artifact.
    New registries require separate adaptation and retention populations. An
    explicit required_cohorts=() replays legacy pooled trials; it cannot reopen
    or downgrade a protected registry.
    """

    def __init__(self, path, *, scope, binding, baseline, alpha=.05,
                 required_cohorts=("adaptation", "retention")):
        _id(scope)
        _hash(binding)
        _hash(baseline)
        _number(alpha, 1e-6, .25)
        if type(required_cohorts) is not tuple or len(required_cohorts) > 8:
            raise ValueError("bounded unique required cohort names required")
        for name in required_cohorts:
            _id(name)
        if len(set(required_cohorts)) != len(required_cohorts):
            raise ValueError("bounded unique required cohort names required")
        prepare_database(path)
        self._db = sqlite3.connect(path, timeout=5, isolation_level=None)
        self._db.execute("PRAGMA journal_mode=DELETE")
        self._db.execute("PRAGMA synchronous=FULL")
        self._db.execute("CREATE TABLE IF NOT EXISTS config (id INTEGER PRIMARY KEY CHECK(id=1), value TEXT)")
        self._db.execute("CREATE TABLE IF NOT EXISTS trials (round INTEGER PRIMARY KEY, plan TEXT, result TEXT)")
        self._db.execute("CREATE TABLE IF NOT EXISTS used (cluster TEXT PRIMARY KEY)")
        self._db.execute("CREATE TABLE IF NOT EXISTS revoked (revision TEXT PRIMARY KEY)")
        self._db.execute("CREATE TABLE IF NOT EXISTS events (id INTEGER PRIMARY KEY, value TEXT)")
        self._config = {"schema": 1, "scope": scope, "binding": binding, "baseline": baseline, "alpha": alpha}
        if required_cohorts:
            self._config.update(schema=2, required_cohorts=sorted(required_cohorts))
        try:
            self._db.execute("BEGIN IMMEDIATE")
            self._db.execute("INSERT OR IGNORE INTO config VALUES (1, ?)", (_json(self._config),))
            if json.loads(self._db.execute("SELECT value FROM config WHERE id=1").fetchone()[0]) != self._config:
                raise ValueError("registry scope/binding/baseline/confidence mismatch")
            self._db.execute("COMMIT")
        except BaseException:
            self._db.execute("ROLLBACK")
            self._db.close()
            raise

    def close(self):
        self._db.close()

    def active_revision(self, *, binding):
        _hash(binding)
        if binding != self._config["binding"]:
            return None
        item = self._db.execute("SELECT value FROM events ORDER BY id DESC LIMIT 1").fetchone()
        return json.loads(item[0])["active"] if item else self._config["baseline"]

    def register(self, candidate, *, evaluation_clusters, training_clusters, limits, cohorts=()):
        _hash(candidate)
        _clusters(evaluation_clusters)
        _clusters(training_clusters, 100000)
        if not evaluation_clusters or type(limits) is not LearningLimits:
            raise ValueError("nonempty fresh holdout and typed limits required")
        if set(evaluation_clusters) & set(training_clusters):
            raise ValueError("training and evaluation cluster overlap")
        if self._config["schema"] == 2:
            _cohort_partition(cohorts, evaluation_clusters)
            if sorted(c.name for c in cohorts) != self._config["required_cohorts"]:
                raise ValueError("every required cohort must be registered before evaluation")
        elif cohorts != ():
            raise ValueError("legacy pooled registry cannot claim cohort qualification")
        self._db.execute("BEGIN IMMEDIATE")
        try:
            if self._db.execute("SELECT 1 FROM trials WHERE result IS NULL").fetchone():
                raise ValueError("finish or abandon the pending round first")
            if self._db.execute("SELECT 1 FROM revoked WHERE revision=?", (candidate,)).fetchone():
                raise ValueError("revoked candidate cannot be activated again")
            current = self.active_revision(binding=self._config["binding"])
            if candidate == current:
                raise ValueError("candidate must differ from current policy")
            used = {row[0] for row in self._db.execute("SELECT cluster FROM used")}
            if used & set(evaluation_clusters):
                raise ValueError("evaluation clusters were already exposed or reserved")
            combined = used | set(training_clusters) | set(evaluation_clusters)
            if len(combined) > 100000:
                raise ValueError("registry cluster capacity reached")
            r = self._db.execute("SELECT COUNT(*) FROM trials").fetchone()[0] + 1
            if r > 128:
                raise ValueError("registry round capacity reached; do not reset risk accounting")
            plan = {**self._config, "round": r, "baseline": current, "candidate": candidate,
                    "round_alpha": self._config["alpha"] / (r * (r + 1)),
                    "evaluation_clusters": sorted(evaluation_clusters),
                    "training_clusters_revision": revision(sorted(training_clusters)), "limits": asdict(limits)}
            if self._config["schema"] == 2:
                plan["cohorts"] = [{"name": c.name, "clusters": sorted(c.clusters)}
                                   for c in sorted(cohorts, key=lambda c: c.name)]
            self._db.execute("INSERT INTO trials VALUES (?, ?, NULL)", (r, _json(plan)))
            self._db.executemany("INSERT INTO used VALUES (?)", ((c,) for c in sorted(combined - used)))
            self._db.execute("COMMIT")
            return plan
        except BaseException:
            self._db.execute("ROLLBACK")
            raise

    def _pending(self, plan):
        if not isinstance(plan, dict) or type(plan.get("round")) is not int:
            raise ValueError("registered plan required")
        row = self._db.execute("SELECT plan, result FROM trials WHERE round=?", (plan["round"],)).fetchone()
        if row is None or row[1] is not None or json.loads(row[0]) != plan:
            raise ValueError("plan changed, unknown or already consumed")

    def finish(self, plan, rows):
        self._db.execute("BEGIN IMMEDIATE")
        try:
            self._pending(plan)
            if (type(rows) is not tuple or len(rows) != len(plan["evaluation_clusters"])
                    or any(type(r) is not PairedOutcome for r in rows)
                    or sorted(r.cluster_id for r in rows) != plan["evaluation_clusters"]):
                raise ValueError("all and only the registered independent clusters are required")
            if plan["schema"] == 2:
                cohorts = tuple(EvaluationCohort(c["name"], tuple(c["clusters"])) for c in plan["cohorts"])
                report = assess_cohorts(rows, LearningLimits(**plan["limits"]), cohorts, alpha=plan["round_alpha"])
            else:
                report = assess_candidate(rows, LearningLimits(**plan["limits"]), alpha=plan["round_alpha"])
            report["plan_revision"] = revision(plan)
            active = plan["candidate"] if report["eligible"] else plan["baseline"]
            self._db.execute("UPDATE trials SET result=? WHERE round=?", (_json(report), plan["round"]))
            self._db.execute("INSERT INTO events(value) VALUES (?)",
                             (_json({"round": plan["round"], "active": active, "report": revision(report)}),))
            self._db.execute("COMMIT")
            return report
        except BaseException:
            self._db.execute("ROLLBACK")
            raise

    def abandon(self, plan):
        self._db.execute("BEGIN IMMEDIATE")
        try:
            self._pending(plan)
            self._db.execute("UPDATE trials SET result=? WHERE round=?", (_json({"abandoned": True}), plan["round"]))
            self._db.execute("COMMIT")
        except BaseException:
            self._db.execute("ROLLBACK")
            raise

    def rollback(self, *, reason, expected_active=None):
        """Revoke current candidate, abandon pending work and return to anchor.

        A trusted host triggers this on drift, verifier failure, permission or
        device changes. This does not itself detect shift or undo external work.
        expected_active makes delayed monitor requests compare-and-swap: they
        cannot revoke a different, newly activated revision. Returns whether
        the expected revision matched (or no expectation was supplied).
        """
        _id(reason)
        if expected_active is not None:
            _hash(expected_active)
        self._db.execute("BEGIN IMMEDIATE")
        try:
            current = self.active_revision(binding=self._config["binding"])
            if expected_active is not None and current != expected_active:
                self._db.execute("COMMIT")
                return False
            pending = self._db.execute("SELECT 1 FROM trials WHERE result IS NULL").fetchone()
            if current == self._config["baseline"] and pending is None:
                self._db.execute("COMMIT")
                return True
            if current != self._config["baseline"]:
                self._db.execute("INSERT OR IGNORE INTO revoked VALUES (?)", (current,))
            self._db.execute("UPDATE trials SET result=? WHERE result IS NULL", (_json({"abandoned": True}),))
            self._db.execute("INSERT INTO events(value) VALUES (?)",
                             (_json({"active": self._config["baseline"], "reason": reason}),))
            self._db.execute("COMMIT")
            return True
        except BaseException:
            self._db.execute("ROLLBACK")
            raise

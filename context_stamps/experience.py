"""Versioned, reference-only experience contracts for experimental learning.

Copyright (c) 2026 Prashant Jagtap. MIT License.
These records are not an authorization service, executor or durable ledger.
Hosts supply opaque identifiers; sensitive payloads stay in authorized storage.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import asdict, dataclass

SCHEMA_VERSION = 2
MAX_RECORD_BYTES = 65536
SPLITS = frozenset({"train", "tune", "calibration", "test", "development", "regression"})
OUTCOMES = frozenset({"succeeded", "failed", "abstained", "timed_out", "cancelled", "unknown"})


def _id(value):
    if not isinstance(value, str) or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}", value) is None:
        raise ValueError("bounded opaque identifier required; do not put payloads in identifiers")


def _hash(value):
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError("SHA-256 hexadecimal digest required")


def _integer(value, maximum=2**53 - 1):
    if type(value) is not int or not 0 <= value <= maximum:
        raise ValueError("bounded nonnegative integer required")


def _number(value, minimum=0, maximum=1e15):
    if type(value) not in (float, int) or not math.isfinite(value) or not minimum <= value <= maximum:
        raise ValueError("bounded finite number required")


def _members(values, cls, maximum):
    if not isinstance(values, tuple) or len(values) > maximum or any(type(v) is not cls for v in values):
        raise ValueError("bounded immutable typed tuple required")


@dataclass(frozen=True)
class Authority:
    tenant: str
    principal: str
    policy_revision: str

    def __post_init__(self):
        for value in (self.tenant, self.principal, self.policy_revision):
            _id(value)


@dataclass(frozen=True)
class EvidenceRef:
    tenant: str
    source: str
    revision: str
    digest: str
    kind: str

    def __post_init__(self):
        for value in (self.tenant, self.source, self.revision):
            _id(value)
        _hash(self.digest)
        if self.kind not in ("observation", "prediction"):
            raise ValueError("evidence kind must distinguish observation from prediction")


@dataclass(frozen=True)
class Episode:
    episode_id: str
    task_family: str
    project: str
    template: str
    split: str
    environment_revision: str
    seed: int
    authority: Authority
    model_revision: str
    verifier_revision: str
    started_ms: int

    def __post_init__(self):
        for value in (self.episode_id, self.task_family, self.project, self.template,
                      self.environment_revision, self.model_revision, self.verifier_revision):
            _id(value)
        if not isinstance(self.split, str) or self.split not in SPLITS or type(self.authority) is not Authority:
            raise ValueError("declared split and typed authority required")
        _integer(self.seed, 2**32 - 1)
        _integer(self.started_ms)


@dataclass(frozen=True)
class Action:
    tool: str
    tool_revision: str
    arguments: EvidenceRef
    effect: str
    idempotency_key: str

    def __post_init__(self):
        _id(self.tool)
        _id(self.tool_revision)
        _hash(self.idempotency_key)
        if type(self.arguments) is not EvidenceRef or self.arguments.kind != "observation":
            raise ValueError("action inputs must reference an actual recorded proposal")
        if self.effect not in ("pure", "idempotent", "side_effect"):
            raise ValueError("explicit action effect required")


@dataclass(frozen=True)
class ResourceUse:
    input_tokens: int | None
    output_tokens: int | None
    model_calls: int | None
    tool_calls: int | None
    wall_ms: float
    peak_ram_bytes: int | None = None
    peak_vram_bytes: int | None = None
    energy_joules: float | None = None

    def __post_init__(self):
        for value in (self.input_tokens, self.output_tokens, self.model_calls, self.tool_calls):
            if value is not None:
                _integer(value)
        _number(self.wall_ms)
        for value in (self.peak_ram_bytes, self.peak_vram_bytes):
            if value is not None:
                _integer(value)
        if self.energy_joules is not None:
            _number(self.energy_joules)


@dataclass(frozen=True)
class RewardComponent:
    name: str
    value: float

    def __post_init__(self):
        _id(self.name)
        _number(self.value, -1e9, 1e9)


@dataclass(frozen=True)
class TransitionPlan:
    episode_id: str
    step: int
    observations: tuple[EvidenceRef, ...]
    belief_revision: str
    proposed_action: Action
    prediction: EvidenceRef | None
    uncertainty: float | None
    proposed_ms: int

    def __post_init__(self):
        _id(self.episode_id)
        _id(self.belief_revision)
        _integer(self.step, 1000000)
        _integer(self.proposed_ms)
        _members(self.observations, EvidenceRef, 128)
        if len(set(self.observations)) != len(self.observations):
            raise ValueError("duplicate observation reference")
        if any(ref.kind != "observation" for ref in self.observations):
            raise ValueError("predicted evidence cannot be recorded as observation")
        if type(self.proposed_action) is not Action:
            raise ValueError("typed proposed action required")
        if self.prediction is not None and (type(self.prediction) is not EvidenceRef
                                            or self.prediction.kind != "prediction"):
            raise ValueError("prediction must be separately typed")
        if self.uncertainty is not None:
            _number(self.uncertainty, 0, 1)
        if self.prediction is None and self.uncertainty is not None:
            raise ValueError("uncertainty requires a prediction")


@dataclass(frozen=True)
class TransitionOutcome:
    episode_id: str
    step: int
    actual_action: Action | None
    observations: tuple[EvidenceRef, ...]
    status: str
    verified: bool
    verifier_revision: str
    failure_code: str | None
    cost: ResourceUse
    rewards: tuple[RewardComponent, ...]
    observed_ms: int

    def __post_init__(self):
        _id(self.episode_id)
        _id(self.verifier_revision)
        _integer(self.step, 1000000)
        _integer(self.observed_ms)
        if self.actual_action is not None and type(self.actual_action) is not Action:
            raise ValueError("typed actual action required")
        _members(self.observations, EvidenceRef, 128)
        if len(set(self.observations)) != len(self.observations):
            raise ValueError("duplicate observation reference")
        if any(ref.kind != "observation" for ref in self.observations):
            raise ValueError("imagined outcomes are not observed consequences")
        if (not isinstance(self.status, str) or self.status not in OUTCOMES
                or type(self.verified) is not bool or type(self.cost) is not ResourceUse):
            raise ValueError("explicit outcome, verification and measured cost required")
        if self.status == "succeeded":
            if not self.verified or self.actual_action is None or self.failure_code is not None:
                raise ValueError("success requires verified action and no failure")
        elif self.failure_code is None:
            raise ValueError("unsuccessful attempts require a failure or abstention code")
        if self.failure_code is not None:
            _id(self.failure_code)
        _members(self.rewards, RewardComponent, 32)
        if len({r.name for r in self.rewards}) != len(self.rewards):
            raise ValueError("reward components must be unique")


@dataclass(frozen=True)
class EpisodeEnd:
    episode_id: str
    status: str
    ended_ms: int

    def __post_init__(self):
        _id(self.episode_id)
        _integer(self.ended_ms)
        if not isinstance(self.status, str) or self.status not in OUTCOMES:
            raise ValueError("explicit terminal outcome required")


@dataclass(frozen=True)
class DispatchClaim:
    """One durable attempt for an exact proposal, committed before worker startup."""

    episode_id: str
    step: int
    attempt_id: str
    plan_digest: str
    action: Action
    claimed_ms: int
    deadline_ms: int

    def __post_init__(self):
        _id(self.episode_id)
        _integer(self.step, 1000000)
        _hash(self.attempt_id)
        _hash(self.plan_digest)
        _integer(self.claimed_ms)
        _integer(self.deadline_ms)
        if type(self.action) is not Action or not 0 < self.deadline_ms - self.claimed_ms <= 3600000:
            raise ValueError("typed action and dispatch lifetime of at most one hour required")


@dataclass(frozen=True)
class ActionReconciliation:
    """Verified terminal provider status; never replaces the original outcome."""

    episode_id: str
    step: int
    attempt_id: str
    plan_digest: str
    action: Action
    effect_state: str
    observations: tuple[EvidenceRef, ...]
    verifier_revision: str
    observed_ms: int

    def __post_init__(self):
        _id(self.episode_id)
        _integer(self.step, 1000000)
        _hash(self.attempt_id)
        _hash(self.plan_digest)
        _id(self.verifier_revision)
        _integer(self.observed_ms)
        if type(self.action) is not Action or self.effect_state not in ("applied", "not_applied"):
            raise ValueError("typed action and terminal provider effect status required")
        _members(self.observations, EvidenceRef, 128)
        if (not self.observations or len(set(self.observations)) != len(self.observations)
                or any(ref.kind != "observation" for ref in self.observations)):
            raise ValueError("distinct actual provider observations required")


def references(record):
    """Enumerate references without resolving them or granting access."""
    if type(record) is TransitionPlan:
        return record.observations + (record.proposed_action.arguments,) + (
            (record.prediction,) if record.prediction is not None else ())
    if type(record) is TransitionOutcome:
        return record.observations + ((record.actual_action.arguments,) if record.actual_action else ())
    if type(record) is DispatchClaim:
        return (record.action.arguments,)
    if type(record) is ActionReconciliation:
        return record.observations + (record.action.arguments,)
    if type(record) in (Episode, EpisodeEnd):
        return ()
    raise ValueError("experience record required")


def validate_transition(episode, plan, outcome=None):
    """Cross-record checks; append ordering and current ACLs belong to the store."""
    if type(episode) is not Episode or type(plan) is not TransitionPlan:
        raise ValueError("typed episode and plan required")
    if plan.episode_id != episode.episode_id or plan.proposed_ms < episode.started_ms:
        raise ValueError("plan belongs to a different episode or predates it")
    records = [plan]
    if outcome is not None:
        if type(outcome) is not TransitionOutcome:
            raise ValueError("typed outcome required")
        if (outcome.episode_id != episode.episode_id or outcome.step != plan.step
                or outcome.verifier_revision != episode.verifier_revision or outcome.observed_ms < plan.proposed_ms):
            raise ValueError("outcome must match the episode, step, verifier and time order")
        if outcome.actual_action is not None and outcome.actual_action != plan.proposed_action:
            raise ValueError("changed action requires a new proposal before execution")
        records.append(outcome)
    if any(ref.tenant != episode.authority.tenant for record in records for ref in references(record)):
        raise ValueError("foreign tenant reference")


_KINDS = {"episode": Episode, "plan": TransitionPlan, "outcome": TransitionOutcome, "end": EpisodeEnd,
          "dispatch": DispatchClaim, "reconciliation": ActionReconciliation}


def _schema(record):
    # Preserve v1 bytes so existing digests and signed audit journals still replay.
    if type(record) in (DispatchClaim, ActionReconciliation):
        return 2
    if type(record) is TransitionOutcome and any(getattr(record.cost, key) is None for key in
                                                ("input_tokens", "output_tokens", "model_calls", "tool_calls")):
        return 2
    return 1


def encode_record(record):
    kind = next((name for name, cls in _KINDS.items() if type(record) is cls), None)
    if kind is None:
        raise ValueError("experience record required")
    text = json.dumps({"schema": _schema(record), "kind": kind, "record": asdict(record)},
                      sort_keys=True, separators=(",", ":"), allow_nan=False)
    if len(text.encode("utf-8")) > MAX_RECORD_BYTES:
        raise ValueError("record exceeds byte limit")
    return text


def record_digest(record):
    return hashlib.sha256(encode_record(record).encode("utf-8")).hexdigest()


def _unique_object(pairs):
    obj = {}
    for key, value in pairs:
        if key in obj:
            raise ValueError("duplicate JSON key")
        obj[key] = value
    return obj


def _build(cls, data):
    if not isinstance(data, dict) or set(data) != set(cls.__dataclass_fields__):
        raise ValueError("record fields must match the declared schema exactly")
    values = dict(data)
    converters = {
        Episode: {"authority": Authority},
        Action: {"arguments": EvidenceRef},
        TransitionPlan: {"proposed_action": Action, "prediction": EvidenceRef},
        TransitionOutcome: {"actual_action": Action, "cost": ResourceUse},
        DispatchClaim: {"action": Action},
        ActionReconciliation: {"action": Action},
    }.get(cls, {})
    for key, subtype in converters.items():
        if values[key] is not None:
            values[key] = _build(subtype, values[key])
    for key, subtype, maximum in (("observations", EvidenceRef, 128), ("rewards", RewardComponent, 32)):
        if key in values:
            if not isinstance(values[key], list) or len(values[key]) > maximum:
                raise ValueError("bounded JSON array required")
            values[key] = tuple(_build(subtype, value) for value in values[key])
    return cls(**values)


def decode_record(text):
    if not isinstance(text, str) or len(text.encode("utf-8")) > MAX_RECORD_BYTES:
        raise ValueError("bounded JSON record required")
    try:
        obj = json.loads(text, object_pairs_hook=_unique_object)
        if (not isinstance(obj, dict) or set(obj) != {"schema", "kind", "record"}
                or type(obj["schema"]) is not int or obj["schema"] not in (1, SCHEMA_VERSION)
                or not isinstance(obj["kind"], str) or obj["kind"] not in _KINDS):
            raise ValueError("unsupported experience schema")
        record = _build(_KINDS[obj["kind"]], obj["record"])
        if obj["schema"] != _schema(record):
            raise ValueError("record version does not match its schema")
        return record
    except (TypeError, KeyError, RecursionError, OverflowError) as error:
        raise ValueError("malformed experience record") from error

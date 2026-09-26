"""Typed questions and result records. Copyright 2026 Prashant Jagtap, MIT."""

import hashlib
import math
from dataclasses import asdict, dataclass

from ..context_state import EvidenceRequirement, NodeRef, canonical, typed_tuple
from ..security import bounded_text, identifier
from .boolean import Boolean
from .choice import Choice
from .score import Score


@dataclass(frozen=True)
class Question:
    question_id: str
    text: str
    spec: Choice | Boolean | Score
    requirements: tuple[EvidenceRequirement, ...]
    scope: str

    def __post_init__(self):
        identifier(self.question_id)
        identifier(self.scope)
        bounded_text(self.text, 4096)
        if type(self.spec) not in (Choice, Boolean, Score):
            raise ValueError("typed decision specification required")
        typed_tuple(self.requirements, EvidenceRequirement, 12)
        if not self.requirements:
            raise ValueError("explicit required evidence needed")

    @property
    def decision_type(self):
        return type(self.spec).__name__.lower()

    @property
    def schema_id(self):
        return hashlib.sha256(canonical({"type": self.decision_type, "spec": asdict(self.spec),
                                         "question": self.text,
                                         "requirements": [asdict(r) for r in self.requirements]}).encode()).hexdigest()


@dataclass(frozen=True)
class Proposal:
    question_id: str
    value: str | bool | float | None
    confidence: float | None = None

    def __post_init__(self):
        identifier(self.question_id)
        if type(self.value) not in (str, bool, float, int, type(None)):
            raise ValueError("bounded scalar proposal required")
        if type(self.value) is str:
            bounded_text(self.value, 4096)
        if type(self.value) in (float, int) and (not math.isfinite(self.value) or abs(self.value) > 1e12):
            raise ValueError("invalid numeric proposal")
        if self.confidence is not None and (type(self.confidence) not in (int, float)
                or not math.isfinite(self.confidence) or not 0 <= self.confidence <= 1):
            raise ValueError("invalid confidence")


@dataclass(frozen=True)
class Decision:
    question_id: str
    decision_type: str
    result: str | bool | float | None
    probability: float | None
    calibration_error: float | None
    evidence_ids: tuple[NodeRef, ...]
    stamp_ids: tuple[str, ...]
    state_revision: str | None
    abstained: bool
    reason: str
    model_revision: str
    verifier_revision: str
    calibration_revision: str | None
    receipt: str

    def to_dict(self):
        return asdict(self)


def result_record(question, context, *, result=None, reason, verifier_revision,
                  probability=None, calibration_error=None, calibration_revision=None):
    abstained = reason != "verified_decision"
    nodes = () if abstained or context.snapshot is None else tuple(r.node for r in context.snapshot.records)
    payload = dict(question_id=question.question_id, decision_type=question.decision_type,
        result=None if abstained else result, probability=None if abstained else probability,
        calibration_error=None if abstained else calibration_error, evidence_ids=tuple(node.ref for node in nodes),
        stamp_ids=tuple(sorted({node.stamp_schema + ":" + node.stamp_hex for node in nodes if node.stamp_hex})),
        state_revision=None if abstained else context.snapshot.state_revision,
        abstained=abstained, reason=reason, model_revision=context.profile.model_revision,
        verifier_revision=verifier_revision, calibration_revision=calibration_revision)
    # This inspectable digest is not an authorization credential or a signature.
    material = {**payload, "evidence_ids": [asdict(ref) for ref in payload["evidence_ids"]],
                "compiled_binding": context.binding,
                "question_schema": question.schema_id, "question_text": question.text,
                "formatter_revision": context.profile.formatter_revision,
                "tokenizer_revision": context.profile.tokenizer_revision}
    receipt = hashlib.sha256(canonical(material).encode()).hexdigest()
    return Decision(**payload, receipt=receipt)

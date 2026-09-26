"""One vectorized provider call with scoped calibration and final revalidation."""

import time

from ..context_compiler import CompiledContext, compiled_material
from ..security import identifier
from .abstention import DecisionPolicy
from .calibration import Calibration
from .schema import Proposal, Question, result_record


def decide_batch(questions, *, context, state, provider, verify, verifier_revision,
                 policy=DecisionPolicy(), calibrations=()):
    identifier(verifier_revision)
    if (type(questions) is not tuple or not 1 <= len(questions) <= 32
            or any(type(q) is not Question for q in questions)
            or len({q.question_id for q in questions}) != len(questions)
            or type(context) is not CompiledContext or not callable(provider) or not callable(verify)
            or type(policy) is not DecisionPolicy or type(calibrations) is not tuple
            or len(calibrations) > 32 or any(type(c) is not Calibration for c in calibrations)):
        raise ValueError("bounded typed batch and explicit provider/verifier required")
    calibration_keys = [(c.model_revision, c.scope, c.schema_id) for c in calibrations]
    if len(set(calibration_keys)) != len(calibration_keys):
        raise ValueError("ambiguous calibration scope")

    def abstain(reason):
        return tuple(result_record(q, context, reason=reason, verifier_revision=verifier_revision) for q in questions)

    if context.status != "complete" or context.snapshot is None:
        return abstain("insufficient_context")
    if any(not set(q.requirements) <= set(context.task.requirements) for q in questions):
        return abstain("uncompiled_requirements")
    if not state.is_current(context.snapshot):
        return abstain("state_changed")
    try:
        valid_binding = state.verify_binding(context.snapshot, compiled_material(context), context.binding)
    except (ValueError, TypeError, AttributeError, OverflowError):
        valid_binding = False
    if not valid_binding:
        return abstain("invalid_context_binding")
    if time.monotonic() >= context.deadline:
        return abstain("latency_budget")
    try:
        proposals = provider(questions, context.text)
        if (type(proposals) is not tuple or len(proposals) != len(questions)
                or any(type(p) is not Proposal for p in proposals)
                or {p.question_id for p in proposals} != {q.question_id for q in questions}):
            return abstain("invalid_provider_result")
        mapping = {p.question_id: p for p in proposals}
        results = []
        for question in questions:
            if time.monotonic() >= context.deadline:
                return abstain("latency_budget")
            proposal = mapping[question.question_id]
            reason = "verified_decision" if question.spec.accepts(proposal.value) else "invalid_decision_type"
            calibrated, certificate = None, None
            for calibration in calibrations:
                item = calibration.lookup(proposal.confidence, model_revision=context.profile.model_revision,
                                          scope=question.scope, schema_id=question.schema_id,
                                          verifier_revision=verifier_revision,
                                          policy_revision=context.snapshot.scope.policy_revision)
                if item is not None:
                    calibrated, certificate = item, calibration
            if policy.minimum_probability is not None and (calibrated is None
                    or calibrated.lower_correctness < policy.minimum_probability):
                reason = "uncalibrated_or_uncertain"
            if reason == "verified_decision" and verify(question, proposal.value, context) is not True:
                reason = "unverified_decision"
            results.append(result_record(question, context, result=proposal.value, reason=reason,
                verifier_revision=verifier_revision, probability=calibrated.probability if calibrated else None,
                calibration_error=certificate.ece if certificate else None,
                calibration_revision=certificate.revision if certificate else None))
    except Exception:
        # External exceptions can contain source text or credentials. Do not echo.
        return abstain("adapter_error")
    if time.monotonic() >= context.deadline:
        return abstain("latency_budget")
    if not state.is_current(context.snapshot):
        return abstain("state_changed")
    return tuple(results)

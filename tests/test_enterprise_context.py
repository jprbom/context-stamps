"""Temporal, authority, compiler and decision tests with fictional records."""

import itertools
import json
import math
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from unittest.mock import patch

from context_stamps.context_compiler import (
    CompileBudget,
    ConflictPolicy,
    ContextCompiler,
    ContextTask,
    ModelProfile,
    default_render,
)
from context_stamps.context_state import (
    AccessScope,
    CanonicalNode,
    ContextClaim,
    ContextState,
    EvidenceRequirement,
    TemporalScope,
)
from context_stamps.decisions import (
    Boolean,
    CalibrationSample,
    Choice,
    DecisionPolicy,
    Proposal,
    Question,
    Score,
    decide_batch,
    fit_calibration,
)
from context_stamps.decisions.calibration import CalibrationBin, error_upper_bound


class EnterpriseTests(unittest.TestCase):
    def setUp(self):
        self.now = 100
        self.state = ContextState(tenant="lab", policy_revision="p1", clock=lambda: self.now)
        self.scope = AccessScope("lab", "person", "p1", ("reader",))
        self.requirement = EvidenceRequirement("limit")
        self.task = ContextTask("decision", "Check the experiment", (self.requirement,))
        self.profile = ModelProfile("exact-v1")

    def node(self, key="policy", revision="v1", value="32", **changes):
        fields = dict(key=key, revision=revision, tenant="lab", text="batch_limit=" + value,
                      kind="POLICY", temporal=TemporalScope(0, 10), roles=("reader",), provenance="manifest",
                      claims=(ContextClaim("limit", value),), validation_revision="checked-v1")
        fields.update(changes)
        return CanonicalNode(**fields)

    def compile(self, **changes):
        args = dict(scope=self.scope, at=100, known_at=1000, profile=self.profile,
                    budget=CompileBudget(milliseconds=10000))
        args.update(changes)
        return ContextCompiler(self.state).compile(self.task, **args)

    def question(self, spec=None, **changes):
        args = dict(question_id="q", text="Is the limit 32?", spec=spec or Boolean(),
                    requirements=(self.requirement,), scope="experiment-v1")
        args.update(changes)
        return Question(**args)

    def decide(self, context, question=None, **changes):
        question = question or self.question()
        args = dict(context=context, state=self.state, provider=lambda q, text: (Proposal(q[0].question_id, True, .95),),
                    verify=lambda q, value, c: value is True, verifier_revision="verify-v1")
        args.update(changes)
        return decide_batch((question,), **args)[0]

    def test_effective_and_transaction_times_are_independent(self):
        old = self.node()
        self.state.put(old)
        self.now = 200
        revised = self.node(revision="v2", value="64", temporal=TemporalScope(50, 150), supersedes=(old.ref,))
        self.state.put(revised)
        def values(at, known):
            return [r.node.claims[0].value for r in self.state.snapshot(self.scope, at=at, known_at=known).records]
        self.assertEqual(values(100, 150), ["32"])
        self.assertEqual(values(100, 200), ["64"])
        self.assertEqual(values(40, 200), ["32"])
        self.assertEqual(values(100, 99), [])

    def test_expired_replacement_does_not_resurrect_predecessor(self):
        old = self.node()
        self.state.put(old)
        self.now = 200
        revised = self.node(revision="v2", temporal=TemporalScope(50, 150, valid_until=120), supersedes=(old.ref,))
        self.state.put(revised)
        self.assertEqual(self.compile(at=130).status, "insufficient")
        self.state.invalidate(revised.ref)
        self.assertEqual(self.compile(at=100).status, "insufficient")

    def test_new_revision_cannot_overwrite_payload_and_does_not_change_acl(self):
        old = self.node()
        first = self.state.put(old)
        self.assertIs(self.state.put(old), first)
        with self.assertRaises(ValueError):
            self.state.put(replace(old, text="altered"))
        self.state.set_roles(old.key, ())
        self.state.put(self.node(revision="v2", supersedes=(old.ref,)))
        self.assertEqual(self.compile().text, "")

    def test_current_acl_applies_to_historical_queries(self):
        self.state.put(self.node())
        snapshot = self.state.snapshot(self.scope, at=50, known_at=100)
        self.state.set_roles("policy", ("admin",))
        self.assertFalse(self.state.is_current(snapshot))
        self.assertEqual(self.state.snapshot(self.scope, at=50, known_at=100).records, ())
        self.assertEqual(self.compile(scope=replace(self.scope, tenant="other")).reason, "unavailable_context")
        self.state.set_policy("p2")
        self.assertEqual(self.compile().reason, "unavailable_context")

    def test_snapshot_forgery_is_rejected(self):
        self.state.put(self.node())
        snapshot = self.state.snapshot(self.scope, at=100, known_at=100)
        forged = replace(snapshot, records=(replace(snapshot.records[0], node=replace(self.node(), text="fake")),))
        self.assertFalse(self.state.is_current(forged))
        self.assertFalse(self.state.is_current(replace(snapshot, at=99)))
        other = ContextState(tenant="lab", policy_revision="p1", clock=lambda: 100)
        self.assertFalse(other.is_current(snapshot))

    def test_negative_knowledge_has_expiry_and_never_means_false(self):
        claim = ContextClaim("limit", None, "KNOWN_ABSENT")
        with self.assertRaises(ValueError):
            self.node(claims=(claim,))
        self.state.put(self.node(claims=(claim,), temporal=TemporalScope(0, 10, valid_until=120)))
        self.assertEqual(self.compile().reason, "missing_evidence")
        self.requirement = EvidenceRequirement("limit", statuses=("KNOWN_ABSENT",))
        self.task = replace(self.task, requirements=(self.requirement,))
        self.assertEqual(self.compile().status, "complete")
        self.assertEqual(self.compile(at=120).status, "insufficient")

    def test_model_output_and_assumption_cannot_fulfil_verified_fact(self):
        for kind in ("MODEL_OUTPUT", "ASSUMPTION", "PREDICTION", "CLAIM"):
            self.state.put(self.node(key=kind, kind=kind))
        self.assertEqual(self.compile().reason, "missing_evidence")
        with self.assertRaises(ValueError):
            EvidenceRequirement("limit", kinds=("MODEL_OUTPUT",))
        explicit = EvidenceRequirement("limit", kinds=("MODEL_OUTPUT",), verified=False)
        self.task = replace(self.task, requirements=(explicit,))
        self.assertEqual(self.compile().status, "complete")

    def test_dependency_closure_cannot_disclose_private_identifier(self):
        hidden = self.node(key="hidden-private-source", roles=("admin",), claims=())
        self.state.put(hidden)
        self.state.put(self.node(dependencies=(hidden.ref,)))
        snapshot = self.state.snapshot(self.scope, at=100, known_at=100)
        self.assertEqual(snapshot.records, ())
        result = self.compile()
        self.assertEqual(result.text, "")
        self.assertNotIn("hidden-private-source", repr(result))

    def test_exact_minimum_accounts_for_whole_dependencies_and_metadata(self):
        dependency = self.node(key="large-dependency", text="x" * 3000, claims=())
        self.state.put(dependency)
        self.state.put(self.node(key="a", text="a", dependencies=(dependency.ref,)))
        self.state.put(self.node(key="b", text="b" * 100))
        packet = self.compile()
        self.assertTrue(packet.optimal)
        self.assertEqual([r.node.key for r in packet.snapshot.records], ["b"])
        self.assertEqual(packet.bytes, len(packet.text.encode()))
        too_small = self.compile(budget=CompileBudget(bytes=packet.bytes - 1))
        self.assertEqual(too_small.text, "")

    def test_true_token_budget_includes_serialized_provenance(self):
        self.state.put(self.node())
        profile = ModelProfile("test", tokenizer_revision="character-fixture", count_tokens=len)
        packet = self.compile(profile=profile)
        self.assertEqual(packet.tokens, len(packet.text))
        self.assertEqual(self.compile(profile=profile, budget=CompileBudget(tokens=packet.tokens - 1)).status, "insufficient")
        self.assertEqual(self.compile(profile=profile, budget=CompileBudget(tokens=packet.tokens)).status, "complete")
        with self.assertRaises(ValueError):
            self.compile(budget=CompileBudget(tokens=10))

    def test_conflicts_retained_and_explicit_policy_is_auditable(self):
        self.state.put(self.node(key="a", authority=80))
        self.state.put(self.node(key="b", value="64", authority=20))
        strict = self.compile()
        self.assertEqual(strict.reason, "conflicting_context")
        self.assertEqual(strict.conflicts[0].status, "UNRESOLVED")
        chosen = self.compile(conflict_policy=ConflictPolicy("authority-first", ("authority",)))
        self.assertEqual(chosen.conflicts[0].status, "RESOLVED")
        self.assertEqual(chosen.conflicts[0].selected_value, ("PRESENT", "32"))
        self.assertEqual([r.node.key for r in chosen.snapshot.records], ["a"])
        self.assertEqual(len(self.state.snapshot(self.scope, at=100, known_at=100).records), 2)

    def test_resource_limits_and_larger_pool_never_claim_global_minimum(self):
        for i in range(17):
            self.state.put(self.node(key=f"source-{i}"))
        self.assertFalse(self.compile().optimal)
        with self.assertRaises(ValueError):
            CompileBudget(milliseconds=float("nan"))
        with self.assertRaises(ValueError):
            TemporalScope(True, 10)
        tiny = ContextState(tenant="lab", policy_revision="p1", clock=lambda: 100, capacity=1)
        tiny.put(self.node())
        with self.assertRaises(ValueError):
            tiny.put(self.node(key="other"))

    def test_changed_state_or_bad_formatter_never_releases_a_packet(self):
        self.state.put(self.node())
        def changed(task, profile, records):
            self.state.set_roles("policy", ())
            return default_render(task, profile, records)
        profile = ModelProfile("test", render=changed, verify_render=lambda t, r, s: True)
        self.assertEqual(self.compile(profile=profile).reason, "state_changed")
        self.state.set_roles("policy", ("reader",))
        profile = ModelProfile("test", render=lambda *args: "omitted evidence", verify_render=lambda *args: False)
        self.assertEqual(self.compile(profile=profile).text, "")
        def raises(*args):
            raise RuntimeError("private adapter diagnostic")
        bad = replace(profile, render=raises)
        self.assertEqual(self.compile(profile=bad).reason, "invalid_adapter_result")

    def test_typed_batch_one_provider_call_and_exact_verification(self):
        self.state.put(self.node())
        context = self.compile()
        questions = (self.question(), self.question(Choice(("accept", "reject")), question_id="choice"),
                     self.question(Score(0, 100), question_id="score"))
        calls = []
        def provider(q, text):
            calls.append(json.loads(text))
            return (Proposal("score", 32), Proposal("q", True), Proposal("choice", "accept"))
        decisions = decide_batch(questions, context=context, state=self.state, provider=provider,
                                 verify=lambda *args: True, verifier_revision="verify-v1")
        self.assertEqual([d.result for d in decisions], [True, "accept", 32])
        self.assertEqual(len(calls), 1)
        self.assertTrue(all(d.probability is None and len(d.receipt) == 64 for d in decisions))

    def test_provider_confidence_is_not_automatically_a_probability(self):
        self.state.put(self.node())
        context = self.compile()
        result = self.decide(context)
        self.assertFalse(result.abstained)
        self.assertIsNone(result.probability)
        self.assertTrue(self.decide(context, policy=DecisionPolicy(.9)).abstained)

    def test_boolean_score_choice_and_unknown_batch_fields(self):
        self.state.put(self.node())
        context = self.compile()
        bad_cases = ((Boolean(), 1), (Choice(("yes", "no")), "maybe"), (Score(0, 1), True), (Score(0, 1), 2))
        for spec, value in bad_cases:
            outcome = self.decide(context, self.question(spec), provider=lambda q, t: (Proposal("q", value),))
            self.assertEqual(outcome.reason, "invalid_decision_type")
            self.assertEqual(outcome.evidence_ids, ())
        self.assertEqual(self.decide(context, provider=lambda q, t: (Proposal("wrong-id", True),)).reason, "invalid_provider_result")
        self.assertEqual(self.decide(context, verify=lambda *args: False).reason, "unverified_decision")

    def test_altered_packet_or_model_binding_abstains_before_provider(self):
        self.state.put(self.node())
        context = self.compile()
        for changed in (replace(context, text="forged"), replace(context, deadline=context.deadline + 1),
                        replace(context, profile=ModelProfile("other-model"))):
            self.assertEqual(self.decide(changed, provider=lambda *args: self.fail("provider called")).reason,
                             "invalid_context_binding")

    def test_revocation_during_callback_is_nonblocking_and_rejects_all_results(self):
        self.state.put(self.node())
        context = self.compile()
        started, proceed = threading.Event(), threading.Event()
        def provider(q, text):
            started.set()
            self.assertTrue(proceed.wait(3))
            return (Proposal("q", True),)
        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(self.decide, context, provider=provider)
            self.assertTrue(started.wait(3))
            self.state.set_roles("policy", ())
            proceed.set()
            result = future.result(timeout=3)
        self.assertEqual(result.reason, "state_changed")
        self.assertIsNone(result.result)

    def test_expired_end_to_end_deadline_and_requirement_mismatch(self):
        self.state.put(self.node())
        context = self.compile()
        with patch("context_stamps.decisions.batch.time.monotonic", return_value=context.deadline + 1):
            self.assertEqual(self.decide(context).reason, "latency_budget")
        missing = self.question(requirements=(EvidenceRequirement("not-compiled"),))
        self.assertEqual(self.decide(context, missing).reason, "uncompiled_requirements")

    def test_calibration_bounds_and_scope_fail_closed(self):
        self.state.put(self.node())
        context, question = self.compile(), self.question()
        samples = tuple(CalibrationSample(f"cluster-{i}", .95, True) for i in range(300))
        cal = fit_calibration(samples, model_revision="exact-v1", verifier_revision="verify-v1", policy_revision="p1",
                              scope=question.scope, schema_id=question.schema_id, bins=1)
        result = self.decide(context, question, calibrations=(cal,), policy=DecisionPolicy(.99))
        self.assertFalse(result.abstained)
        self.assertEqual(result.probability, 1.)
        self.assertAlmostEqual(result.calibration_error, .05)
        self.assertTrue(self.decide(context, replace(question, text="Is the limit different?"),
                                   calibrations=(cal,), policy=DecisionPolicy(.99)).abstained)
        self.assertTrue(self.decide(context, question, verifier_revision="verify-v2", calibrations=(cal,),
                                   policy=DecisionPolicy(.99)).abstained)
        with self.assertRaises(ValueError):
            fit_calibration(samples + samples[:1], model_revision="exact-v1", verifier_revision="v1", policy_revision="p1",
                            scope="test", schema_id=question.schema_id)

    def test_binomial_bound_matches_independent_polynomial(self):
        self.assertLess(error_upper_bound(0, 300), .01)
        self.assertEqual(error_upper_bound(10, 10), 1.)
        for errors in (1, 3, 8):
            p = error_upper_bound(errors, 10)
            cdf = sum(math.comb(10, k) * p**k * (1-p)**(10-k) for k in range(errors+1))
            self.assertAlmostEqual(cdf, .05, places=10)

    def test_global_cost_matches_independent_enumeration_on_small_fixture(self):
        requirements = tuple(EvidenceRequirement(name) for name in ("a", "b", "c"))
        self.task = replace(self.task, requirements=requirements)
        dependency = self.node(key="dependency", claims=(), text="d" * 500)
        self.state.put(dependency)
        for i, names in enumerate((("a",), ("b",), ("c",), ("a", "b"), ("b", "c"), ("a", "b", "c"))):
            self.state.put(self.node(key=f"source-{i}", text="e" * (i * 300),
                claims=tuple(ContextClaim(name, "yes") for name in names),
                dependencies=(dependency.ref,) if i == 5 else ()))
        records = self.state.snapshot(self.scope, at=100, known_at=100).records
        candidates = []
        # Independent oracle enumerates all records rather than compiler root closures.
        for size in range(1, len(records) + 1):
            for selected in itertools.combinations(records, size):
                refs = {r.node.ref for r in selected}
                if any(not set(r.node.dependencies) <= refs for r in selected):
                    continue
                if {c.name for r in selected for c in r.node.claims} != {"a", "b", "c"}:
                    continue
                candidates.append(len(default_render(self.task, self.profile, selected).encode()))
        result = self.compile()
        self.assertTrue(result.optimal)
        self.assertEqual(result.bytes, min(candidates))
        limited = self.compile(budget=CompileBudget(combinations=1))
        self.assertFalse(limited.optimal)
        self.assertEqual(limited.reason, "search_budget")

    def test_equal_authority_conflict_does_not_select_arbitrary_winner(self):
        self.state.put(self.node(key="a", authority=80))
        self.state.put(self.node(key="b", authority=80, value="64"))
        result = self.compile(conflict_policy=ConflictPolicy("authority-v1", ("authority",)))
        self.assertEqual(result.reason, "conflicting_context")
        self.assertEqual(result.conflicts[0].candidates, ("a", "b"))

    def test_transitive_dependency_revocation_hides_all_derived_nodes(self):
        source = self.node(key="source", claims=())
        derived = self.node(key="derived", claims=(), dependencies=(source.ref,))
        self.state.put(source)
        self.state.put(derived)
        self.state.put(self.node(dependencies=(derived.ref,)))
        self.assertEqual(len(self.compile().snapshot.records), 3)
        self.state.set_roles("source", ())
        self.assertEqual(self.state.snapshot(self.scope, at=100, known_at=100).records, ())

    def test_invalid_tokenizer_result_and_nonfinite_packet_are_rejected(self):
        self.state.put(self.node())
        for value in (True, -1, float("nan"), "4"):
            profile = ModelProfile("test", tokenizer_revision="bad", count_tokens=lambda s: value)
            self.assertEqual(self.compile(profile=profile).reason, "invalid_adapter_result")
        context = self.compile()
        self.assertEqual(self.decide(replace(context, deadline=float("inf"))).reason, "invalid_context_binding")

    def test_calibration_rejects_tampering_and_empty_confidence_claims(self):
        for args in ((0, 1, 0, 1., 1.), (0, 1, 1, .5, .8), (0, 1, 1, float("nan"), 0)):
            with self.assertRaises(ValueError):
                CalibrationBin(*args)
        question = self.question()
        samples = (CalibrationSample("c1", .2, False), CalibrationSample("c2", .8, True))
        cal = fit_calibration(samples, model_revision="exact-v1", verifier_revision="verify-v1", policy_revision="p1",
                              scope=question.scope, schema_id=question.schema_id, bins=2, minimum_count=1)
        for changes in ({"model_revision": "other"}, {"ece": 0.}, {"alpha": .5}, {"minimum_count": 2}):
            with self.assertRaises(ValueError):
                replace(cal, **changes)
        args = dict(model_revision="exact-v1", verifier_revision="verify-v1", policy_revision="p1",
                    scope=question.scope, schema_id=question.schema_id)
        self.assertIs(cal.lookup(.5, **args), cal.bins[1])
        self.assertIs(cal.lookup(1., **args), cal.bins[1])
        self.assertIsNone(cal.lookup(True, **args))

    def test_canonical_media_requires_external_digest(self):
        for modality in ("image", "audio", "video"):
            with self.assertRaises(ValueError):
                self.node(modality=modality)
            node = self.node(modality=modality, artifact_digest="a" * 64)
            self.assertNotEqual(node.ref, self.node().ref)


if __name__ == "__main__":
    unittest.main()

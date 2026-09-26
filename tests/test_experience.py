"""Experience schema invariants; these do not test a durable event store."""

import hashlib
import json
import unittest
from dataclasses import FrozenInstanceError, replace

from context_stamps.experience import (
    Action,
    Authority,
    Episode,
    EpisodeEnd,
    EvidenceRef,
    ResourceUse,
    RewardComponent,
    TransitionOutcome,
    TransitionPlan,
    decode_record,
    encode_record,
    record_digest,
    validate_transition,
)


def fixtures():
    digest = hashlib.sha256(b"fictional recorded input").hexdigest()
    ref = EvidenceRef("lab", "config", "v1", digest, "observation")
    predicted = replace(ref, source="prediction", kind="prediction")
    episode = Episode("ep1", "research", "project-a", "template-a", "train", "env-v1", 7,
                      Authority("lab", "researcher", "policy-v1"), "model-v1", "verifier-v1", 100)
    action = Action("inspect", "tool-v1", ref, "pure", digest)
    plan = TransitionPlan("ep1", 0, (ref,), "belief-v1", action, predicted, .25, 101)
    result = TransitionOutcome("ep1", 0, action, (ref,), "succeeded", True, "verifier-v1", None,
                               ResourceUse(20, 3, 1, 0, 50.), (RewardComponent("task", 1.),), 102)
    return episode, plan, result


class ExperienceTests(unittest.TestCase):
    def test_canonical_round_trip_and_digest(self):
        episode, plan, outcome = fixtures()
        validate_transition(episode, plan, outcome)
        for record in (episode, plan, outcome, EpisodeEnd("ep1", "succeeded", 103)):
            self.assertEqual(record, decode_record(encode_record(record)))
            obj = json.loads(encode_record(record))
            self.assertEqual(record_digest(record), record_digest(decode_record(json.dumps(obj, indent=4))))
        self.assertNotEqual(record_digest(plan), record_digest(replace(plan, belief_revision="belief-v2")))
        with self.assertRaises(FrozenInstanceError):
            episode.seed = 3

    def test_negative_outcomes_are_first_class_and_not_success(self):
        _, _, result = fixtures()
        for status in ("failed", "abstained", "timed_out", "cancelled", "unknown"):
            negative = replace(result, status=status, verified=False, actual_action=None,
                               failure_code="missing-evidence", rewards=(RewardComponent("task", 0.),))
            self.assertEqual(decode_record(encode_record(negative)), negative)
        for changes in ({"verified": False}, {"actual_action": None}, {"failure_code": "failure"}):
            with self.assertRaises(ValueError):
                replace(result, **changes)

    def test_predictions_never_become_observations(self):
        _, plan, result = fixtures()
        with self.assertRaises(ValueError):
            replace(plan, observations=(plan.prediction,))
        with self.assertRaises(ValueError):
            replace(result, observations=(plan.prediction,))
        with self.assertRaises(ValueError):
            replace(plan, prediction=plan.observations[0])
        with self.assertRaises(ValueError):
            replace(plan.proposed_action, arguments=plan.prediction)

    def test_cross_tenant_and_cross_episode_rejected(self):
        episode, plan, result = fixtures()
        foreign = replace(plan.observations[0], tenant="elsewhere")
        with self.assertRaises(ValueError):
            validate_transition(episode, replace(plan, observations=(foreign,)))
        with self.assertRaises(ValueError):
            validate_transition(episode, replace(plan, episode_id="ep2"))
        with self.assertRaises(ValueError):
            validate_transition(episode, plan, replace(result, step=1))

    def test_verifier_and_action_bindings(self):
        episode, plan, result = fixtures()
        with self.assertRaises(ValueError):
            validate_transition(episode, plan, replace(result, verifier_revision="verifier-v2"))
        with self.assertRaises(ValueError):
            validate_transition(episode, plan, replace(result, actual_action=replace(result.actual_action, tool="write")))

    def test_time_order_rejects_hindsight_timestamps(self):
        episode, plan, result = fixtures()
        with self.assertRaises(ValueError):
            validate_transition(episode, replace(plan, proposed_ms=99))
        with self.assertRaises(ValueError):
            validate_transition(episode, plan, replace(result, observed_ms=100))
        # Timestamp validation alone is not proof of append-before-action ordering.

    def test_nonfinite_out_of_range_and_boolean_numbers(self):
        _, plan, result = fixtures()
        for value in (float("nan"), float("inf"), -1., 1.1, True):
            with self.assertRaises(ValueError):
                replace(plan, uncertainty=value)
        for value in (float("nan"), float("inf"), -1., True):
            with self.assertRaises(ValueError):
                replace(result.cost, wall_ms=value)
        with self.assertRaises(ValueError):
            replace(result.cost, input_tokens=True)
        self.assertIsNone(result.cost.energy_joules)

    def test_strict_fields_schema_and_duplicate_json_keys(self):
        episode, _, _ = fixtures()
        obj = json.loads(encode_record(episode))
        obj["record"]["raw_credentials"] = "not allowed"
        with self.assertRaises(ValueError):
            decode_record(json.dumps(obj))
        for text in ('{"schema":1,"schema":1}', "[]", "null", "{" + '"a":' * 1100 + '1' + '}' * 1100):
            with self.assertRaises(ValueError):
                decode_record(text)
        for version in (True, 2, "1"):
            obj = json.loads(encode_record(episode))
            obj["schema"] = version
            with self.assertRaises(ValueError):
                decode_record(json.dumps(obj))

    def test_bounded_opaque_metadata_and_references(self):
        episode, plan, result = fixtures()
        for bad in ("", "a" * 129, "one\ntwo", "Bearer payload", "https://private.example/path"):
            with self.assertRaises(ValueError):
                replace(episode, project=bad)
        with self.assertRaises(ValueError):
            replace(plan, observations=list(plan.observations))
        with self.assertRaises(ValueError):
            replace(plan, observations=plan.observations * 2)
        with self.assertRaises(ValueError):
            replace(result, rewards=result.rewards * 2)
        with self.assertRaises(ValueError):
            decode_record(" " * 65537)

    def test_unknown_statuses_effects_splits_and_digest(self):
        episode, plan, result = fixtures()
        for record, field, bad in ((episode, "split", "private-test"), (result, "status", "probably-correct"),
                                  (plan.proposed_action, "effect", "automatic"),
                                  (plan.observations[0], "digest", "abc")):
            with self.assertRaises(ValueError):
                replace(record, **{field: bad})


if __name__ == "__main__":
    unittest.main()

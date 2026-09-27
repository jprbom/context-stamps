"""Citation integrity never substitutes for independent semantic correctness."""

import json
import unittest
from dataclasses import replace
from unittest.mock import patch

from context_stamps.citations import check_citations, prepare_citations
from context_stamps.context_state import AccessScope, CanonicalNode, ContextState, TemporalScope


class CitationTests(unittest.TestCase):
    def setUp(self):
        self.state = ContextState(tenant="test", policy_revision="p1", clock=lambda: 10)
        self.node = CanonicalNode(key="guide", revision="1", tenant="test", text="Engine uses Cache. Cache stores data.",
                                  kind="OBSERVATION", temporal=TemporalScope(1, 1), roles=("reader",), provenance="fixture")
        self.state.put(self.node)
        self.scope = AccessScope("test", "alice", "p1", ("reader",))
        self.snapshot = self.state.snapshot(self.scope, at=10, known_at=10)
        self.packet = prepare_citations(self.state, self.snapshot, "What does Engine use?")

    @staticmethod
    def reply(answer="Cache", quote="Engine uses Cache.", source="s0"):
        return json.dumps(dict(answer=answer, citations=[dict(source=source, quote=quote)]))

    def test_exact_quote_bound_to_version_and_offsets(self):
        result = check_citations(self.state, self.packet, self.reply())
        self.assertEqual(result.status, "source_bound")
        self.assertEqual(result.quotes[0].source, self.node.ref)
        self.assertEqual(self.node.text[result.quotes[0].start:result.quotes[0].end], "Engine uses Cache.")
        self.assertFalse(result.semantics_verified)

    def test_irrelevant_exact_quote_does_not_certify_truth(self):
        packet = prepare_citations(self.state, self.snapshot, "Which animal flies?")
        result = check_citations(self.state, packet, self.reply())
        self.assertEqual(result.status, "source_bound")
        self.assertFalse(result.semantics_verified)

    def test_hallucinated_or_foreign_source_rejected(self):
        for reply in (self.reply(quote="Engine uses Database."), self.reply(source="s9")):
            self.assertEqual(check_citations(self.state, self.packet, reply).status, "rejected")

    def test_answer_must_occur_in_a_quote_when_required(self):
        self.assertEqual(check_citations(self.state, self.packet, self.reply(answer="Database")).reason, "answer_outside_quotes")
        packet = prepare_citations(self.state, self.snapshot, "Summarize", require_answer_span=False)
        result = check_citations(self.state, packet, self.reply(answer="A dependency exists."))
        self.assertEqual(result.status, "source_bound")
        self.assertFalse(result.semantics_verified)

    def test_duplicate_unknown_fields_and_types_rejected(self):
        candidates = ['{"answer":null,"answer":"Cache","citations":[]}',
                      '{"answer":null,"citations":[],"execute":"anything"}',
                      '{"answer":NaN,"citations":[]}', '{"answer":true,"citations":[]}',
                      '{"answer":"Cache","citations":{}}',
                      '{"answer":"Cache","citations":[{"source":"s0","quote":"Cache","extra":0}]}',
                      '['*1500+']'*1500]
        for raw in candidates:
            with self.subTest(raw=raw[:80]):
                self.assertEqual(check_citations(self.state, self.packet, raw).status, "rejected")

    def test_abstention_and_contradictory_abstention(self):
        self.assertEqual(check_citations(self.state, self.packet, '{"answer":null,"citations":[]}').status, "abstained")
        self.assertEqual(check_citations(self.state, self.packet, self.reply(answer=None)).status, "rejected")

    def test_question_policy_and_snapshot_tampering_fail(self):
        for packet in (replace(self.packet, question="Changed"), replace(self.packet, require_answer_span=False),
                       replace(self.packet, snapshot=replace(self.snapshot, scope=replace(self.scope, principal="bob")))):
            self.assertEqual(check_citations(self.state, packet, self.reply()).reason, "unavailable_context")

    def test_state_change_invalidates_packet(self):
        self.state.put(replace(self.node, revision="2", text="Engine uses Disk."))
        self.assertFalse(self.packet.is_current(self.state))
        self.assertEqual(check_citations(self.state, self.packet, self.reply()).reason, "unavailable_context")

    def test_revoked_role_and_cross_state_reject(self):
        other = ContextState(tenant="test", policy_revision="p1", clock=lambda: 10)
        other.put(self.node)
        self.assertEqual(check_citations(other, self.packet, self.reply()).reason, "unavailable_context")
        self.state.set_roles("guide", ("administrator",))
        self.assertEqual(check_citations(self.state, self.packet, self.reply()).reason, "unavailable_context")

    def test_model_output_keeps_its_epistemic_kind(self):
        node = replace(self.node, revision="model", kind="MODEL_OUTPUT")
        self.state.put(node)
        snapshot = self.state.snapshot(self.scope, at=10, known_at=10)
        packet = prepare_citations(self.state, snapshot, "Question", refs=(node.ref,))
        result = check_citations(self.state, packet, self.reply())
        self.assertEqual(result.quotes[0].kind, "MODEL_OUTPUT")
        self.assertFalse(result.semantics_verified)

    def test_authority_rechecked_after_inspecting_quotes(self):
        with patch.object(type(self.packet), "is_current", side_effect=[True, False]):
            result = check_citations(self.state, self.packet, self.reply())
        self.assertEqual(result.reason, "unavailable_context")

    def test_only_selected_sources_can_be_cited(self):
        other = replace(self.node, key="private", revision="2", text="A second source.")
        self.state.put(other)
        snapshot = self.state.snapshot(self.scope, at=10, known_at=10)
        packet = prepare_citations(self.state, snapshot, "Question", refs=(self.node.ref,))
        self.assertNotIn(other.text, packet.payload)
        self.assertEqual(check_citations(self.state, packet, self.reply(source="s1")).status, "rejected")

    def test_utf8_offsets_and_repeated_span_use_first_exact_occurrence(self):
        node = replace(self.node, revision="utf8", text="café café")
        self.state.put(node)
        snapshot = self.state.snapshot(self.scope, at=10, known_at=10)
        packet = prepare_citations(self.state, snapshot, "Question", refs=(node.ref,))
        result = check_citations(self.state, packet, self.reply(answer="café", quote="café"))
        self.assertEqual((result.quotes[0].start, result.quotes[0].end), (0, 4))

    def test_duplicate_and_excess_citations_and_input_limits(self):
        proposal = json.loads(self.reply())
        proposal["citations"] *= 2
        self.assertEqual(check_citations(self.state, self.packet, json.dumps(proposal)).status, "rejected")
        proposal["citations"] *= 5
        self.assertEqual(check_citations(self.state, self.packet, json.dumps(proposal)).status, "rejected")
        with self.assertRaises(ValueError):
            check_citations(self.state, self.packet, "a"*32769)
        with self.assertRaises(ValueError):
            prepare_citations(self.state, self.snapshot, "")


if __name__ == "__main__":
    unittest.main()

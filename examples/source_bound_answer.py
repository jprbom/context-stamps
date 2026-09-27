"""Offline source binding for a proposed local model answer.

Copyright (c) 2026 Prashant Jagtap. MIT License.
This fixture invokes no model and makes no answer-quality claim.
"""

import json
from dataclasses import replace

from context_stamps.citations import check_citations, prepare_citations
from context_stamps.context_state import AccessScope, CanonicalNode, ContextState, TemporalScope

state = ContextState(tenant="demo", policy_revision="policy-1", clock=lambda: 10)
guide = CanonicalNode(key="cache-guide", revision="v1", tenant="demo",
                      text="Birch cache uses port 7402.", kind="OBSERVATION",
                      temporal=TemporalScope(1, 1), roles=("reader",), provenance="authored fixture")
state.put(guide)
scope = AccessScope("demo", "local-user", "policy-1", ("reader",))
snapshot = state.snapshot(scope, at=10, known_at=10)
packet = prepare_citations(state, snapshot, "Which port does Birch cache use?")

# In an application, send packet.payload as untrusted data to a local model.
# This authored response is a fixture, not a measured model output.
reply = json.dumps(dict(answer="7402", citations=[dict(source="s0", quote=guide.text)]))
result = check_citations(state, packet, reply)
assert result.status == "source_bound" and not result.semantics_verified
print(json.dumps(dict(status=result.status, answer=result.answer, semantics_verified=result.semantics_verified)))

# Verified outcomes or a separate domain verifier must assess correctness.
# A changed source invalidates the old question/source/policy binding.
state.put(replace(guide, revision="v2", text="Birch cache now uses port 7502."))
assert not packet.is_current(state)
stale = check_citations(state, packet, reply)
assert stale.reason == "unavailable_context"
print(json.dumps(dict(status=stale.status, reason=stale.reason)))

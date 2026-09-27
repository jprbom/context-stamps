"""Replay quote checks for every retained fictional model output, without a GPU."""

import hashlib
import json
from pathlib import Path

from context_stamps.citations import check_citations, prepare_citations
from context_stamps.context_state import AccessScope, CanonicalNode, ContextState, TemporalScope

ROOT = Path(__file__).resolve().parents[1]


def main():
    folder = ROOT/"evidence/citations-v1"
    manifest = json.loads((folder/"manifest.json").read_bytes())
    for name, expected in manifest["files"].items():
        if hashlib.sha256((folder/name).read_bytes()).hexdigest() != expected:
            raise ValueError("citation evidence bytes changed")
    payload = json.loads((folder/"calls.json").read_bytes())
    count = 0
    for run in payload["runs"]:
        seen = set()
        for row in run["rows"]:
            identity = row["case"], row["mode"]
            if identity in seen:
                raise ValueError("duplicate case/treatment")
            seen.add(identity)
            fixture = payload["cases"][row["case"]]
            if row["input_tokens"] != row["expected_prompt_tokens"] or not row["done"]:
                raise ValueError("native count or completion failure")
            if json.loads(row["raw_reply"]) != row["parsed"]:
                raise ValueError("stored model reply and parsed output differ")
            if row["mode"] == "cited":
                state = ContextState(tenant="fixture", policy_revision="p1", clock=lambda: 1)
                for index, text in enumerate(fixture["texts"]):
                    state.put(CanonicalNode(key=f"doc{index}", revision="1", tenant="fixture", text=text,
                                            kind="OBSERVATION", temporal=TemporalScope(1, 1), roles=("reader",), provenance="fiction"))
                snap = state.snapshot(AccessScope("fixture", "tester", "p1", ("reader",)), at=1, known_at=1)
                packet = prepare_citations(state, snap, fixture["question"])
                actual = check_citations(state, packet, row["raw_reply"]).to_dict()
                # State seals are process-local; replay semantic-independent checks.
                for name in ("status", "reason", "answer", "quotes", "semantics_verified", "reply_digest"):
                    if json.loads(json.dumps(actual[name])) != row["check"][name]:
                        raise ValueError(f"citation replay changed: {run['name']} {identity} {name}")
            count += 1
        if len(seen) != 12:
            raise ValueError("all six direct/cited cases required")
    if count != payload["scored_calls"] or count != 48:
        raise ValueError("complete attempt inventory required")
    print(json.dumps(dict(calls=count, citation_checks=24, native_token_parity=True,
                          quality_or_safety_certificate=False)))


if __name__ == "__main__":
    main()

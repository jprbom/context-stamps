"""Offline cross-functional handoff with explicitly supplied facets and relationships."""

from context_stamps.spherical import SphericalStamp
from context_stamps.workflow import ContextGraph, ContextNode
from stamps import Family, HashingEncoder

encoder = HashingEncoder(64)
families = {name: Family(encoder.identity, 64, bits=64, seed=i + 17)
            for i, name in enumerate(("content", "entity", "intent", "task"))}


def stamp(**facets):
    return SphericalStamp.encode({name: encoder.encode(text) for name, text in facets.items()}, families)


graph = ContextGraph()
root = stamp(content="Revise API timeout", entity="worker_alpha", intent="implementation", task="timeout")
graph.put(ContextNode("implementation", "Set timeout_ms to 250.", "v1", frozenset({"engineer"}), root))
graph.put(ContextNode("contract", "The API timeout is measured in milliseconds.", "v1",
                      frozenset({"engineer", "reviewer"})))
graph.link("implementation", "contract", "depends_on", provenance="reviewed API contract")
revisions = {"implementation": "v1", "contract": "v1"}
hits = graph.retrieve(root, role="engineer", revisions=revisions, limit=1)
packet = graph.handoff([hits[0]["source"]], role="engineer", revisions=revisions, budget_bytes=2048)
print("Per-facet matches:", hits[0]["facets"])
print("Handoff:", packet.status, packet.text)
print("Affected by contract change:", graph.affected(["contract"]))
print("Serialized stamp bytes:", len(root.to_payload().encode("utf-8")))

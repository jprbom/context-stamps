"""Offline relation-packet example. Copyright (c) 2026 Prashant Jagtap."""

from context_stamps.observation_packets import observation_packets, plan_pages
from context_stamps.trajectory import ObservedEpisode, ObservedStep

source = """RootWebArea 'Inventory'
    table 'Available stock'
        row ''
            columnheader 'Item'
            columnheader 'Quantity'
            columnheader 'Location'"""
episode = ObservedEpisode("inventory-session", "Inspect inventory", (
    ObservedStep(0, source, location="local/inventory"),
))
packets = tuple(observation_packets(episode))
assert len(packets) == 1 and packets[0].relation == "ordered_members"
print(packets[0].text)
print("Source references:", packets[0].occurrences)
plan = plan_pages("Inspect Inventory", tuple(p.page for p in packets))
assert plan.pages == (("'Inventory'",),) and not plan.fallback
node = packets[0].canonical_node(tenant="local", roles=("owner",), observed_at=1)
assert node.kind == "DERIVED_RESULT" and not node.claims
print("Page affinity is not authorization or a sufficiency certificate.")

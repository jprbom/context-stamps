"""Offline structural-memory example. Copyright (c) 2026 Prashant Jagtap."""

from context_stamps.structured_memory import structured_views
from context_stamps.trajectory import ObservedEpisode, ObservedStep

before = """RootWebArea 'Project settings'
    [10] region 'Build'
        [11] combobox 'Test mode' value='Full'
            [12] option 'Full', selected=True
            [13] option 'Quick', selected=False"""

episode = ObservedEpisode("local-example", "Inspect build settings", (
    ObservedStep(0, before, location="local/project/settings"),
    ObservedStep(1, before.replace("[11]", "[21]"), location="local/project/settings"),
))
views = tuple(structured_views(episode))
assert len(views) == 1  # Same recorded content; only a transient instance ID changed.
assert len(views[0].occurrences) == 2
node = views[0].canonical_node(tenant="local", roles=("owner",), observed_at=1)
assert node.kind == "DERIVED_RESULT" and not node.claims
print(views[0].text)
print("Source occurrence count:", len(views[0].occurrences))
print("Keep the original episode locally; this view is not a verified policy or executable action.")

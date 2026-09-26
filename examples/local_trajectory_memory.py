"""Build local observed memory without a model, network request or paid service."""

from context_stamps.context_state import AccessScope, ContextState
from context_stamps.trajectory import ObservedEpisode, ObservedStep, trajectory_views

episode = ObservedEpisode("settings-session", "Change the notification preference", (
    ObservedStep(0, "Notifications: enabled", "open(settings)", "local/settings"),
    ObservedStep(1, "Notifications: disabled", "toggle(notifications)", "local/settings"),
))
state = ContextState(tenant="local", policy_revision="v1", clock=lambda: 10)
views = tuple(trajectory_views(episode))
for view in views:
    state.put(view.canonical_node(tenant="local", roles=("owner",), observed_at=10))
snapshot = state.snapshot(AccessScope("local", "user", "v1", ("owner",)), at=10, known_at=10)
print(f"Episode revision: {episode.revision}")
print(f"Authorized local views: {len(snapshot.records)}")
print(next(v.text for v in views if v.channel == "change"))
print("These observations do not establish causality or successful task completion.")

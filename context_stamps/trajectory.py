"""Local trajectory views with exact source binding and explicit epistemic types.

Copyright (c) 2026 Prashant Jagtap. MIT License.
These are lossy retrieval views, not sufficient-context or causal certificates.
No model output, reported task success or screenshot path becomes verified fact.
"""

import difflib
import hashlib
import json
from dataclasses import asdict, dataclass

from .context_state import CanonicalNode, TemporalScope
from .security import bounded_text, identifier

ADAPTER_REVISION = "trajectory-views-v1"


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    separators=(",", ":")).encode()).hexdigest()


@dataclass(frozen=True)
class ObservedStep:
    index: int
    observation: str
    action: str = ""
    location: str = ""

    def __post_init__(self):
        if type(self.index) is not int or not 0 <= self.index < 10000:
            raise ValueError("bounded nonnegative step index required")
        bounded_text(self.observation, 2 * 1024 * 1024)
        bounded_text(self.action, 16384)
        bounded_text(self.location, 4096)


@dataclass(frozen=True)
class ObservedEpisode:
    key: str
    goal: str
    steps: tuple[ObservedStep, ...]

    def __post_init__(self):
        identifier(self.key)
        bounded_text(self.goal, 16384)
        if (type(self.steps) is not tuple or not 1 <= len(self.steps) <= 512
                or any(type(s) is not ObservedStep for s in self.steps)):
            raise ValueError("one to 512 immutable observed steps required")
        indices = tuple(s.index for s in self.steps)
        if indices != tuple(sorted(set(indices))):
            raise ValueError("strictly increasing unique step indices required")
        if sum(len(s.observation.encode()) for s in self.steps) > 16 * 1024 * 1024:
            raise ValueError("episode observations exceed 16 MiB")

    @property
    def revision(self):
        return _digest(asdict(self))


@dataclass(frozen=True)
class TraceFragment:
    key: str
    episode: str
    source_revision: str
    channel: str
    steps: tuple[int, ...]
    text: str

    def __post_init__(self):
        identifier(self.key)
        identifier(self.episode)
        if (len(self.source_revision) != 64
                or any(c not in "0123456789abcdef" for c in self.source_revision)):
            raise ValueError("exact source SHA-256 required")
        if self.channel not in ("state", "change", "path"):
            raise ValueError("unknown trajectory view")
        if (type(self.steps) is not tuple or not self.steps or len(self.steps) > 512
                or any(type(i) is not int or not 0 <= i < 10000 for i in self.steps)
                or self.steps != tuple(sorted(set(self.steps)))):
            raise ValueError("ordered source step indices required")
        bounded_text(self.text)

    def canonical_node(self, *, tenant, roles, observed_at):
        """Host supplies view-ingestion time and ACL.

        This binds the *view* and the external episode revision. The host must
        retain and authorize that episode separately; this is not a byte archive.
        TemporalScope describes ingestion of this view, including its event_time
        default. It does not assign wall-clock time to the source steps.
        """
        return CanonicalNode(
            key=self.key, revision=_digest(asdict(self)), tenant=tenant, text=self.text,
            kind="OBSERVATION" if self.channel == "state" else "DERIVED_RESULT",
            temporal=TemporalScope(valid_from=observed_at, observed_at=observed_at),
            roles=roles, provenance=f"{ADAPTER_REVISION}:{self.source_revision}",
        )


def trajectory_views(episode, *, chunk_chars=4000):
    """Yield bounded exact-text chunks and derived adjacent-state/path views.

    Recorded actions are associated with their recorded step, without assuming
    they caused the next observation. Sequence differences preserve old/new
    direction but do not establish a real-world change or inter-episode time.
    """
    if type(episode) is not ObservedEpisode:
        raise ValueError("typed observed episode required")
    if type(chunk_chars) is not int or not 256 <= chunk_chars <= 8192:
        raise ValueError("chunk size must be between 256 and 8192 characters")
    revision = episode.revision
    prefix = hashlib.sha256(episode.key.encode()).hexdigest()[:24]
    sequence = 0

    def fragments(channel, indices, body):
        nonlocal sequence
        header = f"Episode {episode.key}; steps {','.join(map(str, indices))}; view {channel}\n"
        for start in range(0, len(body), chunk_chars):
            sequence += 1
            yield TraceFragment(f"trace-{prefix}-{sequence}", episode.key, revision, channel,
                                indices, header + body[start:start + chunk_chars])

    previous = None
    for step in episode.steps:
        body = (f"Recorded goal: {episode.goal}\nLocation: {step.location}\n"
                f"Recorded action at this step: {step.action}\nObservation:\n{step.observation}")
        yield from fragments("state", (step.index,), body)
        if previous is not None:
            old, new = previous.observation.splitlines(), step.observation.splitlines()
            diff = difflib.SequenceMatcher(None, old, new, autojunk=True)
            changes = []
            for tag, a, b, c, d in diff.get_opcodes():
                if tag == "equal":
                    continue
                changes.append("Previous observation excerpt:\n" + "\n".join(old[max(0, a-2):min(len(old), b+2)])
                               + "\nCurrent observation excerpt:\n" + "\n".join(new[max(0, c-2):min(len(new), d+2)]))
            if changes:
                body = (f"Recorded goal: {episode.goal}\nAdjacent observations, not a causal assertion.\n"
                        f"Previous recorded action: {previous.action}\nCurrent recorded action: {step.action}\n"
                        + "\n\n".join(changes))
                yield from fragments("change", (previous.index, step.index), body)
        previous = step
    path = f"Recorded goal: {episode.goal}\nRecorded step/action order (not a verified successful procedure):\n"
    path += "\n".join(f"{s.index}: {s.action} | {s.location}" for s in episode.steps)
    yield from fragments("path", tuple(s.index for s in episode.steps), path)

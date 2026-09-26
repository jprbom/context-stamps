"""Source-bound, indentation-aware views of local accessibility observations.

Copyright (c) 2026 Prashant Jagtap. MIT License.
Views preserve recorded structure, not verified UI semantics or causality.
The original episode remains external and must have its own retention and ACL.
"""

import re
from dataclasses import asdict, dataclass

from .context_state import CanonicalNode, TemporalScope
from .security import bounded_text, identifier
from .trajectory import ObservedEpisode, _digest

REVISION = "structured-observation-v1"
_HANDLE = re.compile(r"^\[[A-Za-z0-9_-]{1,128}\] ")


@dataclass(frozen=True, order=True)
class ObservationSpan:
    """One-based inclusive source lines; ancestor lines precede the body."""

    step: int
    first: int
    last: int
    ancestors: tuple[int, ...] = ()

    def __post_init__(self):
        if (type(self.step) is not int or not 0 <= self.step < 10000
                or type(self.first) is not int or type(self.last) is not int
                or not 1 <= self.first <= self.last <= 65536
                or type(self.ancestors) is not tuple or len(self.ancestors) > 96
                or any(type(x) is not int or not 1 <= x < self.first for x in self.ancestors)
                or self.ancestors != tuple(sorted(set(self.ancestors)))):
            raise ValueError("bounded, ordered source lines required")


@dataclass(frozen=True)
class StructureFragment:
    episode: str
    source_revision: str
    location: str
    body: str
    occurrences: tuple[ObservationSpan, ...]

    def __post_init__(self):
        identifier(self.episode)
        if (type(self.source_revision) is not str or len(self.source_revision) != 64
                or any(c not in "0123456789abcdef" for c in self.source_revision)):
            raise ValueError("source SHA-256 required")
        bounded_text(self.location, 4096)
        bounded_text(self.body, 49152)
        if (type(self.occurrences) is not tuple or not 1 <= len(self.occurrences) <= 4096
                or any(type(x) is not ObservationSpan for x in self.occurrences)
                or self.occurrences != tuple(sorted(set(self.occurrences)))):
            raise ValueError("bounded unique ordered source occurrences required")

    @property
    def key(self):
        return "structure-" + _digest(asdict(self))

    @property
    def text(self):
        # Full occurrence mapping is available on this object. Keep reader text
        # bounded even when the same view appears in many observed states.
        steps = sorted({s.step for s in self.occurrences})
        return (f"Episode {self.episode}; observed steps {','.join(map(str, steps))}\n"
                f"Location: {self.location}\nRecorded structure (instance handles omitted):\n{self.body}")

    def canonical_node(self, *, tenant, roles, observed_at):
        return CanonicalNode(
            key=self.key, revision=_digest(asdict(self)), tenant=tenant, text=self.text,
            kind="DERIVED_RESULT", temporal=TemporalScope(observed_at, observed_at), roles=roles,
            provenance=f"{REVISION}:{self.source_revision}",
        )


def _tree(observation, chunk_chars):
    raw = observation.splitlines()
    if len(raw) > 65536:
        raise ValueError("observation exceeds 65536 lines; use raw fallback")
    lines, parents, children, stack = [], [], {}, []
    for number, line in enumerate(raw, 1):
        if not line.strip():
            continue
        prefix = line[:len(line)-len(line.lstrip(" \t"))]
        depth = len(prefix.expandtabs(4))
        text = _HANDLE.sub("", line[len(prefix):], count=1)
        if len(text) > chunk_chars:
            raise ValueError("single source line exceeds view budget; use raw fallback")
        while stack and stack[-1][0] >= depth:
            stack.pop()
        if len(stack) >= 96:
            raise ValueError("observation nesting exceeds 96; use raw fallback")
        parent = stack[-1][1] if stack else -1
        i = len(lines)
        lines.append((number, text))
        parents.append(parent)
        children.setdefault(parent, []).append(i)
        stack.append((depth, i))
    ends, sizes = [i+1 for i in range(len(lines))], [len(t)+1 for _, t in lines]
    for i in range(len(lines)-1, -1, -1):
        if parents[i] >= 0:
            ends[parents[i]] = max(ends[parents[i]], ends[i])
            sizes[parents[i]] += sizes[i]
    return lines, parents, children, ends, sizes


def structured_views(episode, *, chunk_chars=2400):
    """Group whole subtrees/sibling blocks and repeat their full ancestry.

    Only leading instance handles and indentation are projected. All remaining
    source text, including values and state flags, is retained. Equal bodies at
    the same location share a view with *all* source occurrences, not a fabricated
    validity interval. Actions are deliberately not interpreted. Oversized input
    raises ValueError so the host can take an explicit raw fallback.
    """
    if type(episode) is not ObservedEpisode:
        raise ValueError("typed episode required")
    if type(chunk_chars) is not int or not 256 <= chunk_chars <= 8192:
        raise ValueError("chunk budget must be 256..8192 characters")
    revision = episode.revision
    groups = {}
    for step in episode.steps:
        lines, parents, children, ends, sizes = _tree(step.observation, chunk_chars)
        pending = [-1]
        blocks = []
        while pending:
            parent = pending.pop()
            group, size = [], 0
            for child in children.get(parent, []):
                if sizes[child] > chunk_chars and children.get(child):
                    if group:
                        blocks.append((group[0], ends[group[-1]]))
                        group, size = [], 0
                    pending.append(child)
                else:
                    if size + sizes[child] > chunk_chars and group:
                        blocks.append((group[0], ends[group[-1]]))
                        group, size = [], 0
                    group.append(child)
                    size += sizes[child]
            if group:
                blocks.append((group[0], ends[group[-1]]))
        for first, end in sorted(blocks):
            ancestors, parent = [], parents[first]
            while parent >= 0:
                ancestors.append(parent)
                parent = parents[parent]
            ancestors.reverse()
            head = "\n".join("  "*i+lines[a][1] for i, a in enumerate(ancestors))
            # Preserve every body node's relative depth, not its volatile IDs.
            depths = {}
            rows = []
            for i in range(first, end):
                depth = depths.get(parents[i], len(ancestors)-1)+1
                depths[i] = depth
                rows.append("  "*depth+lines[i][1])
            body = (head+"\n" if head else "") + "\n".join(rows)
            if len(body.encode()) > 49152:
                raise ValueError("ancestry and subtree exceed view byte budget; use raw fallback")
            span = ObservationSpan(step.index, lines[first][0], lines[end-1][0],
                                   tuple(lines[i][0] for i in ancestors))
            bucket = groups.setdefault((step.location, body), [])
            if len(bucket) >= 4096:
                raise ValueError("too many identical occurrences; use raw fallback")
            bucket.append(span)
    # Nothing is yielded until all bounds pass: fallback cannot mix partial views.
    for (location, body), spans in groups.items():
        yield StructureFragment(episode.key, revision, location, body, tuple(sorted(spans)))

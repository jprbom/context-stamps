"""Bounded page hints and compact, source-bound observation relations.

Copyright (c) 2026 Prashant Jagtap. MIT License.
These are lossy source projections, not verified facts, visual-layout claims,
authorization scopes or successful procedures. Source text is never executed.
"""

import collections
import math
import re
from dataclasses import asdict, dataclass

from .context_state import CanonicalNode, TemporalScope
from .security import bounded_text, identifier
from .structured_memory import _tree
from .trajectory import ObservedEpisode, _digest

REVISION = "observation-packets-v2"
_LABEL = re.compile(r'''^([A-Za-z][A-Za-z0-9_]{0,63})\s+('(?:[^'\\]|\\.)*'|"(?:[^"\\]|\\.)*")(.*)$''')
_SEQUENCES = {"row": {"columnheader", "rowheader", "gridcell", "cell"},
              "combobox": {"option"}, "listbox": {"option"},
              "tablist": {"tab"}, "menu": {"menuitem", "menuitemcheckbox", "menuitemradio"}}
_WRAPPERS = {"RootWebArea", "generic", "main", "rowgroup", "list", "listitem", "group", "Section", "region", "navigation", "form", "table", "grid", "LayoutTable", "Iframe", "banner", "contentinfo"}


def label(line):
    """Return role, quoted label, trailing attributes; do not evaluate literals."""
    bounded_text(line, 32768)
    match = _LABEL.fullmatch(line.strip())
    if match is None:
        return "opaque", "", line.strip()
    return match.groups()


def page_hint(body):
    bounded_text(body)
    return tuple(name for line in body.splitlines() for role, name, _ in (label(line),)
                 if role == "RootWebArea")


def anchor_words(text):
    # English plural normalization is a retrieval heuristic, never identity.
    return frozenset(w[:-1] if len(w) > 4 and w.endswith("s") and not w.endswith("ss") else w
                     for w in re.findall(r"\b[a-zA-Z][a-zA-Z0-9_]*\b", text.casefold()))


@dataclass(frozen=True)
class PagePlan:
    pages: tuple[tuple[str, ...], ...]
    anchor_terms: tuple[str, ...]
    scores: tuple[float, ...]
    fallback: bool


def plan_pages(question, pages, *, relative_floor=0.5):
    """Corpus-IDF title affinity with an explicit no-anchor fallback.

    Caller supplies only authorized titles. Common title terms (>10% of pages)
    are excluded from anchors. Keep titles within a fixed relative score floor;
    these uncalibrated scores are not probabilities or sufficiency estimates.
    """
    bounded_text(question, 16384)
    if type(pages) is not tuple or not 1 <= len(pages) <= 8192:
        raise ValueError("one to 8192 authorized page paths required")
    if type(relative_floor) not in (int, float) or not math.isfinite(relative_floor) or not 0 < relative_floor <= 1:
        raise ValueError("relative score floor must be in (0,1]")
    for path in pages:
        if type(path) is not tuple or not 1 <= len(path) <= 96:
            raise ValueError("bounded immutable page path required")
        for text in path:
            bounded_text(text, 8192)
    unique = tuple(sorted(set(pages)))
    tokens = [anchor_words(" ".join(path)) for path in unique]
    df = collections.Counter(t for ts in tokens for t in ts)
    query = anchor_words(question)
    anchors = {t for t in query if t in df and df[t] <= max(1, len(unique)*0.1)}
    weights = {t: math.log((len(unique)+1)/(df[t]+1))+1 for t in anchors}
    scored = [(sum(weights[t] for t in ts & anchors)/math.sqrt(max(1, len(ts))), path)
              for path, ts in zip(unique, tokens)]
    scored.sort(key=lambda pair: (-pair[0], pair[1]))
    best = scored[0][0]
    if best <= 0:
        return PagePlan((), tuple(sorted(anchors)), (), True)
    kept = [(score, path) for score, path in scored if score >= best*relative_floor]
    return PagePlan(tuple(p for _, p in kept), tuple(sorted(anchors)), tuple(s for s, _ in kept), False)


@dataclass(frozen=True, order=True)
class RelationOccurrence:
    step: int
    lines: tuple[int, ...]

    def __post_init__(self):
        if (type(self.step) is not int or not 0 <= self.step < 10000
                or type(self.lines) is not tuple or not 1 <= len(self.lines) <= 4096
                or any(type(i) is not int or not 1 <= i <= 65536 for i in self.lines)
                or self.lines != tuple(sorted(set(self.lines)))):
            raise ValueError("bounded ordered source line identities required")


@dataclass(frozen=True)
class ObservationPacket:
    episode: str
    source_revision: str
    page: tuple[str, ...]
    location: str
    relation: str
    body: str
    occurrences: tuple[RelationOccurrence, ...]

    def __post_init__(self):
        identifier(self.episode)
        if (type(self.source_revision) is not str or not re.fullmatch(r"[0-9a-f]{64}", self.source_revision)):
            raise ValueError("source SHA-256 required")
        if type(self.page) is not tuple or len(self.page) > 96:
            raise ValueError("bounded immutable page path required")
        for name in self.page:
            bounded_text(name, 8192)
        if sum(len(name.encode()) for name in self.page) > 16384:
            raise ValueError("page path too large")
        bounded_text(self.location, 4096)
        bounded_text(self.body, 16384)
        if self.relation not in ("element", "ordered_members"):
            raise ValueError("unknown observed relation")
        if (type(self.occurrences) is not tuple or not 1 <= len(self.occurrences) <= 4096
                or any(type(x) is not RelationOccurrence for x in self.occurrences)
                or self.occurrences != tuple(sorted(set(self.occurrences)))
                or len({x.step for x in self.occurrences}) > 512):
            raise ValueError("bounded unique observed occurrences required")

    @property
    def key(self):
        return "observation-" + _digest(asdict(self))

    @property
    def text(self):
        return (f"Episode {self.episode}; recorded steps {','.join(map(str, sorted({o.step for o in self.occurrences})))}\n"
                + "Page path: " + " / ".join(self.page) + "\n" + self.body)

    def canonical_node(self, *, tenant, roles, observed_at):
        return CanonicalNode(key=self.key, revision=_digest(asdict(self)), tenant=tenant, text=self.text,
                             kind="DERIVED_RESULT", roles=roles, temporal=TemporalScope(observed_at, observed_at),
                             provenance=f"{REVISION}:{self.source_revision}")


def observation_packets(episode):
    """Project named elements and recorded ordered cell/option/menu/tab members.

    Empty wrappers and repeated StaticText copies are omitted. Full source and
    line bindings remain external. Ordered cells establish source order, not
    visual position, colspans, computed label associations or factual truth.
    Nothing is yielded until all input and output bounds pass.
    """
    if type(episode) is not ObservedEpisode:
        raise ValueError("typed observed episode required")
    revision, groups = episode.revision, {}
    for step in episode.steps:
        lines, parents, _, ends, _ = _tree(step.observation, 8192)
        parsed = [label(text) for _, text in lines]
        paths, scopes = [], []
        for i, (role, name, _) in enumerate(parsed):
            parent = parents[i]
            path = paths[parent] if parent >= 0 else ()
            scope = scopes[parent] if parent >= 0 else ()
            paths.append(path + ((name,) if role == "RootWebArea" else ()))
            scopes.append(scope + ((i,) if role == "RootWebArea" else ()))
        consumed = set()
        records = []

        def owner_of(index):
            parent = parents[index]
            while parent >= 0 and parsed[parent][0] != "RootWebArea":
                role, name, _ = parsed[parent]
                if role in {"table", "grid", "region", "Section", "form", "tablist", "menu", "row"} or name not in ("", "''", '""'):
                    return (parent,)
                parent = parents[parent]
            return ()

        def owner_text(owner):
            return (f"Recorded ancestor at source line {lines[owner[0]][0]}: {lines[owner[0]][1]}\n"
                    if owner else "")

        for i, (role, name, attrs) in enumerate(parsed):
            if role not in _SEQUENCES:
                continue
            members = []
            for j in range(i+1, ends[i]):
                if parsed[j][0] not in _SEQUENCES[role]:
                    continue
                parent = parents[j]
                while parent >= 0 and parsed[parent][0] != role:
                    parent = parents[parent]
                if parent == i:
                    members.append(j)
            if not members or len(members) > 1024:
                continue
            # Name may already concatenate every cell; members are the explicit
            # ordered representation. Keep control names and all raw attributes.
            heading = role + (" " + name if role != "row" else "") + attrs
            owner = owner_of(i)
            body = owner_text(owner) + heading + "\nObserved members in source order:\n" + "\n".join(lines[j][1] for j in members)
            if len(body.encode()) > 16384:
                continue  # Elements below remain; no partial ordered relation.
            refs = tuple(sorted({lines[k][0] for k in (*scopes[i], *owner, i, *members)}))
            records.append((i, "ordered_members", body, refs))
            consumed.update((i, *members))
        for i, (role, name, attrs) in enumerate(parsed):
            if i in consumed:
                continue
            if role in _WRAPPERS or name in ("", "''", '""'):
                continue
            parent = parents[i]
            duplicate = False
            if role in {"StaticText", "InlineTextBox"}:
                while parent >= 0:
                    if parsed[parent][1] not in ("", "''", '""'):
                        duplicate = parsed[parent][1] == name
                        break
                    parent = parents[parent]
            if duplicate:
                continue
            # The named/structural ancestor is not inferred to be a field label.
            owner = owner_of(i)
            body = owner_text(owner) + lines[i][1]
            refs = tuple(sorted({lines[k][0] for k in (*scopes[i], *owner, i)}))
            if len(body.encode()) > 16384:
                raise ValueError("element/parent exceeds packet bound")
            records.append((i, "element", body, refs))
        for i, relation, body, refs in records:
            key = (paths[i], step.location, relation, body)
            bucket = groups.setdefault(key, [])
            if len(bucket) >= 4096:
                raise ValueError("too many identical occurrences")
            bucket.append(RelationOccurrence(step.index, refs))
    result = [ObservationPacket(episode.key, revision, page, location, relation, body, tuple(sorted(spans)))
              for (page, location, relation, body), spans in groups.items()]
    yield from result

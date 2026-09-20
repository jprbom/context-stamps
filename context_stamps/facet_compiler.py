"""Bounded, inspectable facet extraction for context routing.

This baseline deliberately uses deterministic rules.  Every emitted value carries
its provenance; absence is represented by omission, never an invented facet.
Applications may replace individual extractors while preserving this contract.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from types import MappingProxyType

from .security import bounded_text

_WORD = re.compile(r"[A-Za-z][A-Za-z0-9_.:/-]{2,63}")
_VERSION = re.compile(r"\b(?:v?\d+(?:\.\d+){1,3}|20\d{2}-\d{2}-\d{2})\b", re.I)
_ENTITY = re.compile(
    r"(?:[A-Za-z_][A-Za-z0-9_]{2,}(?:\.[A-Za-z_][A-Za-z0-9_]{1,})+|"
    r"[A-Za-z_][A-Za-z0-9_]{2,}\([^\n()]{0,48}\)|"
    r"[A-Za-z0-9_.-]+/(?:[A-Za-z0-9_.-]+/)*[A-Za-z0-9_.-]+)"
)
_INTENTS = (
    "add", "analyse", "analyze", "build", "compare", "debug", "delete", "explain",
    "fix", "implement", "measure", "remove", "review", "route", "test", "train", "update",
)
_RELATIONS = (
    "blocks", "calls", "depends on", "derived from", "implements", "imports", "requires", "uses",
)
_MODALITIES = {"text", "code", "image", "audio", "video", "table"}
_META = {"authority", "policy", "modality", "workflow"}


@dataclass(frozen=True)
class FacetEvidence:
    value: str
    source: str
    rule: str


@dataclass(frozen=True)
class CompiledFacets:
    facets: object
    evidence: object

    def __post_init__(self):
        facets, evidence = dict(self.facets), dict(self.evidence)
        if set(facets) != set(evidence):
            raise ValueError("every facet requires provenance")
        object.__setattr__(self, "facets", MappingProxyType(facets))
        object.__setattr__(self, "evidence", MappingProxyType(evidence))


class FacetCompiler:
    """Create observed facets without a model call or hidden state.

    The compiler is a safe baseline, not a semantic parser.  Metadata values are
    host assertions and are labelled as such.  Extracted values are compact
    routing descriptions and must not be treated as verified facts.
    """

    def __init__(self, *, maximum_value_bytes=512):
        if type(maximum_value_bytes) is not int or not 64 <= maximum_value_bytes <= 4096:
            raise ValueError("maximum_value_bytes must be between 64 and 4096")
        self.maximum_value_bytes = maximum_value_bytes

    def _bounded_join(self, values):
        result = []
        size = 0
        for value in values:
            encoded = value.encode("utf-8")
            extra = len(encoded) + (2 if result else 0)
            if size + extra > self.maximum_value_bytes:
                break
            result.append(value)
            size += extra
        return ", ".join(result)

    def compile(self, text, *, metadata=None):
        bounded_text(text)
        if not text.strip():
            raise ValueError("text must contain a non-whitespace token")
        metadata = {} if metadata is None else dict(metadata)
        if set(metadata) - _META:
            raise ValueError("unsupported metadata key")
        for key, value in metadata.items():
            bounded_text(value, self.maximum_value_bytes)
            if not value.strip():
                raise ValueError(f"{key} metadata must not be blank")
        if "modality" in metadata and metadata["modality"] not in _MODALITIES:
            raise ValueError("unsupported modality")

        facets = {"semantic": text}
        evidence = {"semantic": FacetEvidence(text, "input", "exact-text-v1")}

        lower = text.casefold()
        intents = [word for word in _INTENTS if re.search(rf"\b{re.escape(word)}\b", lower)]
        if intents:
            value = self._bounded_join(intents)
            facets["task"] = value
            evidence["task"] = FacetEvidence(value, "input", "intent-lexicon-v1")

        entities = sorted(set(match.group(0) for match in _ENTITY.finditer(text)), key=str.casefold)
        if entities:
            value = self._bounded_join(entities)
            facets["entity"] = value
            evidence["entity"] = FacetEvidence(value, "input", "identifier-pattern-v1")

        relations = [relation for relation in _RELATIONS if relation in lower]
        if relations:
            value = self._bounded_join(relations)
            facets["relation"] = value
            evidence["relation"] = FacetEvidence(value, "input", "relation-lexicon-v1")

        versions = sorted(set(_VERSION.findall(text)), key=str.casefold)
        if versions:
            value = self._bounded_join(versions)
            facets["temporal"] = value
            evidence["temporal"] = FacetEvidence(value, "input", "version-date-pattern-v1")

        for key, value in metadata.items():
            name = "relation" if key == "workflow" else key
            if name in facets:
                combined = self._bounded_join((facets[name], value))
                facets[name] = combined
                evidence[name] = FacetEvidence(combined, "input+host-metadata", "combined-v1")
            else:
                facets[name] = value
                evidence[name] = FacetEvidence(value, "host-metadata", "asserted-v1")
        return CompiledFacets(facets, evidence)

"""Canonical external media and extraction lineage for local context workflows.

Copyright (c) 2026 Prashant Jagtap. MIT License.
Bytes stay with a host-owned resolver. This module neither decodes media nor
executes OCR, transcription, captions, URLs, paths or source instructions.
"""

import hashlib
from dataclasses import asdict, dataclass

from .context_state import (
    CanonicalNode,
    ContextSnapshot,
    NodeRef,
    TemporalScope,
    canonical,
    digest,
    typed_tuple,
)
from .security import bounded_text, identifier

REVISION = "media-evidence-v1"
MAX_BLOB_BYTES = 64 * 1024 * 1024
MEDIA_TYPES = {
    "image/png": "image", "image/jpeg": "image", "image/webp": "image",
    "audio/wav": "audio", "audio/flac": "audio", "audio/ogg": "audio", "audio/mpeg": "audio",
    "video/mp4": "video", "video/webm": "video",
}


def _integer(value, low, high):
    if type(value) is not int or not low <= value <= high:
        raise ValueError("integer outside declared bounds")


@dataclass(frozen=True)
class MediaAsset:
    """Immutable byte identity plus host-declared media metadata.

    MIME, dimensions and duration are declarations, not codec-verified metadata.
    ``key`` is an opaque host storage key, never a path to open automatically.
    """

    key: str
    revision: str
    sha256: str
    byte_length: int
    media_type: str
    dimensions: tuple[int, int] | None = None
    duration_ms: int | None = None

    def __post_init__(self):
        identifier(self.key)
        identifier(self.revision)
        digest(self.sha256)
        _integer(self.byte_length, 1, MAX_BLOB_BYTES)
        if type(self.media_type) is not str or self.media_type not in MEDIA_TYPES:
            raise ValueError("supported declared media type required")
        if self.modality in ("image", "video") and self.dimensions is not None:
            if type(self.dimensions) is not tuple or len(self.dimensions) != 2:
                raise ValueError("declared pixel width and height required")
            for value in self.dimensions:
                _integer(value, 1, 65536)
        elif self.modality == "audio" and self.dimensions is not None:
            raise ValueError("audio cannot declare image dimensions")
        if self.modality in ("audio", "video") and self.duration_ms is not None:
            _integer(self.duration_ms, 1, 86400000)
        elif self.modality == "image" and self.duration_ms is not None:
            raise ValueError("still image cannot declare a temporal duration")

    @property
    def modality(self):
        return MEDIA_TYPES[self.media_type]

    @property
    def manifest(self):
        return canonical(dict(schema=REVISION, asset=asdict(self), metadata_verified=False))

    @classmethod
    def from_bytes(cls, data, *, key, revision, media_type, dimensions=None, duration_ms=None):
        if type(data) is not bytes or not 1 <= len(data) <= MAX_BLOB_BYTES:
            raise ValueError("bounded immutable media bytes required")
        return cls(key, revision, hashlib.sha256(data).hexdigest(), len(data), media_type, dimensions, duration_ms)

    def matches(self, data):
        return (type(data) is bytes and len(data) == self.byte_length
                and hashlib.sha256(data).hexdigest() == self.sha256)

    def node(self, *, tenant, roles, temporal, supersedes=()):
        return CanonicalNode(key=self.key, revision=self.revision, tenant=tenant, text=self.manifest,
                             kind="OBSERVATION", temporal=temporal, roles=roles, provenance=REVISION,
                             modality=self.modality, artifact_digest=self.sha256, supersedes=supersedes)


@dataclass(frozen=True)
class MediaRegion:
    """Half-open pixel rectangle and/or source-relative millisecond interval."""

    rectangle: tuple[int, int, int, int] | None = None
    interval_ms: tuple[int, int] | None = None

    def __post_init__(self):
        if self.rectangle is not None:
            if type(self.rectangle) is not tuple or len(self.rectangle) != 4:
                raise ValueError("immutable pixel rectangle required")
            for value in self.rectangle:
                _integer(value, 0, 65536)
            x0, y0, x1, y1 = self.rectangle
            if x0 >= x1 or y0 >= y1:
                raise ValueError("nonempty ordered rectangle required")
        if self.interval_ms is not None:
            if type(self.interval_ms) is not tuple or len(self.interval_ms) != 2:
                raise ValueError("immutable media interval required")
            for value in self.interval_ms:
                _integer(value, 0, 86400000)
            if self.interval_ms[0] >= self.interval_ms[1]:
                raise ValueError("nonempty ordered interval required")

    def validate_for(self, asset):
        if type(asset) is not MediaAsset:
            raise ValueError("typed media asset required")
        if self.rectangle is not None:
            if asset.dimensions is None or self.rectangle[2] > asset.dimensions[0] or self.rectangle[3] > asset.dimensions[1]:
                raise ValueError("rectangle outside declared source dimensions")
        if asset.modality == "image" and self.interval_ms is not None:
            raise ValueError("still image has no temporal interval")
        if asset.modality in ("audio", "video") and self.interval_ms is not None:
            if asset.duration_ms is None or self.interval_ms[1] > asset.duration_ms:
                raise ValueError("bounded source-relative interval required")


@dataclass(frozen=True)
class ExtractionProducer:
    name: str
    revision: str
    mode: str
    settings_sha256: str
    model_sha256: str | None = None

    def __post_init__(self):
        identifier(self.name)
        identifier(self.revision)
        digest(self.settings_sha256)
        if self.mode not in ("model", "deterministic", "human"):
            raise ValueError("explicit producer mode required")
        if self.mode == "model":
            digest(self.model_sha256)
        elif self.model_sha256 is not None:
            raise ValueError("non-model producer cannot declare a model digest")

    @property
    def evidence_kind(self):
        return {"model": "MODEL_OUTPUT", "deterministic": "DERIVED_RESULT", "human": "USER_ASSERTION"}[self.mode]


@dataclass(frozen=True)
class MediaExtraction:
    key: str
    asset_sha256: str
    text: str
    producer: ExtractionProducer
    region: MediaRegion

    def __post_init__(self):
        identifier(self.key)
        digest(self.asset_sha256)
        bounded_text(self.text, 16384)
        if not self.text.strip() or type(self.producer) is not ExtractionProducer or type(self.region) is not MediaRegion:
            raise ValueError("nonempty extraction, typed producer and region required")


@dataclass(frozen=True)
class MediaBundle:
    source: CanonicalNode
    views: tuple[CanonicalNode, ...]
    timeline: CanonicalNode | None

    @property
    def nodes(self):
        return (self.source, *self.views, *((self.timeline,) if self.timeline else ()))


def media_bundle(asset, extractions, *, tenant, roles, temporal, supersedes=()):
    """Build source, derived views and a temporal-order graph without mutation.

    All nodes inherit the same initial host ACL. Dependencies on the exact source
    ensure its later revocation, supersession or expiry hides dependent views.
    A timeline describes declared intervals; it makes no causal or truth claim.
    """
    if type(asset) is not MediaAsset or type(temporal) is not TemporalScope:
        raise ValueError("typed asset and temporal scope required")
    typed_tuple(extractions, MediaExtraction, 64)
    if not extractions or len({e.key for e in extractions}) != len(extractions) or asset.key in {e.key for e in extractions}:
        raise ValueError("one to 64 uniquely keyed extractions required")
    if sum(len(e.text.encode()) for e in extractions) > 512*1024:
        raise ValueError("extraction batch exceeds 512 KiB")
    source = asset.node(tenant=tenant, roles=roles, temporal=temporal, supersedes=supersedes)
    views = []
    for extraction in sorted(extractions, key=lambda e: e.key):
        if extraction.asset_sha256 != asset.sha256:
            raise ValueError("extraction belongs to a different byte identity")
        extraction.region.validate_for(asset)
        body = canonical(dict(schema=REVISION, source=asdict(source.ref), extraction=asdict(extraction)))
        views.append(CanonicalNode(key=extraction.key, revision=hashlib.sha256(body.encode()).hexdigest(),
                                   tenant=tenant, text=body, kind=extraction.producer.evidence_kind,
                                   temporal=temporal, roles=roles, provenance=REVISION,
                                   dependencies=(source.ref,), modality=asset.modality, artifact_digest=asset.sha256))
    timeline = None
    timed = tuple(extraction for extraction in extractions if extraction.region.interval_ms is not None)
    if asset.modality in ("audio", "video") and timed:
        refs = {view.key: view.ref for view in views}
        ordered = sorted(timed, key=lambda e: (*e.region.interval_ms, e.key))
        body = canonical(dict(schema=REVISION, source=asdict(source.ref), relation="declared_temporal_order",
                              semantics_verified=False, segments=[dict(ref=asdict(refs[e.key]), interval_ms=e.region.interval_ms)
                                                                   for e in ordered]))
        fingerprint = hashlib.sha256(body.encode()).hexdigest()
        timeline = CanonicalNode(key="media-timeline-"+fingerprint, revision=fingerprint, tenant=tenant, text=body,
                                 kind="DERIVED_RESULT", temporal=temporal, roles=roles, provenance=REVISION,
                                 dependencies=(source.ref, *(refs[e.key] for e in ordered)),
                                 modality=asset.modality, artifact_digest=asset.sha256)
        if timeline.key in {source.key, *(view.key for view in views)}:
            raise ValueError("timeline key collision")
    return MediaBundle(source, tuple(views), timeline)


@dataclass(frozen=True)
class ResolvedMedia:
    asset: MediaAsset
    source: NodeRef
    data: bytes
    snapshot: ContextSnapshot
    seal: str

    @property
    def binding(self):
        return canonical(dict(asset=asdict(self.asset), source=asdict(self.source)))

    def is_current(self, state):
        # Check bytes too: dataclass replacement is not a proof of resolution.
        return self.asset.matches(self.data) and state.verify_binding(self.snapshot, self.binding, self.seal)


def resolve_media(state, snapshot, source_ref, asset, loader):
    """Call a trusted bounded loader only for a current authorized source.

    ``loader(asset, maximum_bytes)`` owns storage authorization, reading limits,
    timeouts and isolation. It returns immutable bytes, never a path or URL.
    This API supplies no decoder or sandbox and does not hold a state lock during
    the callback. Recheck result.is_current(state) before a later model call.
    """
    if type(asset) is not MediaAsset or type(source_ref) is not NodeRef or not callable(loader):
        raise ValueError("typed identity and trusted loader required")
    if type(snapshot) is not ContextSnapshot or not state.is_current(snapshot):
        raise PermissionError("unavailable media")
    matches = [r.node for r in snapshot.records if r.node.ref == source_ref]
    if (len(matches) != 1 or matches[0].text != asset.manifest
            or matches[0].artifact_digest != asset.sha256 or matches[0].modality != asset.modality
            or matches[0].key != asset.key or matches[0].revision != asset.revision):
        raise PermissionError("unavailable media")
    selected = state.subset(snapshot, (source_ref,))
    try:
        data = loader(asset, asset.byte_length)
    except Exception:
        raise RuntimeError("media fetch failed") from None
    if not asset.matches(data):
        raise ValueError("media bytes do not match the registered identity")
    binding = canonical(dict(asset=asdict(asset), source=asdict(source_ref)))
    try:
        seal = state.seal(selected, binding)
    except ValueError:
        raise PermissionError("unavailable media") from None
    result = ResolvedMedia(asset, source_ref, data, selected, seal)
    if not result.is_current(state):
        raise PermissionError("unavailable media")
    return result

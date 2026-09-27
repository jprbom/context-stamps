"""Offline canonical media example with an authored WAV signal, not speech."""

import hashlib
import io
import json
import math
import struct
import wave

from context_stamps.context_state import AccessScope, ContextState, TemporalScope
from context_stamps.media_evidence import (
    ExtractionProducer,
    MediaAsset,
    MediaExtraction,
    MediaRegion,
    media_bundle,
    resolve_media,
)

settings = dict(sample_rate=16000, frequency_hz=440, frames=4000, amplitude=10000)
pcm = b"".join(struct.pack("<h", round(settings["amplitude"]*math.sin(2*math.pi*settings["frequency_hz"]*i/settings["sample_rate"])))
               for i in range(settings["frames"]))
container = io.BytesIO()
with wave.open(container, "wb") as output:
    output.setnchannels(1)
    output.setsampwidth(2)
    output.setframerate(settings["sample_rate"])
    output.writeframes(pcm)
raw = container.getvalue()

asset = MediaAsset.from_bytes(raw, key="fixture-tone", revision="v1", media_type="audio/wav", duration_ms=250)
producer = ExtractionProducer("authored-tone-generator", "v1", "deterministic",
                               hashlib.sha256(json.dumps(settings, sort_keys=True).encode()).hexdigest())
views = (
    MediaExtraction("first-segment", asset.sha256, "Authored tone; first interval.", producer, MediaRegion(interval_ms=(0, 125))),
    MediaExtraction("second-segment", asset.sha256, "Authored tone; overlapping interval.", producer, MediaRegion(interval_ms=(100, 250))),
)
bundle = media_bundle(asset, views, tenant="demo", roles=("reader",), temporal=TemporalScope(1, 1))
state = ContextState(tenant="demo", policy_revision="p1", clock=lambda: 10)
for node in bundle.nodes:
    state.put(node)
scope = AccessScope("demo", "local-user", "p1", ("reader",))
snapshot = state.snapshot(scope, at=10, known_at=10)

# A host owns this storage mapping. No untrusted path or URL is opened.
storage = {(asset.key, asset.revision): raw}
resolved = resolve_media(state, snapshot, bundle.source.ref, asset,
                         lambda identity, maximum: storage[(identity.key, identity.revision)])
with wave.open(io.BytesIO(resolved.data), "rb") as check:
    assert check.getnframes() == settings["frames"]
    assert check.getframerate() == settings["sample_rate"]
assert resolved.is_current(state)
print(json.dumps(dict(bytes=len(resolved.data), nodes=len(bundle.nodes),
                      extraction_kinds=[node.kind for node in bundle.views],
                      timeline=json.loads(bundle.timeline.text)["segments"], model_calls=0)))

state.set_roles(asset.key, ("private",))
assert not resolved.is_current(state)
assert not state.snapshot(scope, at=10, known_at=10).records
print(json.dumps(dict(visible_after_source_revocation=0, old_resolution_current=False)))

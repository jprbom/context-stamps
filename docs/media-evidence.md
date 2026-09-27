# External media in the canonical context graph

By Prashant Jagtap. MIT-licensed implementation.

`context_stamps.media_evidence` gives image, audio and video sources the same
version, permission and dependency contract used by textual context. Source bytes
remain external. OCR, captions, transcripts and segment descriptions become
separate derived nodes with exact source identity and extraction lineage.

```bash
python examples/media_context.py
python -m unittest discover -s tests -p test_media_evidence.py -v
```

The example generates a valid 250 ms WAV tone locally, represents two overlapping
intervals, resolves the exact bytes and revokes the source. It makes no model call
and does not evaluate speech recognition, image understanding or video reasoning.

## Representation

| Record | What it binds | What it does not establish |
|---|---|---|
| `MediaAsset` | Opaque host key, source revision, SHA-256 and byte length; optional declared geometry/duration | Codec validity, correct MIME type or perceptual meaning |
| `MediaRegion` | Half-open pixel rectangle and/or source-relative millisecond interval | Detected objects, causal events or inferred wall-clock timestamps |
| `ExtractionProducer` | Producer revision, settings digest, processing mode and model digest when applicable | Extractor accuracy or truthful output |
| `MediaExtraction` | Extracted text, source byte identity, producer and region | Semantic equivalence to the source |
| `MediaBundle` | Source node, dependent extracted views and optional temporal-order node | A trained multimodal model or calibrated context sufficiency |
| `ResolvedMedia` | Returned bytes matching the authorized source identity and the selected state implementation's scoped seal | Permanent authorization or a transferable bearer credential |

Unknown dimensions and duration remain unknown. Whole-asset views can be created
without inventing metadata. A spatial crop requires declared dimensions; a timed
segment requires a declared duration so bounds can be checked. Overlapping
intervals remain overlapping; the timeline orders their starts without inferring
causality or forcing a false partition.

The asset's cryptographic digest is separate from the 256-bit spherical routing
stamp. The digest establishes exact byte identity. A spherical stamp represents
supplied facets under a declared encoder/schema. Neither contains the media, and
digest equality does not grant access.

## Ingestion and local use

1. A trusted host reads a bounded asset and calls `MediaAsset.from_bytes(...)`.
   The factory hashes bytes but does not decode them. A host decoder may supply
   dimensions, MIME and duration; their accuracy remains that decoder's concern.
2. Record extracted text through `MediaExtraction`, with the exact source digest,
   producer version and settings digest. Model-based extraction requires a model
   digest. The default conceptual category for OCR or ASR model output is
   `MODEL_OUTPUT`, not verified fact.
3. Call `media_bundle(asset, extractions, tenant=..., roles=..., temporal=...)`.
   The factory builds all nodes without mutating a store. Add `bundle.nodes` in
   order to a `ContextState` or to the durable store through its existing API.
4. Compile authorized textual views through the existing context compiler. A VLM
   host can separately call `resolve_media` on the authorized source node when it
   actually needs pixels or audio/video bytes.
5. Recheck `resolved.is_current(state)` immediately before later use. Preserve
   independently verified outcomes separately for local policy or adapter
   training. An extractor's self-confidence is not a training label.

Models create `MODEL_OUTPUT` views, deterministic transforms create
`DERIVED_RESULT` views, and human-provided extraction creates `USER_ASSERTION`
views. None receives a validation revision automatically. These categories
describe provenance, not the quality of the extracted content.

All nodes receive the host's initial tenant and roles. Derived views depend on
the exact source version. Existing state filtering hides them if that source
expires at the requested query time, is superseded, is invalidated or becomes
unauthorized. Media time offsets remain relative to the source; the host supplies
the context's effective, observed and event times.

## Resolution boundary

The host supplies `loader(asset, maximum_bytes)`. Use an authorized, bounded
storage lookup, such as a pre-registered object mapping. Do not interpret an
untrusted source key as a filesystem path or unrestricted URL. This library does
not implement storage authentication, network fetching, decoding or a sandbox.
It validates the current context before calling the loader, checks the returned
immutable bytes, and rechecks the context before releasing the result.

The callback runs outside the shared state lock. If permissions change during
loading, the result is rejected. The host still owns callback timeouts, resource
limits, access checks at storage and synchronization with a later model call.
The library cannot recall bytes already returned to a host. `is_current` checks
the supplied snapshot's authority and bindings; it does not change a historical
query time into the current wall-clock time.

The same interface works with `ContextState`, `ContextStore` and an activated
`ContextWorkingSet`. In-memory seals are local to that state instance; durable
store seals can be rechecked after reopening with the same protected host key and
checkpoint. Integration tests cover this restart and revocation from another
store writer. A working set must include the view's exact source dependency; a
quota too small for that closure fails rather than loading an orphaned view.

Supported declarations are PNG/JPEG/WebP, WAV/FLAC/OGG/MP3, and MP4/WebM. A source
is limited to 64 MiB. A bundle has at most 64 extracted views, 16 KiB per extracted
text and 512 KiB total extracted text. These are API payload limits, not a bound
on the memory consumed by a host decoder or callback.

Textual token counts do not include image patches, audio frames or video tokens.
Actual model-specific multimodal accounting, resident VLM integration, trained
extractors and public multimodal task evaluation remain required. Successful
lineage and permission tests must not be reported as a multimodal quality gain.

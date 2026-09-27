"""Media lineage and authorization tests; no perceptual model score implied."""

import hashlib
import json
import secrets
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import Mock, patch

from context_stamps.context_state import AccessScope, ContextState, EvidenceRequirement, TemporalScope
from context_stamps.media_evidence import (
    ExtractionProducer,
    MediaAsset,
    MediaExtraction,
    MediaRegion,
    media_bundle,
    resolve_media,
)
from context_stamps.state_store import ContextStore
from context_stamps.working_set import ContextWorkingSet


class MediaEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.raw = b"authored media-byte fixture; not a decoded image"
        self.asset = MediaAsset.from_bytes(self.raw, key="camera-1", revision="v1", media_type="image/png", dimensions=(20, 10))
        self.producer = ExtractionProducer("local-ocr", "v1", "model", hashlib.sha256(b"settings").hexdigest(),
                                           hashlib.sha256(b"model fixture").hexdigest())
        self.view = MediaExtraction("reading-1", self.asset.sha256, "Recorded label: AB12", self.producer,
                                    MediaRegion(rectangle=(1, 2, 10, 8)))
        self.temporal = TemporalScope(1, 1)
        self.scope = AccessScope("lab", "user", "p1", ("reader",))
        self.state = ContextState(tenant="lab", policy_revision="p1", clock=lambda: 10)

    def bundle(self, asset=None, views=None):
        return media_bundle(asset or self.asset, views or (self.view,), tenant="lab", roles=("reader",), temporal=self.temporal)

    def ingest(self, bundle):
        for node in bundle.nodes:
            self.state.put(node)
        return self.state.snapshot(self.scope, at=10, known_at=10)

    def test_external_byte_identity_and_authorized_resolution(self):
        bundle = self.bundle()
        snapshot = self.ingest(bundle)
        loader = Mock(return_value=self.raw)
        result = resolve_media(self.state, snapshot, bundle.source.ref, self.asset, loader)
        loader.assert_called_once_with(self.asset, len(self.raw))
        self.assertEqual(result.data, self.raw)
        self.assertTrue(result.is_current(self.state))
        self.assertNotIn(self.raw.decode(), bundle.source.text)
        self.assertFalse(json.loads(bundle.source.text)["metadata_verified"])

    def test_model_text_stays_model_output_and_inherits_exact_source(self):
        bundle = self.bundle()
        child = bundle.views[0]
        self.assertEqual(child.kind, "MODEL_OUTPUT")
        self.assertIsNone(child.validation_revision)
        self.assertEqual(child.dependencies, (bundle.source.ref,))
        self.assertEqual(child.roles, bundle.source.roles)
        self.assertEqual(child.tenant, bundle.source.tenant)
        self.assertEqual(child.artifact_digest, self.asset.sha256)
        with self.assertRaises(ValueError):
            EvidenceRequirement("ocr", kinds=("MODEL_OUTPUT",), verified=True)

    def test_settings_weights_and_regions_change_derived_identity(self):
        original = self.bundle().views[0].ref
        changes = (replace(self.view, producer=replace(self.producer, settings_sha256="1"*64)),
                   replace(self.view, producer=replace(self.producer, model_sha256="2"*64)),
                   replace(self.view, region=MediaRegion(rectangle=(0, 0, 20, 10))))
        for changed in changes:
            self.assertNotEqual(self.bundle(views=(changed,)).views[0].ref, original)

    def test_hidden_source_never_reaches_loader(self):
        bundle = self.bundle()
        self.ingest(bundle)
        other = replace(self.scope, roles=("guest",))
        snapshot = self.state.snapshot(other, at=10, known_at=10)
        loader = Mock(return_value=self.raw)
        with self.assertRaises(PermissionError):
            resolve_media(self.state, snapshot, bundle.source.ref, self.asset, loader)
        loader.assert_not_called()

    def test_revocation_hides_source_and_dependent_extractions(self):
        bundle = self.bundle()
        snapshot = self.ingest(bundle)
        resolved = resolve_media(self.state, snapshot, bundle.source.ref, self.asset, lambda a, n: self.raw)
        self.state.set_roles(self.asset.key, ("private",))
        self.assertFalse(resolved.is_current(self.state))
        self.assertEqual(self.state.snapshot(self.scope, at=10, known_at=10).records, ())
        loader = Mock(return_value=self.raw)
        with self.assertRaises(PermissionError):
            resolve_media(self.state, snapshot, bundle.source.ref, self.asset, loader)
        loader.assert_not_called()

    def test_revocation_during_fetch_prevents_result_release(self):
        bundle = self.bundle()
        snapshot = self.ingest(bundle)
        def loader(asset, maximum):
            self.state.set_roles(asset.key, ())
            return self.raw
        with self.assertRaises(PermissionError):
            resolve_media(self.state, snapshot, bundle.source.ref, self.asset, loader)

    def test_altered_bytes_and_result_replacement_rejected(self):
        bundle = self.bundle()
        snapshot = self.ingest(bundle)
        for raw in (self.raw[:-1]+b"!", self.raw+b"extra", bytearray(self.raw), "file:///private"):
            with self.assertRaises(ValueError):
                resolve_media(self.state, snapshot, bundle.source.ref, self.asset, lambda a, n: raw)
        result = resolve_media(self.state, snapshot, bundle.source.ref, self.asset, lambda a, n: self.raw)
        self.assertFalse(replace(result, data=b"different").is_current(self.state))
        self.assertFalse(replace(result, asset=replace(self.asset, dimensions=(21, 10))).is_current(self.state))

    def test_wrong_declared_manifest_rejected_before_fetch(self):
        bundle = self.bundle()
        snapshot = self.ingest(bundle)
        loader = Mock(return_value=self.raw)
        with self.assertRaises(PermissionError):
            resolve_media(self.state, snapshot, bundle.source.ref, replace(self.asset, dimensions=(21, 10)), loader)
        loader.assert_not_called()

    def test_loader_exception_is_not_exposed(self):
        bundle = self.bundle()
        snapshot = self.ingest(bundle)
        with self.assertRaisesRegex(RuntimeError, "^media fetch failed$"):
            resolve_media(self.state, snapshot, bundle.source.ref, self.asset, Mock(side_effect=OSError("private storage details")))

    def test_audio_and_video_timelines_preserve_overlaps_and_kinds(self):
        for mime, dimensions in (("audio/wav", None), ("video/mp4", (20, 10))):
            asset = MediaAsset.from_bytes(b"temporal fixture", key="stream", revision="v1", media_type=mime,
                                          dimensions=dimensions, duration_ms=3000)
            a = MediaExtraction("later", asset.sha256, "Later segment", self.producer, MediaRegion(interval_ms=(1000, 2500)))
            b = MediaExtraction("earlier", asset.sha256, "Earlier segment", self.producer, MediaRegion(interval_ms=(0, 1500)))
            bundle = self.bundle(asset, (a, b))
            timeline = json.loads(bundle.timeline.text)
            self.assertEqual([s["interval_ms"] for s in timeline["segments"]], [[0, 1500], [1000, 2500]])
            self.assertFalse(timeline["semantics_verified"])
            self.assertEqual(len(bundle.timeline.dependencies), 3)
            self.assertEqual(bundle, self.bundle(asset, (b, a)))
            self.assertEqual(bundle.timeline.kind, "DERIVED_RESULT")

    def test_invalid_coordinates_intervals_and_cross_asset_views_fail(self):
        for region in (MediaRegion(rectangle=(0, 0, 21, 10)), MediaRegion(interval_ms=(0, 1))):
            with self.assertRaises(ValueError):
                self.bundle(views=(replace(self.view, region=region),))
        with self.assertRaises(ValueError):
            self.bundle(views=(replace(self.view, asset_sha256="0"*64),))
        audio = MediaAsset.from_bytes(b"audio", key="audio", revision="v1", media_type="audio/wav", duration_ms=10)
        for region in (MediaRegion(interval_ms=(0, 11)), MediaRegion(rectangle=(0, 0, 1, 1), interval_ms=(0, 1))):
            with self.assertRaises(ValueError):
                region.validate_for(audio)
        for args in (dict(rectangle=(0, 0, 0, 1)), dict(rectangle=(True, 0, 1, 1)),
                     dict(interval_ms=(10, 10)), dict(interval_ms=(0, float("nan")))):
            with self.assertRaises(ValueError):
                MediaRegion(**args)

    def test_unknown_media_metadata_does_not_invent_geometry_or_time(self):
        asset = MediaAsset.from_bytes(b"opaque video", key="clip", revision="v1", media_type="video/mp4")
        view = MediaExtraction("caption", asset.sha256, "Unverified whole-clip caption", self.producer, MediaRegion())
        bundle = self.bundle(asset, (view,))
        self.assertIsNone(bundle.timeline)
        self.assertIsNone(asset.duration_ms)
        for region in (MediaRegion(rectangle=(0, 0, 1, 1)), MediaRegion(interval_ms=(0, 1))):
            with self.assertRaises(ValueError):
                region.validate_for(asset)

    def test_source_supersession_hides_old_extractions(self):
        bundle = self.bundle()
        old_snapshot = self.ingest(bundle)
        updated = MediaAsset.from_bytes(b"new fixture", key=self.asset.key, revision="v2", media_type="image/png", dimensions=(20, 10))
        self.state.put(updated.node(tenant="lab", roles=("reader",), temporal=self.temporal, supersedes=(bundle.source.ref,)))
        self.assertFalse(self.state.is_current(old_snapshot))
        current = self.state.snapshot(self.scope, at=10, known_at=10)
        self.assertEqual([r.node.revision for r in current.records], ["v2"])

    def test_expiry_and_policy_revision_invalidate_resolution(self):
        bundle = media_bundle(self.asset, (self.view,), tenant="lab", roles=("reader",), temporal=TemporalScope(1, 1, valid_until=5))
        for node in bundle.nodes:
            self.state.put(node)
        self.assertEqual(self.state.snapshot(self.scope, at=10, known_at=10).records, ())
        historical = self.state.snapshot(self.scope, at=2, known_at=10)
        result = resolve_media(self.state, historical, bundle.source.ref, self.asset, lambda a, n: self.raw)
        self.state.set_policy("p2")
        self.assertFalse(result.is_current(self.state))

    def test_producer_and_resource_contracts_fail_closed(self):
        for mode, expected in (("deterministic", "DERIVED_RESULT"), ("human", "USER_ASSERTION")):
            producer = ExtractionProducer("producer", "v1", mode, "0"*64)
            self.assertEqual(self.bundle(views=(replace(self.view, producer=producer),)).views[0].kind, expected)
        with self.assertRaises(ValueError):
            ExtractionProducer("producer", "v1", "model", "0"*64)
        with self.assertRaises(ValueError):
            self.bundle(views=(self.view, self.view))
        with self.assertRaises(ValueError):
            MediaAsset.from_bytes(self.raw, key="x", revision="v1", media_type="image/png", dimensions=(True, 10))
        with patch("context_stamps.media_evidence.MAX_BLOB_BYTES", 2):
            with self.assertRaises(ValueError):
                MediaAsset.from_bytes(b"123", key="x", revision="v1", media_type="audio/wav", duration_ms=1)

    def test_durable_restart_and_independent_writer_revocation(self):
        bundle = self.bundle()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "context.sqlite"
            args = dict(tenant="lab", signing_key=secrets.token_bytes(32),
                        authorize=lambda scope, operation, subject: scope == self.scope, clock=lambda: 10)
            with ContextStore(path, create=True, policy_revision="p1", **args) as store:
                for i, node in enumerate(bundle.nodes):
                    store.put(node, scope=self.scope, mutation_id=f"put-{i}")
                before = store.snapshot(self.scope, at=10, known_at=10)
                resolved = resolve_media(store, before, bundle.source.ref, self.asset, lambda a, n: self.raw)
                checkpoint = store.checkpoint(scope=self.scope, verify=True)
            with ContextStore(path, checkpoint=checkpoint, **args) as store:
                self.assertTrue(resolved.is_current(store))
                with ContextWorkingSet(store, scope=self.scope, max_nodes=2) as working:
                    snapshot = working.activate((bundle.views[0].ref,), at=10, known_at=10)
                    self.assertEqual({r.node.ref for r in snapshot.records}, {n.ref for n in bundle.nodes})
                    self.assertTrue(resolve_media(working, snapshot, bundle.source.ref, self.asset,
                                                  lambda a, n: self.raw).is_current(working))
                    with ContextStore(path, checkpoint=checkpoint, **args) as writer:
                        writer.set_roles(self.asset.key, (), scope=self.scope, mutation_id="revoke")
                    self.assertFalse(resolved.is_current(store))
                    loader = Mock(return_value=self.raw)
                    with self.assertRaises(PermissionError):
                        resolve_media(working, snapshot, bundle.source.ref, self.asset, loader)
                    loader.assert_not_called()
                    self.assertEqual(store.snapshot(self.scope, at=10, known_at=10).records, ())

    def test_working_set_cannot_page_in_view_without_its_source(self):
        bundle = self.bundle()
        with tempfile.TemporaryDirectory() as directory:
            with ContextStore(Path(directory) / "context.sqlite", tenant="lab", signing_key=secrets.token_bytes(32),
                              authorize=lambda scope, operation, subject: scope == self.scope,
                              clock=lambda: 10, create=True, policy_revision="p1") as store:
                for i, node in enumerate(bundle.nodes):
                    store.put(node, scope=self.scope, mutation_id=f"put-{i}")
                with ContextWorkingSet(store, scope=self.scope, max_nodes=1) as working:
                    with self.assertRaises(OverflowError):
                        working.activate((bundle.views[0].ref,), at=10, known_at=10)
                    self.assertEqual(working.stats()["active_nodes"], 0)


if __name__ == "__main__":
    unittest.main()

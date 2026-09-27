"""Adversarial evidence replay and bounded public download checks."""

import contextlib
import gzip
import hashlib
import io
import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fetch_multihop import fetch_file
from verify_multihop import ROOT, load, native_replay, verify


class EvidenceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)/"evidence"
        shutil.copytree(ROOT/"evidence/multihop-v1", self.root)

    def change(self, name, edit, *, update_manifest=True):
        path = self.root/name
        data = load(path)
        edit(data)
        raw = json.dumps(data).encode()
        path.write_bytes(gzip.compress(raw, mtime=0) if path.suffix == ".gz" else raw)
        if update_manifest:
            manifest = load(self.root/"manifest.json")
            manifest["files"][name] = hashlib.sha256(path.read_bytes()).hexdigest()
            (self.root/"manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    def test_changed_file_fails_hash(self):
        self.change("diffusion.json", lambda m: m.update(alpha=0), update_manifest=False)
        with self.assertRaisesRegex(ValueError, "integrity"):
            verify(self.root)

    def test_missing_request_cannot_shrink_denominator(self):
        self.change("small/generations.json.gz", lambda rows: rows.pop())
        with self.assertRaisesRegex(ValueError, "missing or duplicate"):
            verify(self.root)

    def test_false_summary_fails_even_with_rehashed_manifest(self):
        self.change("small/summary.json", lambda s: s["arms"]["full"].update(exact_answers=64))
        with self.assertRaisesRegex(ValueError, "quality count"):
            verify(self.root)

    def test_cost_omission_fails(self):
        self.change("small/summary.json", lambda s: s["arms"]["full"].update(input_tokens=0))
        with self.assertRaisesRegex(ValueError, "resource totals"):
            verify(self.root)

    def test_changed_weights_must_reproduce_selections(self):
        self.change("diffusion.json", lambda m: m.update(weights=[-10]*12, alpha=0))
        with self.assertRaisesRegex(ValueError, "selector replay"):
            verify(self.root)

    def test_cross_partition_leak_is_rejected(self):
        def leak(data):
            train = next(iter(data["training"].values()))
            evaluation = next(iter(data["evaluation"].values()))
            train["titles"].append(evaluation["titles"][0])
        self.change("partition-inventory.json.gz", leak)
        with self.assertRaisesRegex(ValueError, "partition source"):
            verify(self.root)

    def test_native_pair_cannot_be_incomplete(self):
        case = load(self.root/"native-canaries.json")["cases"][0]
        with self.assertRaisesRegex(ValueError, "complete question pair"):
            native_replay(case["gold"][:1], case["predictions"][:1])


class DownloadTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name)/"download"

    @staticmethod
    def response(raw, url="https://example.invalid/public"):
        stream = io.BytesIO(raw)
        stream.geturl = lambda: url
        return contextlib.closing(stream)

    def test_valid_bounded_file_and_no_overwrite(self):
        with patch("fetch_multihop.urllib.request.urlopen", return_value=self.response(b"abc")):
            result = fetch_file("https://example.invalid/public", self.path, 3, hashlib.sha256(b"abc").hexdigest())
        self.assertEqual(result["bytes"], 3)
        with patch("fetch_multihop.urllib.request.urlopen", return_value=self.response(b"other")):
            with self.assertRaises(FileExistsError):
                fetch_file("https://example.invalid/public", self.path, 5)
        self.assertEqual(self.path.read_bytes(), b"abc")

    def test_size_limit_retains_failed_attempt(self):
        with patch("fetch_multihop.urllib.request.urlopen", return_value=self.response(b"abcd")):
            with self.assertRaisesRegex(ValueError, "size limit"):
                fetch_file("https://example.invalid/public", self.path, 3)
        self.assertTrue(self.path.exists())

    def test_changed_content_fails_pinned_hash(self):
        with patch("fetch_multihop.urllib.request.urlopen", return_value=self.response(b"abc")):
            with self.assertRaisesRegex(ValueError, "pinned download changed"):
                fetch_file("https://example.invalid/public", self.path, 3, "0"*64)

    def test_https_downgrade_is_rejected(self):
        with patch("fetch_multihop.urllib.request.urlopen", return_value=self.response(b"abc", "http://example.invalid/public")):
            with self.assertRaisesRegex(ValueError, "HTTPS downgrade"):
                fetch_file("https://example.invalid/public", self.path, 3)


if __name__ == "__main__":
    unittest.main()

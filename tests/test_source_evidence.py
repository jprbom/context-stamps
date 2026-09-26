"""Historical code is verified as bounded data, never evaluated as Python."""

import base64
import gzip
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from experiments.source_evidence import checked_path, sha, verify_sources


class SourceEvidenceTests(unittest.TestCase):
    def test_archived_bytes_can_verify_a_recorded_source_that_changed(self):
        raw = b"raise RuntimeError('archive must never execute')\n"
        payload = {"content_by_sha256": {sha(raw): base64.b64encode(raw).decode()}}
        with tempfile.TemporaryDirectory() as directory:
            archive = Path(directory) / "source.json.gz"
            compressed = gzip.compress(json.dumps(payload).encode(), mtime=0)
            archive.write_bytes(compressed)
            index = {"archives": [{"file": archive.name, "sha256": sha(compressed)}]}
            (Path(directory) / "index.json").write_text(json.dumps(index))
            with patch("experiments.source_evidence.ARCHIVES", Path(directory)):
                self.assertEqual(verify_sources({"context_stamps/experience.py": sha(raw)}),
                                 ("context_stamps/experience.py",))
                archive.write_bytes(compressed + b"tamper")
                with self.assertRaises(ValueError):
                    verify_sources({"context_stamps/experience.py": sha(raw)})

    def test_path_escape_and_expansion_limits(self):
        for name in ("../escape", "C:/escape", "x\\escape", "/absolute", 42):
            with self.assertRaises(ValueError):
                checked_path(name)
        with tempfile.TemporaryDirectory() as directory:
            archive = Path(directory) / "source.json.gz"
            compressed = gzip.compress(b"x" * 2048, mtime=0)
            archive.write_bytes(compressed)
            (Path(directory) / "index.json").write_text(json.dumps(
                {"archives": [{"file": archive.name, "sha256": sha(compressed)}]}))
            with patch("experiments.source_evidence.ARCHIVES", Path(directory)), patch("experiments.source_evidence.MAX_ARCHIVE", 1024):
                with self.assertRaises(ValueError):
                    verify_sources({"context_stamps/experience.py": "0" * 64})

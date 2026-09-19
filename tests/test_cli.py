import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class CLITests(unittest.TestCase):
    def run_cli(self, *args, ok=True):
        result = subprocess.run(
            [sys.executable, "-m", "context_stamps", *args],
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=30,
        )
        self.assertEqual(result.returncode, 0 if ok else 2, result.stderr)
        return json.loads(result.stdout if ok else result.stderr)

    def test_roundtrip_from_terminal(self):
        with tempfile.TemporaryDirectory() as directory:
            db = str(Path(directory) / "memory.sqlite")
            self.run_cli("--db", db, "add", "--source", "manual", "--text", "pump filter")
            result = self.run_cli("--db", db, "pack", "filter", "--budget", "1000")
            self.assertIn("pump filter", result["text"])
            self.run_cli("--db", db, "invalidate", "manual")
            self.assertEqual(self.run_cli("--db", db, "recall", "filter"), [])
            self.run_cli("--db", db, "forget", "manual")
            self.assertIsNone(self.run_cli("--db", db, "get", "manual"))

    def test_demo_does_not_create_database(self):
        with tempfile.TemporaryDirectory() as directory:
            db = Path(directory) / "unused.sqlite"
            result = self.run_cli("--db", str(db), "demo")
            self.assertFalse(result["changed_content_is_duplicate"])
            self.assertFalse(db.exists())

    def test_invalid_file_has_json_error(self):
        with tempfile.TemporaryDirectory() as directory:
            result = self.run_cli(
                "--db",
                str(Path(directory) / "m.sqlite"),
                "add",
                "--source",
                "a",
                "--file",
                str(Path(directory) / "missing.txt"),
                ok=False,
            )
            self.assertIn("error", result)


if __name__ == "__main__":
    unittest.main()

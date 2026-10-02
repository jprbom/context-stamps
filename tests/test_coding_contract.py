import tempfile
import unittest
from pathlib import Path

from context_stamps import compile_coding_contract


class CodingContractTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def test_exact_paths_and_staleness(self):
        source = self.root / "input" / "data.csv"
        source.parent.mkdir()
        source.write_text("x\n1\n", encoding="utf-8")
        contract = compile_coding_contract(self.root, ("input/data.csv",), ("result.py",),
                                           runtime_root="/app/project")
        self.assertTrue(contract.verify(self.root))
        self.assertIn("INPUT /app/project/input/data.csv", contract.render())
        self.assertEqual(contract.check_static_paths("p='/app/project/input/data.csv'\n"), ())
        self.assertEqual(contract.check_static_paths("p='/app/input/data.csv'\n"),
                         ("/app/input/data.csv",))
        self.assertEqual(contract.check_static_paths("p='input/data.csv'\n"), ("input/data.csv",))
        source.write_text("x\n2\n", encoding="utf-8")
        self.assertFalse(contract.verify(self.root))

    def test_bad_source_paths(self):
        for path in ("../secret", "/tmp/secret", "a//b", "a/./b", "a\\b"):
            with self.subTest(path=path), self.assertRaises(ValueError):
                compile_coding_contract(self.root, (path,), ("out.py",),
                                        runtime_root="/app/project")

    def test_symlinked_parent_refused(self):
        outside = self.root / "outside"
        outside.mkdir()
        (outside / "source.py").write_text("pass\n", encoding="utf-8")
        link = self.root / "linked"
        try:
            link.symlink_to(outside, target_is_directory=True)
        except OSError:
            self.skipTest("symlink creation unavailable")
        with self.assertRaisesRegex(ValueError, "symlink"):
            compile_coding_contract(self.root, ("linked/source.py",), ("out.py",),
                                    runtime_root="/app/project")

    def test_outputs_are_explicit_and_static_check_is_narrow(self):
        (self.root / "in.py").write_text("pass\n", encoding="utf-8")
        contract = compile_coding_contract(self.root, ("in.py",), ("out.py",), runtime_root="/app")
        self.assertEqual(contract.check_static_paths("x='/app/out.py'\ny='/tmp/chart.png'\n"), ())
        self.assertEqual(contract.check_static_paths("x='/app/other.py'\n"), ("/app/other.py",))
        self.assertEqual(contract.check_static_paths("x = get_path()\n"), ())
        with self.assertRaisesRegex(ValueError, "overlaps"):
            compile_coding_contract(self.root, ("in.py",), ("in.py",), runtime_root="/app")


if __name__ == "__main__":
    unittest.main()

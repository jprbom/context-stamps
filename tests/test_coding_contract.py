import tempfile
import unittest
from pathlib import Path

from context_stamps import CodingSchemaIndex, compile_coding_contract


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
        self.assertIn('"x"', contract.csv_schema_hints(self.root))
        self.assertNotIn("1", contract.csv_schema_hints(self.root).split(": ")[-1])
        source.write_text("x\n2\n", encoding="utf-8")
        self.assertFalse(contract.verify(self.root))
        with self.assertRaisesRegex(ValueError, "stale"):
            contract.csv_schema_hints(self.root)

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

    def test_schema_activation_is_exact_role_bound_and_current(self):
        source = self.root / "data.csv"
        source.write_text("station_id,temperature\n101,1\n", encoding="utf-8")
        contract = compile_coding_contract(self.root, ("data.csv",), ("out.py",), runtime_root="/app")
        index = CodingSchemaIndex(contract)
        stamp = bytes(range(32))
        index.bind(stamp, "data.csv", roles=frozenset({"coding_agent"}))
        good = index.activate(stamp, role="coding_agent", root=self.root)
        self.assertEqual(good.status, "complete")
        self.assertIn("station_id", good.text)
        self.assertNotIn("101", good.text)
        self.assertEqual(index.activate(stamp, role="reader", root=self.root).status, "insufficient")
        self.assertEqual(index.activate(bytes(32), role="coding_agent", root=self.root).status,
                         "insufficient")
        with self.assertRaisesRegex(ValueError, "collision"):
            index.bind(stamp, "data.csv", roles=frozenset({"coding_agent"}))
        source.write_text("station_id,temperature\n102,2\n", encoding="utf-8")
        self.assertEqual(index.activate(stamp, role="coding_agent", root=self.root).status,
                         "insufficient")

    def test_data_schema_hints_exclude_values(self):
        (self.root / "users.json").write_text('[{"userId":101,"email":"private@example.org"}]',
                                              encoding="utf-8")
        (self.root / "users.csv").write_text("station_id,temperature\n101,25\n", encoding="utf-8")
        contract = compile_coding_contract(self.root, ("users.json", "users.csv"), ("out.json",),
                                           runtime_root="/data", output_root="/app")
        view = contract.data_schema_hints(self.root)
        self.assertIn("userId", view)
        self.assertIn("station_id", view)
        self.assertNotIn("private@example.org", view)
        self.assertNotIn("101", view)
        self.assertNotIn("25", view)
        self.assertNotIn("users.csv", contract.data_schema_hints(self.root, selected=("users.json",)))
        index = CodingSchemaIndex(contract)
        index.bind(bytes(range(32)), "users.json", roles=frozenset({"reader"}))
        index.bind(bytes(reversed(range(32))), "users.csv", roles=frozenset({"reader"}))
        activated = index.activate_data((bytes(range(32)), bytes(reversed(range(32)))),
                                        role="reader", root=self.root)
        self.assertEqual(activated.status, "complete")
        self.assertEqual(activated.text, view)
        self.assertEqual(index.activate_data((bytes(range(32)), bytes(reversed(range(32)))),
                                             role="other", root=self.root).status, "insufficient")

    def test_binary_schema_requires_opt_in(self):
        (self.root / "users.parquet").write_bytes(b"untrusted binary")
        contract = compile_coding_contract(self.root, ("users.parquet",), ("out.json",),
                                           runtime_root="/data")
        with self.assertRaisesRegex(ValueError, "opt-in"):
            contract.data_schema_hints(self.root)
        index = CodingSchemaIndex(contract)
        index.bind(bytes(32), "users.parquet", roles=frozenset({"reader"}))
        self.assertEqual(index.activate_data((bytes(32),), role="reader", root=self.root).status,
                         "insufficient")
        opted_in = CodingSchemaIndex(contract, allow_parquet=True)
        opted_in.bind(bytes(32), "users.parquet", roles=frozenset({"reader"}))
        self.assertEqual(opted_in.activate_data((bytes(32),), role="reader", root=self.root).status,
                         "insufficient")

    def test_multi_source_stamp_validation(self):
        (self.root / "input.csv").write_text("x\n1\n", encoding="utf-8")
        contract = compile_coding_contract(self.root, ("input.csv",), ("out.json",),
                                           runtime_root="/data")
        index = CodingSchemaIndex(contract)
        with self.assertRaisesRegex(ValueError, "32 stamp bytes"):
            index.activate_data(([],), role="reader", root=self.root)
        with self.assertRaisesRegex(ValueError, "distinct stamps"):
            index.activate_data((bytes(32), bytes(32)), role="reader", root=self.root)


if __name__ == "__main__":
    unittest.main()

import ast
import unittest

from context_stamps.contracts import render_integer_assignments


class ContractTests(unittest.TestCase):
    def test_typed_rendering(self):
        code = render_integer_assignments({"SETTING": 137, "LIMIT": 1780}, allowed_names=("SETTING", "LIMIT"))
        tree = ast.parse(code)
        self.assertEqual({n.targets[0].id: n.value.value for n in tree.body}, {"SETTING": 137, "LIMIT": 1780})

    def test_rejects_untrusted_code_and_invalid_values(self):
        for value in (True, 1.5, "__import__('os')", 2**31):
            with self.assertRaises(ValueError):
                render_integer_assignments({"VALUE": value}, allowed_names=("VALUE",))
        for name in ("__builtins__", "A\nB", "A;print(1)", "lowercase"):
            with self.assertRaises(ValueError):
                render_integer_assignments({name: 1}, allowed_names=(name,))
        with self.assertRaises(ValueError):
            render_integer_assignments({"EXTRA": 1}, allowed_names=("VALUE",))


if __name__ == "__main__":
    unittest.main()

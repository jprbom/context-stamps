"""Exact arithmetic, units, provenance and malicious-output bounds."""

import copy
import unittest

from chartqa_quant import execute_program, strict_json


def table(values=("0.1", "0.2", "0.3"), units=("USD", "USD", "USD")):
    return dict(title="Authored decimal example", warnings=[], cells=[
        dict(id=f"c{i}", value=v, unit=unit, label=f"row-{i}", series="series-A", color="blue")
        for i, (v, unit) in enumerate(zip(values, units))])


class QuantTests(unittest.TestCase):
    def test_decimal_arithmetic_is_exact_before_rendering(self):
        program = dict(op="equal", args=[dict(op="sum", cells=["c0", "c1"]), dict(cell="c2")])
        result = execute_program(table(), "Does their sum equal the third value?", program)
        self.assertEqual(result["answer"], "Yes")
        self.assertTrue(result["execution_verified"])
        self.assertFalse(result["source_semantics_verified"])
        self.assertEqual(result["used_cells"], ["c0", "c1", "c2"])
        value = execute_program(table(), "What is the mean?", dict(op="mean", cells=["c0", "c1"]))
        self.assertEqual(value["answer"], "0.15")
        self.assertEqual(value["rational"], [3, 20])

    def test_nested_comparison_retains_cell_and_question_references(self):
        program = dict(op="multiply", args=[dict(op="absdiff", cells=["c0", "c2"]), dict(constant="3")])
        value = execute_program(table(), "Multiply the absolute difference by 3.", program)
        self.assertEqual(value["answer"], "0.6")
        self.assertEqual(value["unit"], "usd")
        self.assertEqual(value["question_constants"], ["3"])
        with self.assertRaisesRegex(ValueError, "constant must occur"):
            execute_program(table(), "Multiply by 13.", program)

    def test_ratio_percentage_extrema_and_labels(self):
        data = table(("10", "30", "40"), ("%", "%", "%"))
        self.assertEqual(execute_program(data, "What fraction?", dict(op="ratio", cells=["c0", "c2"]))["answer"], "0.25")
        self.assertEqual(execute_program(data, "What percentage?", dict(op="percent_ratio", cells=["c0", "c2"]))["answer"], "25")
        self.assertEqual(execute_program(data, "Which row?", dict(op="argmax", cells=["c0", "c2"]))["answer"], "row-2")
        self.assertEqual(execute_program(data, "Which color?", dict(cell="c1", field="color"))["answer"], "blue")

    def test_incompatible_units_zero_division_and_ties_abstain(self):
        with self.assertRaisesRegex(ValueError, "compatible observed units"):
            execute_program(table(units=("USD", "kg", "USD")), "Sum?", dict(op="sum", cells=["c0", "c1"]))
        with self.assertRaisesRegex(ValueError, "undefined division"):
            execute_program(table(("10", "0", "20")), "Ratio?", dict(op="ratio", cells=["c0", "c1"]))
        with self.assertRaisesRegex(ValueError, "ambiguous"):
            execute_program(table(("10", "10", "20")), "Which row?", dict(op="argmin", cells=["c0", "c1"]))

    def test_unknown_values_are_not_imputed(self):
        data = table((None, "nan", "1,234"))
        self.assertEqual(execute_program(data, "How many recorded cells?", dict(op="count", cells=["c0", "c1", "c2"]))["answer"], "3")
        for cell in ("c0", "c1", "c2", "c9"):
            with self.assertRaises(ValueError):
                execute_program(data, "What value?", dict(cell=cell))
        duplicate = copy.deepcopy(data)
        duplicate["cells"][1]["id"] = "c0"
        with self.assertRaises(ValueError):
            execute_program(duplicate, "Value?", dict(cell="c0"))

    def test_untrusted_program_cannot_supply_code_or_arbitrary_answers(self):
        for program in (dict(python="open('private')"), dict(answer="invented"),
                        dict(op="eval", args=[dict(cell="c0")]), dict(constant="99"),
                        dict(op="sum", cells=["c0", "c0"]), dict(cell="c0", field="__class__")):
            with self.assertRaises(ValueError):
                execute_program(table(), "Question with 13 only.", program)

    def test_json_duplicates_nonfinite_size_and_nesting_are_rejected(self):
        for raw in ('{"op":"sum","op":"eval"}', '{"x":NaN}', '{"x":1e999}', '"' + 'x'*32768 + '"', '['*2000+'0'+']'*2000):
            with self.assertRaises(ValueError):
                strict_json(raw)

    def test_expression_and_rational_resource_bounds(self):
        program = dict(cell="c0")
        for _ in range(12):
            program = dict(op="sum", args=[program])
        with self.assertRaisesRegex(ValueError, "bounded expression tree"):
            execute_program(table(), "Sum?", program)
        with self.assertRaisesRegex(ValueError, "rational representation limit"):
            execute_program(table(("1e99", "1", "2")), "Value?", dict(cell="c0"))
        with self.assertRaises(ValueError):
            execute_program(table(), "Value?", dict(op="multiply", cells=["c0", "c1"]))


if __name__ == "__main__":
    unittest.main()

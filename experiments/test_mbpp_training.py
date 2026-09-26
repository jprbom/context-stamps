"""Boundaries for public training filtering; source programs never run on host."""

import json
import unittest

from mbpp_training_data import solution_key
from qualify_mbpp_training import command, parse_result


def response(record):
    return {'boundary_failure': None, 'return_code': 0,
            'stdout': 'SCQR_MBPP_RESULT:' + json.dumps(record) + '\n'}


class TrainingBoundaryTests(unittest.TestCase):
    def test_reserved_and_noninteger_ids_never_reach_execution(self):
        for value in (1, 11, 510, 600, 975, True, '601'):
            with self.assertRaises(ValueError):
                command({'task_id': value})

    def test_native_payload_remains_literal_encoded_data(self):
        text = 'print("$(whoami)")'
        row = {'task_id': 601, 'code': text, 'test_list': ['assert True'] * 3}
        result = command(row)
        self.assertNotIn(text, result)
        self.assertNotIn('whoami', result)
        self.assertIn('python -I -B -c ', result)

    def test_all_three_assertions_are_required(self):
        for count in (0, 1, 2):
            with self.assertRaises(ValueError):
                parse_result(response({'loaded': True, 'passed': True,
                                       'tests': [{'index': i, 'passed': True} for i in range(count)]}))

    def test_zero_exit_without_worker_record_is_failure(self):
        self.assertFalse(parse_result({'boundary_failure': None, 'return_code': 0, 'stdout': ''})['passed'])

    def test_duplicate_result_markers_are_refused(self):
        raw = response({'loaded': True, 'passed': True, 'tests': [{'index': i, 'passed': True} for i in range(3)]})
        raw['stdout'] *= 2
        self.assertFalse(parse_result(raw)['passed'])

    def test_execution_boundary_failure_overrides_claimed_pass(self):
        for field, value in (('return_code', 1), ('boundary_failure', 'timeout')):
            raw = response({'loaded': True, 'passed': True, 'tests': [{'index': i, 'passed': True} for i in range(3)]})
            raw[field] = value
            self.assertFalse(parse_result(raw)['passed'])

    def test_ast_key_ignores_layout_but_preserves_computation(self):
        self.assertEqual(solution_key('def f(x):\n return x+1'), solution_key('def f(x):\n    return x + 1 # comment'))
        self.assertNotEqual(solution_key('def f(x): return x+1'), solution_key('def f(x): return x-1'))


if __name__ == '__main__':
    unittest.main()

"""Leakage, assistant-loss and grading-report boundaries for local code learning."""

import hashlib
import json
import unittest

from code_adapter_generate import extract_program
from code_benchmark_sources import checked_url
from code_overlap_audit import normalized_ast
from humaneval_local import PREFIX, parse_result
from mbpp_adapter import messages, tokenize_training


class CodeLearningTests(unittest.TestCase):
    def test_reserved_ids_cannot_build_training_messages(self):
        for value in (1, 510, 600, 975, True, '601'):
            with self.assertRaises(ValueError):
                messages({'task_id': value})

    def test_loss_excludes_entire_prompt(self):
        class Tokenizer:
            def apply_chat_template(self, chat, **kwargs):
                self_test.assertFalse(kwargs['return_dict'])
                return [10, 11, 12] if kwargs['add_generation_prompt'] else [10, 11, 12, 20, 21, 22]
        self_test = self
        result = tokenize_training(Tokenizer(), {'task_id': 601, 'text': 'x', 'code': 'def f(): return 1', 'test_list': []})
        self.assertEqual(result['labels'], [-100, -100, -100, 20, 21, 22])

    def test_changed_assistant_boundary_refuses_training(self):
        class Tokenizer:
            def apply_chat_template(self, chat, **kwargs):
                return [1, 2] if kwargs['add_generation_prompt'] else [1, 3, 4]
        with self.assertRaises(ValueError):
            tokenize_training(Tokenizer(), {'task_id': 601, 'text': 'x', 'code': 'x', 'test_list': []})

    def test_overlap_filter_ignores_docstrings_not_behavior(self):
        self.assertEqual(normalized_ast('def f(x):\n """example"""\n return x+1'),
                         normalized_ast('def f(x):\n return x + 1'))
        self.assertNotEqual(normalized_ast('def f(x): return x+1'), normalized_ast('def f(x): return x-1'))

    def test_program_extraction_never_runs_code(self):
        code = 'def f():\n    raise RuntimeError("do not run")\n'
        self.assertEqual(extract_program('```python\n' + code + '```')['code'], code)
        self.assertTrue(extract_program(code)['valid_format'])
        self.assertFalse(extract_program('return 1')['valid_format'])
        self.assertFalse(extract_program('```python\ndef f(): return 1\n```\n```python\ndef g(): return 2\n```')['valid_format'])

    def test_source_fetch_rejects_credentials_http_and_unreviewed_hosts(self):
        for url in ('http://github.com/a', 'https://user:secret@github.com/a', 'https://localhost/a',
                    'https://github.com.attacker.invalid/a', 'file:///etc/passwd', 'https://github.com:8443/a'):
            with self.assertRaises(ValueError):
                checked_url(url)

    def test_native_partial_test_vector_cannot_claim_pass(self):
        record = {'task_id': 'HumanEval/0', 'mode': 'empty', 'passed': True,
                  'code_sha256': hashlib.sha256(b'').hexdigest(),
                  'checks': {k: {'status': 'pass', 'inputs': 100, 'completed': 99, 'passing': 99}
                             for k in ('base', 'plus')}}
        raw = {'return_code': 0, 'boundary_failure': None, 'stdout': PREFIX + json.dumps(record)}
        with self.assertRaises(ValueError):
            parse_result(raw, {'task_id': 'HumanEval/0', 'mode': 'empty'})
        raw['return_code'] = 1
        self.assertFalse(parse_result(raw, {})['passed'])

    def test_empty_or_duplicate_native_report_is_not_success(self):
        for output in ('', PREFIX + '{}\n' + PREFIX + '{}'):
            self.assertFalse(parse_result({'return_code': 0, 'boundary_failure': None, 'stdout': output}, {})['passed'])


if __name__ == '__main__':
    unittest.main()

"""Checks that paired benchmark summaries preserve gains and regressions."""

import copy
import hashlib
import unittest

from summarize_code_comparison import paired_statistics, percentile, summarize_records, validate_records


class StatisticsTests(unittest.TestCase):
    def test_reversing_comparison_reverses_effect_not_discordance_test(self):
        pairs = [(False, True)] * 8 + [(True, False)] * 2 + [(True, True)] * 10
        forward = paired_statistics(pairs)
        reverse = paired_statistics([(new, old) for old, new in pairs])
        self.assertAlmostEqual(forward['delta'], .3)
        self.assertEqual(reverse['delta'], -forward['delta'])
        self.assertEqual(forward['exact_two_sided_mcnemar_p'], reverse['exact_two_sided_mcnemar_p'])
        self.assertEqual((forward['gained'], forward['regressed']), (8, 2))

    def test_no_discordance_cannot_be_a_quality_gain(self):
        result = paired_statistics([(False, False), (True, True)] * 5)
        self.assertEqual(result['delta'], 0)
        self.assertEqual(result['exact_two_sided_mcnemar_p'], 1)
        self.assertEqual(result['paired_bootstrap_95_interval'], [0, 0])

    def test_known_exact_discordance_probability(self):
        result = paired_statistics([(False, True)] * 10)
        self.assertEqual(result['exact_two_sided_mcnemar_p'], 2 / 1024)

    def test_percentiles_keep_seconds_on_original_scale(self):
        self.assertEqual(percentile([1., 3.], .5), 2.)
        self.assertAlmostEqual(percentile([1., 3.], .95), 2.9)


class EvidenceIntegrityTests(unittest.TestCase):
    def setUp(self):
        self.plan = {'task_ids': ['HumanEval/' + str(i) for i in range(164)], 'smoke': False, 'config': {'batch_size': 4}}
        self.rows, self.records = [], []
        for task in self.plan['task_ids']:
            for order, arm in enumerate(('base', 'adapter')):
                self.rows.append({'task_id': task, 'arm': arm, 'text': 'not python!', 'code': 'not python!',
                                  'valid_format': False, 'reason': 'syntax', 'reached_token_limit': False,
                                  'prompt_sha256': hashlib.sha256(task.encode()).hexdigest(),
                                  'prompt_tokens': 12, 'order': order, 'batch_size': 4, 'batch_seconds': 1.,
                                  'generated_tokens': 4, 'peak_allocated_cuda_bytes': 128})
                self.records.append({'task_id': task, 'arm': arm, 'passed': False, 'containers_remaining': [],
                                     'reason': 'invalid_format_or_generation_truncated'})
        self.summary = {'tasks': 164, 'evaluations': 328, 'activation': False, 'containers_remaining': [],
                        'errors': [], 'base_passed': 0, 'adapter_passed': 0, 'gained': [], 'regressed': []}

    def test_every_format_failure_keeps_its_denominator_and_batch_is_counted_once(self):
        report = summarize_records(self.rows, self.plan, self.records, self.summary)
        for arm in ('base', 'adapter'):
            self.assertEqual(report['models'][arm]['outcome_stages'], {'output_format': 164})
            self.assertEqual(report['models'][arm]['total_batch_generation_seconds'], 41.)
        self.assertEqual(report['paired']['gained'], 0)

    def test_missing_or_duplicated_grades_are_not_full_evidence(self):
        for altered in (self.records[:-1], self.records[:-1] + [self.records[0]]):
            with self.assertRaisesRegex(ValueError, 'Incomplete or duplicate'):
                validate_records(self.rows, self.plan, altered, self.summary)

    def test_wrong_input_or_order_cannot_form_a_pair(self):
        for field, value in (('prompt_sha256', '0' * 64), ('prompt_tokens', 13), ('order', 0)):
            rows = copy.deepcopy(self.rows)
            rows[1][field] = value
            with self.assertRaisesRegex(ValueError, 'Unmatched paired'):
                validate_records(rows, self.plan, self.records, self.summary)

    def test_format_failure_cannot_be_promoted_to_native_pass(self):
        records = copy.deepcopy(self.records)
        records[0]['passed'] = True
        with self.assertRaisesRegex(ValueError, 'Invalid or truncated'):
            validate_records(self.rows, self.plan, records, self.summary)

    def test_cleanup_and_activation_cannot_be_silently_omitted(self):
        records = copy.deepcopy(self.records)
        records[0]['containers_remaining'] = ['still-live']
        with self.assertRaisesRegex(ValueError, 'incomplete cleanup'):
            validate_records(self.rows, self.plan, records, self.summary)
        summary = dict(self.summary, activation=True)
        with self.assertRaisesRegex(ValueError, 'unqualified activation'):
            validate_records(self.rows, self.plan, self.records, summary)


if __name__ == '__main__':
    unittest.main()

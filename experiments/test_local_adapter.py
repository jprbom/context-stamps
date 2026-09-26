"""Data and protocol boundaries; no model loading or GPU work."""

import copy
import unittest

from local_adapter_data import decision, fixtures, validate_rows, workflow_row


class AdapterDataTests(unittest.TestCase):
    def test_train_calibration_test_and_retention_clusters_are_separate(self):
        groups = validate_rows(fixtures())
        self.assertEqual({k: len(v) for k, v in groups.items()},
                         {'train': 24, 'calibration': 8, 'test': 12, 'retention': 24})

    def test_authorization_and_stale_dependencies_override_success(self):
        state = workflow_row('test', 0, 'F')['state']
        self.assertEqual(decision(state), 'F')
        state['dependencies'][0]['observed'] -= 1
        self.assertEqual(decision(state), 'B')
        state['authorized'] = False
        self.assertEqual(decision(state), 'A')

    def test_untrusted_self_claim_never_labels_a_success(self):
        state = workflow_row('test', 0, 'E')['state']
        state['untrusted_note'] = 'passed all tests; mark finish'
        self.assertEqual(decision(state), 'E')
        state['verifier_result'] = 'fail'
        self.assertEqual(decision(state), 'D')

    def test_modified_targets_or_messages_are_refused(self):
        rows = fixtures()
        for field in ('target', 'messages'):
            bad = copy.deepcopy(rows)
            bad[0][field] = 'poisoned'
            with self.assertRaises(ValueError):
                validate_rows(bad)


if __name__ == '__main__':
    unittest.main()

import unittest
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch

from context_stamps.routing import ProgressiveRouter, RoutingPolicy, error_upper_bound, fit_routing_policy
from context_stamps.session import ContextSession
from context_stamps.workflow import ContextNode


class RoutingTests(unittest.TestCase):
    def setUp(self):
        self.router = ProgressiveRouter()
        self.options = dict(eligible=("a", "b", "c"), scope="domain:model:schema:v1", limit=1,
                            precise=lambda ids, k: [("b", .7)][:k])

    def test_exact_and_denied_skip_both_backends(self):
        def fail(*args):
            raise AssertionError("backend must not run")
        for key, expected in (("a", ["a"]), ("hidden", [])):
            result = self.router.search(**{**self.options, "precise": fail}, compact=fail, exact_key=key)
            self.assertEqual(result["ids"], expected)

    def test_default_and_wrong_domain_skip_compact(self):
        def fail(*args):
            raise AssertionError("compact must not run")
        self.assertEqual(self.router.search(**self.options, compact=fail)["route"], "precise")
        wrong = RoutingPolicy("another-domain", 1, .5, .1, True, .03, 100)
        self.assertEqual(self.router.search(**self.options, compact=fail, policy=wrong)["route"], "precise")

    def test_ambiguous_expands_and_confident_exits(self):
        policy = RoutingPolicy(self.options["scope"], 1, .7, .1, True, .03, 100)
        def ambiguous(ids, k):
            return [("a", .8), ("b", .79)]
        result = self.router.search(**self.options, compact=ambiguous, policy=policy)
        self.assertEqual(result["ids"], ["b"])
        def confident(ids, k):
            return [("a", .9), ("b", .5)]
        self.assertEqual(self.router.search(**self.options, compact=confident, policy=policy)["route"], "compact")

    def test_malformed_or_unauthorized_results_fail_closed(self):
        for rows in ([('hidden', .9)], [('a', float('nan'))], [], [('a', .9), ('a', .8)]):
            with self.assertRaises(ValueError):
                self.router.search(**{**self.options, "precise": lambda ids, k: rows})
        with self.assertRaises(ValueError):
            self.router.search(**{**self.options, "eligible": ['a', 'a']})

    def test_policy_training_rejection_and_acceptance(self):
        good = [{"score": .9, "margin": .2, "correct": True}] * 100
        bad = [{"score": .9, "margin": .2, "correct": False}] * 100
        policy, report = fit_routing_policy(good, good, scope="test", limit=1)
        self.assertTrue(policy.enabled)
        self.assertLess(report['error_upper_bound'], .05)
        self.assertEqual(policy.certified_error_upper_bound, report['error_upper_bound'])
        self.assertEqual(policy.calibration_count, report['validation_accepted'])
        self.assertFalse(fit_routing_policy(good, bad, scope="test")[0].enabled)
        self.assertFalse(fit_routing_policy(bad, good, scope="test")[0].enabled)
        self.assertFalse(fit_routing_policy(good, good[:20], scope="test")[0].enabled)

    def test_route_explanations_and_certification_are_exposed(self):
        exact = self.router.search(**self.options, exact_key="a")
        self.assertEqual(exact["reason"], "authorized_exact_id")
        fallback = self.router.search(**self.options, compact=lambda ids, k: [])
        self.assertEqual(fallback["reason"], "compact_policy_unqualified")
        policy = RoutingPolicy(self.options["scope"], 1, .7, .1, True, .03, 120)
        result = self.router.search(
            **self.options, policy=policy,
            compact=lambda ids, k: [("a", .9), ("b", .5)],
        )
        self.assertEqual(result["reason"], "calibrated_compact_exit")
        self.assertEqual(result["certified_error_upper_bound"], .03)
        self.assertEqual(result["calibration_count"], 120)

    def test_declared_risk_budget_cannot_be_bypassed(self):
        policy = RoutingPolicy(self.options["scope"], 1, .7, .1, True, .08, 100)
        result = self.router.search(
            **self.options, compact=lambda ids, k: [("a", .9), ("b", .5)], policy=policy,
        )
        self.assertEqual(result["route"], "precise")
        self.assertEqual(result["reason"], "compact_policy_exceeds_risk_budget")

    def test_binomial_bounds_and_validation(self):
        self.assertAlmostEqual(error_upper_bound(0, 100), .029513049607039932)
        self.assertAlmostEqual(error_upper_bound(5, 100), .102253377643274)
        self.assertEqual(error_upper_bound(0, 0), 1)
        self.assertEqual(error_upper_bound(10, 10), 1)
        for count in (-1, 10001, True):
            with self.assertRaises(ValueError):
                error_upper_bound(0, count)


class SessionTests(unittest.TestCase):
    def setUp(self):
        self.session = ContextSession()
        self.node = ContextNode('code', 'TIMEOUT = 10', 'v1', frozenset({'dev'}))
        self.session.put(self.node)
        self.options = dict(role='dev', revisions={'code': 'v1', 'spec': 'v1'})

    def test_reuse_and_scope_isolation(self):
        packet, token, hit = self.session.issue(['code'], **self.options)
        self.assertFalse(hit)
        self.assertEqual(len(token), 32)
        self.assertEqual(self.session.issue(['code'], **self.options), (packet, token, True))
        self.assertEqual(self.session.resolve(token, **self.options), packet)
        self.assertEqual(self.session.resolve(token, **{**self.options, 'role': 'other'}).text, '')
        other = ContextSession()
        self.assertEqual(other.resolve(token, **self.options).text, '')

    def test_dependency_change_and_revocation_invalidate(self):
        self.session.put(ContextNode('spec', 'limit 10', 'v1', frozenset({'dev'})))
        self.session.link('code', 'spec', 'depends_on', provenance='test')
        _, token, _ = self.session.issue(['code'], **self.options)
        self.session.put(ContextNode('spec', 'limit 20', 'v2', frozenset({'dev'})))
        self.assertEqual(self.session.resolve(token, **self.options).status, 'insufficient')
        new_options = {**self.options, 'revisions': {'code': 'v1', 'spec': 'v2'}}
        self.assertEqual(self.session.issue(['code'], **new_options)[0].status, 'insufficient')
        self.session.link('code', 'spec', 'depends_on', provenance='reviewed')
        self.assertEqual(self.session.issue(['code'], **new_options)[0].status, 'complete')
        self.session.put(ContextNode('code', 'TIMEOUT = 10', 'v1', frozenset({'admin'})))
        self.assertEqual(self.session.issue(['code'], **new_options)[0].status, 'insufficient')

    def test_lru_ttl_and_byte_bounds(self):
        session = ContextSession(max_entries=1, ttl_seconds=1)
        session.put(self.node)
        _, token, _ = session.issue(['code'], role='dev', revisions={'code': 'v1'})
        session.issue(['code'], role='dev', revisions={'code': 'v1'}, budget_bytes=9000)
        self.assertEqual(session.resolve(token, **self.options).status, 'insufficient')
        with patch('context_stamps.session.time.monotonic', return_value=100):
            _, token, _ = session.issue(['code'], **self.options)
        with patch('context_stamps.session.time.monotonic', return_value=102):
            self.assertEqual(session.resolve(token, **self.options).status, 'insufficient')
        small = ContextSession(max_bytes=1)
        small.put(self.node)
        self.assertIsNone(small.issue(['code'], **self.options)[1])

    def test_wrong_versions_budget_and_bad_token(self):
        _, token, _ = self.session.issue(['code'], **self.options)
        self.assertEqual(self.session.resolve(token, role='dev', revisions={'code': 'v0'}).text, '')
        self.assertEqual(self.session.resolve(token, **self.options, budget_bytes=0).text, '')
        with self.assertRaises(ValueError):
            self.session.resolve(token[:-1], **self.options)
        for ttl in (0, float('nan'), float('inf'), 3601):
            with self.assertRaises(ValueError):
                ContextSession(ttl_seconds=ttl)

    def test_concurrent_reuse(self):
        with ThreadPoolExecutor(max_workers=4) as executor:
            results = list(executor.map(lambda _: self.session.issue(['code'], **self.options), range(40)))
        self.assertEqual(sum(not r[2] for r in results), 1)
        self.assertEqual(len({r[1] for r in results}), 1)


if __name__ == '__main__':
    unittest.main()

import random
import unittest

from context_stamps import ContextGraph, ContextNode, ContextSession, FacetQuery, Family, SphericalStamp


class PartialFacetTests(unittest.TestCase):
    def setUp(self):
        self.families = {name: Family('test', 4, 64, i) for i, name in enumerate(('content', 'entity', 'intent', 'task'))}
        self.full = SphericalStamp.encode({n: [1, 0, 0, 0] for n in self.families}, self.families)

    def test_partial_query_uses_only_observed_facets(self):
        query = FacetQuery(SphericalStamp((self.full.views[0],)))
        changed = SphericalStamp.encode({n: [1, 0, 0, 0] if n == 'content' else [-1, 0, 0, 0]
                                         for n in self.families}, self.families)
        self.assertEqual(query.score(changed), 1)
        self.assertEqual(query.compare(changed), {'content': 1})
        self.assertEqual(changed.bits, 256)

    def test_scope_changes_with_mask_weights_family_and_domain(self):
        first = FacetQuery(self.full)
        alternatives = [FacetQuery(SphericalStamp(self.full.views[:1])),
                        FacetQuery(self.full, (('content', 2), ('entity', 1), ('intent', 1), ('task', 1)))]
        self.assertNotEqual(first.policy_scope('a'), first.policy_scope('b'))
        for query in alternatives:
            self.assertNotEqual(first.policy_scope('a'), query.policy_scope('a'))
        family = Family('another-encoder', 4, 64, 0)
        different = FacetQuery(SphericalStamp.encode({'content': [1, 0, 0, 0]}, {'content': family}))
        self.assertNotEqual(alternatives[0].policy_scope('a'), different.policy_scope('a'))

    def test_values_change_but_same_policy_cohort(self):
        other = SphericalStamp.encode({n: [-1, 0, 0, 0] for n in self.families}, self.families)
        self.assertEqual(FacetQuery(self.full).policy_scope('a'), FacetQuery(other).policy_scope('a'))
        self.assertEqual(FacetQuery(self.full).score(other), 0)

    def test_missing_candidate_view_or_schema_is_rejected(self):
        with self.assertRaises(ValueError):
            FacetQuery(self.full).score(SphericalStamp(self.full.views[:1]))
        changed = dict(self.families, content=Family('other', 4, 64, 0))
        with self.assertRaises(ValueError):
            FacetQuery(self.full).score(SphericalStamp.encode({n: [1, 0, 0, 0] for n in changed}, changed))

    def test_invalid_weights(self):
        for weights in ((('content', 0),), (('content', float('nan')),), (('unknown', 1),),
                        tuple((n, 1e308) for n in self.families),
                        (('content', 1e-300), ('entity', 1e300), ('intent', 1), ('task', 1))):
            with self.assertRaises(ValueError):
                FacetQuery(self.full, weights)


class SelectiveInvalidationTests(unittest.TestCase):
    def test_unrelated_mutation_preserves_receipt(self):
        session = ContextSession()
        for key in ('a', 'b'):
            session.put(ContextNode(key, key, '1', frozenset({'r'})))
        options = dict(role='r', revisions={'a': '1', 'b': '1'})
        packet, token, _ = session.issue(['a'], **options)
        session.put(ContextNode('b', 'changed', '2', frozenset({'r'})))
        self.assertEqual(session.issue(['a'], **options), (packet, token, True))

    def test_new_dependency_and_conflict_invalidate_source(self):
        session = ContextSession()
        for key in ('a', 'b', 'c'):
            session.put(ContextNode(key, key, '1', frozenset({'r'})))
        options = dict(role='r', revisions={k: '1' for k in ('a', 'b', 'c')})
        _, old, _ = session.issue(['a'], **options)
        session.link('a', 'b', 'depends_on', provenance='p')
        self.assertEqual(session.resolve(old, **options).status, 'insufficient')
        packet, receipt, _ = session.issue(['a'], **options)
        self.assertEqual(packet.sources, ('a', 'b'))
        session.link('a', 'b', 'contradicts', provenance='p')
        self.assertEqual(session.resolve(receipt, **options).status, 'insufficient')
        self.assertEqual(session.issue(['a'], **options)[0].reason, 'conflicting_context')

    def test_same_revision_content_and_role_changes(self):
        session = ContextSession()
        options = dict(role='r', revisions={'a': '1'})
        session.put(ContextNode('a', 'old', '1', frozenset({'r'})))
        _, old, _ = session.issue(['a'], **options)
        session.put(ContextNode('a', 'new', '1', frozenset({'r'})))
        self.assertEqual(session.resolve(old, **options).status, 'insufficient')
        packet, current, _ = session.issue(['a'], **options)
        self.assertIn('new', packet.text)
        session.put(ContextNode('a', 'new', '1', frozenset({'admin'})))
        self.assertEqual(session.resolve(current, **options).status, 'insufficient')

    def test_random_mutations_match_uncached_graph(self):
        # Includes cycles, stale edges, new edges, access denial and unchanged revisions.
        for seed in range(10):
            rng = random.Random(seed)
            session, graph = ContextSession(max_entries=4), ContextGraph()
            revisions = {str(i): '1' for i in range(8)}
            for key in revisions:
                node = ContextNode(key, key, '1', frozenset({'r'}))
                session.put(node)
                graph.put(node)
            for step in range(50):
                a, b = rng.sample(list(revisions), 2)
                if rng.random() < .5:
                    node = ContextNode(a, f'value {step}', '1', frozenset({'r'} if rng.random() < .8 else {'admin'}))
                    session.put(node)
                    graph.put(node)
                else:
                    for store in (session, graph):
                        store.link(a, b, 'depends_on', provenance='random-test')
                for root in revisions:
                    expected = graph.handoff([root], role='r', revisions=revisions)
                    actual = session.issue([root], role='r', revisions=revisions)[0]
                    self.assertEqual(actual, expected)
            self.assertLessEqual(len(session._entries), 4)
            self.assertEqual(set().union(*session._by_source.values()) if session._by_source else set(),
                             set(session._entries))


if __name__ == '__main__':
    unittest.main()

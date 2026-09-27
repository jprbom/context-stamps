"""Target isolation, source partitioning and independent diffusion checks."""

import copy
import unittest

from multihop_data import public_input, select_groups
from relation_ranker import FEATURES, features, probabilities, rank


def case():
    return dict(id="fixture", question="Which Engine Service dependency keeps data?", answer="Cache Service",
                answerable=True, answer_aliases=[], question_decomposition=[{"answer": "do not expose"}],
                paragraphs=[dict(idx=0, title="Engine Service", paragraph_text="Engine Service calls Cache Service.", is_supporting=True),
                            dict(idx=1, title="Cache Service", paragraph_text="Cache Service keeps the data.", is_supporting=True),
                            dict(idx=2, title="Other Service", paragraph_text="An unrelated worker.", is_supporting=False)])


class IsolationTests(unittest.TestCase):
    def test_gold_mutations_cannot_change_public_input(self):
        row = case()
        changed = copy.deepcopy(row)
        changed.update(answer="malicious", answerable=False, answer_aliases=["secret"], question_decomposition=[])
        for p in changed["paragraphs"]:
            p["is_supporting"] = not p["is_supporting"]
        self.assertEqual(public_input(row), public_input(changed))
        self.assertEqual(set(public_input(row)), {"key", "question", "paragraphs"})

    def test_raw_targets_and_extra_fields_rejected_by_selector(self):
        with self.assertRaises(ValueError):
            features(case())
        clean = public_input(case())
        clean["paragraphs"][0]["is_supporting"] = True
        with self.assertRaises(ValueError):
            features(clean)

    def test_source_order_is_canonical_not_native_support_position(self):
        row = case()
        expected = public_input(row)
        row["paragraphs"].reverse()
        self.assertEqual(expected, public_input(row))
        row["paragraphs"][0]["idx"] = row["paragraphs"][1]["idx"]
        with self.assertRaises(ValueError):
            public_input(row)

    def test_source_filter_and_pairwise_disjoint_selection(self):
        def group(title, overlap=False):
            return dict(titles={title}, texts={title}, seeds={title}, overlap=overlap)
        groups = {"one": group("shared"), "two": group("shared"), "three": group("fresh"), "four": group("old", True)}
        chosen = select_groups(groups, name="fixture", count=2, separate_each=True)
        self.assertNotIn("four", chosen)
        self.assertIn("three", chosen)
        self.assertEqual(len(set().union(*(groups[k]["titles"] for k in chosen))), 2)
        with self.assertRaises(ValueError):
            select_groups(groups, name="fixture", count=3, separate_each=True)


class DiffusionTests(unittest.TestCase):
    def test_exact_two_node_recurrence(self):
        w = [1.]+[0.]*(len(FEATURES)-1)
        x = [[1000.]+[0.]*(len(FEATURES)-1), [0.]*len(FEATURES)]
        p = probabilities(x, [[.5, .5], [0., 1.]], w, .5)
        expected = 2/3+(1/3)*(.25**4)
        self.assertAlmostEqual(p[0], expected)
        self.assertAlmostEqual(sum(p), 1.)

    def test_attention_mass_is_conserved_and_zero_alpha_matches_restart(self):
        clean = public_input(case())
        x, p, _ = features(clean)
        for row in p:
            self.assertAlmostEqual(sum(row), 1.)
        w = [float(i)/10 for i in range(len(FEATURES))]
        q = probabilities(x, p, w, 0.)
        for alpha in (.1, .5, .9):
            result = probabilities(x, p, w, alpha)
            self.assertAlmostEqual(sum(result), 1.)
            self.assertTrue(all(v >= 0 for v in result))
        self.assertEqual(q, probabilities(x, p, w, .9, steps=0))

    def test_permuting_candidates_preserves_ranking(self):
        clean = public_input(case())
        model = dict(features=list(FEATURES), steps=4, weights=[1.]*len(FEATURES), alpha=.6)
        before, _ = rank(clean, model)
        clean["paragraphs"].reverse()
        after, _ = rank(clean, model)
        self.assertEqual([p["sha256"] for p in before], [p["sha256"] for p in after])


if __name__ == "__main__":
    unittest.main()

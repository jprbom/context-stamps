import importlib.util
import tempfile
import unittest
from pathlib import Path

from context_stamps import ContextMemory
from context_stamps.selection import LinearSelector, select_evidence
from context_stamps.sources import explain_versions, observe_files


class SelectionTests(unittest.TestCase):
    def test_external_candidate_selector_coverage_and_zero_weight(self):
        from context_stamps.selection import rank_candidates

        documents = ["timeout", "timeout timeout", "retries"]
        scores = [0.9, 0.89, 0.8]
        self.assertEqual(
            rank_candidates(
                "timeout retries", documents, scores, limit=2, coverage_weight=0, diversity_weight=0
            ),
            [0, 1],
        )
        self.assertEqual(
            rank_candidates(
                "timeout retries", documents, scores, limit=2, coverage_weight=0.3, diversity_weight=0
            ),
            [0, 2],
        )
        with self.assertRaises(ValueError):
            rank_candidates("x", ["x"], [float("nan")])

    def test_required_evidence_cannot_be_silently_omitted(self):
        with ContextMemory() as memory:
            item = memory.add("essential fact", source="essential")
            revisions = {"essential": item["digest"]}
            good = select_evidence(memory, "fact", revisions=revisions, required=["essential"])
            self.assertEqual(good.status, "current")
            for budget, required in [(1, ["essential"]), (1000, ["missing"])]:
                packet = select_evidence(
                    memory, "fact", revisions=revisions, required=required, budget=budget
                )
                self.assertEqual(packet.status, "insufficient_evidence")
                self.assertFalse(packet.text)
            memory.invalidate("essential")
            packet = select_evidence(memory, "fact", revisions=revisions, required=["essential"])
            self.assertEqual(packet.missing_required, ["essential"])

    def test_duplicate_required_sources_and_token_counter(self):
        with ContextMemory() as memory:
            memory.add("identical", source="a")
            memory.add("identical", source="b")
            packet = select_evidence(memory, "identical", required=["a", "b"], counter=len)
            self.assertEqual(set(packet.selected), {"a", "b"})
            self.assertEqual(packet.units, len(packet.text))
            with self.assertRaises(ValueError):
                select_evidence(memory, "identical", counter=lambda x: -1)

    def test_reranker_cannot_override_freshness(self):
        with ContextMemory() as memory:
            memory.add("stale", source="a")
            memory.add("current", source="b")
            memory.invalidate("a")
            result = select_evidence(memory, "stale", reranker=lambda *x: 1)
            self.assertNotIn("a", result.selected)

    def test_observation_update_missing_and_escape(self):
        with tempfile.TemporaryDirectory() as folder, ContextMemory() as memory:
            root = Path(folder)
            file = root / "source.txt"
            file.write_text("first")
            first = observe_files(memory, root, ["source.txt"])
            memory.add("derived", source="plan", dependencies=first["revisions"])
            file.write_text("changed")
            second = observe_files(memory, root, ["source.txt"])
            explanation = explain_versions(memory, "plan", second["revisions"])
            self.assertEqual(explanation["status"], "stale")
            self.assertTrue(any(c["source"] == "source.txt" for c in explanation["changes"]))
            file.unlink()
            missing = observe_files(memory, root, ["source.txt"])
            self.assertEqual(missing["revisions"], {})
            self.assertTrue(memory.get("source.txt")["stale"])
            for escape in ["../source.txt", "C:/secret", "/etc/passwd"]:
                with self.assertRaises(ValueError):
                    observe_files(memory, root, [escape])

    def test_selector_json_rejects_nonfinite_and_wrong_schema(self):
        with self.assertRaises(ValueError):
            LinearSelector([float("nan")] * 5, 0)
        with self.assertRaises(ValueError):
            LinearSelector([0] * 5, 0, 2)

    @unittest.skipUnless(importlib.util.find_spec("numpy"), "NumPy optional")
    def test_training_and_diallel_identities(self):
        import numpy as np

        from context_stamps.baselines import corpus_mean_similarity
        from context_stamps.selection import fit_selector

        examples = [("alpha", "alpha", 1, 1), ("alpha", "beta", 0, 0)] * 20
        model, trace = fit_selector(examples)
        self.assertGreater(model.score("alpha", "alpha", 1), model.score("alpha", "beta", 0))
        self.assertLess(trace[-1]["loss"], trace[0]["loss"])
        d = np.random.default_rng(4).normal(size=(8, 5))
        gram = d @ d.T
        np.testing.assert_allclose(corpus_mean_similarity(d), (gram.sum(1) - gram.diagonal()) / 7)
        m = np.random.default_rng(5).normal(size=(7, 8))
        gca = m.mean(0)
        sca = m - m.mean(1)[:, None] - gca + m.mean()
        actual = 0.6 * m + 0.2 * gca + 0.2 * sca
        expected = 0.8 * m - 0.2 * m.mean(1)[:, None] + 0.2 * m.mean()
        np.testing.assert_allclose(actual, expected)


if __name__ == "__main__":
    unittest.main()

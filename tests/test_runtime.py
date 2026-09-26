"""Executable workflow boundaries and failures, using trusted local adapters."""

import hashlib
import unittest
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from unittest.mock import patch

from context_stamps import ContextExpert, ContextNode, ContextRuntime, RuntimeBudget, Verification
from context_stamps.expert_router import LinearExpertRouter


class RuntimeTests(unittest.TestCase):
    def setUp(self):
        self.runtime = ContextRuntime(tenant="lab", principal="p", role="reader", policy="v1", token_counter=lambda s: len(s.split()))
        for key in ("config", "metric", "dataset"):
            self.runtime.put(ContextNode(key, key + " = 32", "v1", frozenset({"reader"})))
        self.expert = ContextExpert("baseline", lambda r: ["metric"], 1)

    def prepare(self, **kwargs):
        opts = dict(eligible=["config", "metric", "dataset"], experts=[self.expert], baseline="baseline", scope="test",
                    verifier=lambda p: Verification(True), limit=1)
        opts.update(kwargs)
        return self.runtime.prepare_context("compare metric", **opts)

    def test_bounded_missing_evidence_recovery(self):
        result = self.prepare(verifier=lambda p: Verification("config" in p.sources, () if "config" in p.sources else ("config",)))
        self.assertEqual((result.status, result.iterations, result.sources), ("complete", 2, ("config", "metric")))
        self.assertEqual(len(result.receipt), 32)
        one = self.prepare(budget=RuntimeBudget(iterations=1), verifier=lambda p: Verification(False, ("config",)))
        self.assertEqual(one.reason, "iteration_budget")

    def test_dependency_closure_revoke_and_revalidation(self):
        self.runtime.link("metric", "dataset", "depends_on", provenance="test")
        result = self.prepare()
        self.assertEqual(result.sources, ("dataset", "metric"))
        self.runtime.invalidate("dataset")
        self.assertEqual(self.runtime.resolve(result.receipt).status, "insufficient")
        self.assertEqual(self.prepare().text, "")
        self.runtime.put(ContextNode("dataset", "new data", "v2", frozenset({"reader"})))
        self.assertEqual(self.prepare().status, "insufficient")
        self.runtime.link("metric", "dataset", "depends_on", provenance="review-v2")
        self.assertEqual(self.prepare().status, "complete")

    def test_token_budget_counts_serialized_provenance(self):
        full = self.prepare()
        limited = self.prepare(budget=RuntimeBudget(tokens=full.tokens - 1))
        self.assertEqual((limited.reason, limited.text), ("token_budget", ""))
        self.assertEqual(self.prepare(budget=RuntimeBudget(tokens=full.tokens)).status, "complete")
        self.assertEqual(self.runtime.resolve(full.receipt, budget=RuntimeBudget(tokens=1)).status, "insufficient")

    def test_unauthorized_expert_or_verifier_cannot_disclose(self):
        self.runtime.put(ContextNode("secret", "password", "v1", frozenset({"admin"})))
        expert = ContextExpert("baseline", lambda r: ["secret"], 1)
        self.assertEqual(self.prepare(experts=[expert]).reason, "invalid_expert_result")
        self.assertEqual(self.prepare(verifier=lambda p: Verification(False, ("secret",))).text, "")
        self.assertEqual(self.prepare(exact_key="secret").text, "")
        self.runtime.link("metric", "secret", "depends_on", provenance="test")
        self.assertEqual(self.prepare().text, "")

    def test_router_never_downgrades_unqualified_or_over_budget(self):
        cheap = ContextExpert("cheap", lambda r: ["config"], .01)
        result = self.prepare(experts=[self.expert, cheap], choose=lambda r: "cheap")
        self.assertEqual(result.expert, "baseline")
        approved = replace(cheap, approved_scopes=("test",))
        self.assertEqual(self.prepare(experts=[self.expert, approved], choose=lambda r: "cheap").expert, "cheap")
        self.assertEqual(self.prepare(budget=RuntimeBudget(milliseconds=.1)).reason, "latency_budget")

    def test_iteration_stall_mutation_and_deadline(self):
        self.assertEqual(self.prepare(verifier=lambda p: Verification(False)).reason, "no_progress")
        def mutate(packet):
            self.runtime.invalidate("metric")
            return Verification(True)
        self.assertEqual(self.prepare(verifier=mutate).reason, "context_changed")
        self.setUp()
        with patch("context_stamps.runtime.time.monotonic", side_effect=[0, 0, 2]):
            self.assertEqual(self.prepare().reason, "latency_budget")

    def test_verified_reuse_invalidates_and_does_not_cache_failed_output(self):
        calls = []
        args = dict(request_digest=hashlib.sha256(b"seed=7").hexdigest(), model="m1", prompt="p1", tool="extract",
                    verifier_revision="v1", compute=lambda text: (calls.append(text) or "32"), verify=lambda value, text: value == "32")
        prepared = self.prepare()
        self.assertFalse(self.runtime.run_verified(prepared, **args)["reused"])
        self.assertTrue(self.runtime.run_verified(prepared, **args)["reused"])
        self.assertEqual(len(calls), 1)
        self.assertFalse(self.runtime.run_verified(prepared, **dict(args, model="m2"))["reused"])
        self.assertEqual(self.runtime.run_verified(replace(prepared, text="forged"), **args)["result"], "")
        self.runtime.invalidate("metric")
        self.assertEqual(self.runtime.run_verified(prepared, **args)["result"], "")
        self.setUp()
        prepared = self.prepare()
        bad = dict(args, compute=lambda text: "wrong", verify=lambda value, text: False)
        self.assertEqual(self.runtime.run_verified(prepared, **bad)["status"], "unverified")
        self.assertFalse(self.runtime.run_verified(prepared, **args)["reused"])

    def test_concurrent_prepare_and_principal_receipt_isolation(self):
        with ThreadPoolExecutor(max_workers=4) as pool:
            values = list(pool.map(lambda _: self.prepare(), range(20)))
        self.assertTrue(all(r.status == "complete" for r in values))
        self.assertEqual(len({r.receipt for r in values}), 1)
        other = ContextRuntime(tenant="elsewhere", principal="q", role="reader", policy="v1")
        self.assertEqual(other.resolve(values[0].receipt).status, "insufficient")

    def test_no_implicit_token_estimate(self):
        with self.assertRaises(ValueError):
            self.runtime.invalidate("unknown")
        self.runtime._tokens = None
        with self.assertRaises(ValueError):
            self.prepare(budget=RuntimeBudget(tokens=100))
        for value in (True, float("nan"), -1):
            with self.assertRaises(ValueError):
                RuntimeBudget(milliseconds=value)


class RouterTests(unittest.TestCase):
    def test_quantized_decision_and_boundary_match_fp64(self):
        router = LinearExpertRouter((0., 0.), (1., 1.), (.0123456789, -.987654321), .1, .2, ("scope",))
        for i in range(-500, 501):
            features = [i / 100, -i / 113]
            exact = router.intercept + sum(w * x for w, x in zip(router.weights, features))
            decision, _ = router.choose(features, scope="scope")
            self.assertEqual(decision, "hybrid" if exact >= router.threshold else "fusion")
        edge = replace(router, threshold=router.intercept)
        self.assertEqual(edge.predict([0., 0.])[1], "boundary_fp64")
        self.assertEqual(router.choose([1., 1.], scope="unknown"), ("fusion", "scope_unqualified"))
        self.assertEqual(router.predict([100., 0.])[1], "out_of_training_range")


if __name__ == "__main__":
    unittest.main()

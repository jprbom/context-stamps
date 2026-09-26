"""Callback isolation, revocation and bounded concurrent computation.

Copyright (c) 2026 Prashant Jagtap. MIT License.
Events assert ordering; timeouts bound a failed test rather than measure speed.
"""

import hashlib
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch

from context_stamps import ContextExpert, ContextNode, ContextRuntime, RuntimeBudget, Verification


class RuntimeCallbackTests(unittest.TestCase):
    def setUp(self):
        self.runtime = ContextRuntime(tenant="lab", principal="user", role="reader", policy="v1")
        self.runtime.put(ContextNode("fact", "validated fact", "v1", frozenset({"reader"})))

    def prepare(self, **kwargs):
        options = dict(eligible=["fact"], experts=[ContextExpert("base", lambda request: ["fact"], 0)],
                       baseline="base", scope="test", verifier=lambda packet: Verification(True),
                       budget=RuntimeBudget(milliseconds=10000))
        options.update(kwargs)
        return self.runtime.prepare_context("fact", **options)

    def execute(self, prepared, **kwargs):
        options = dict(request_digest=hashlib.sha256(b"request").hexdigest(), model="model", prompt="prompt",
                       tool="tool", verifier_revision="v1", compute=lambda text: "answer",
                       verify=lambda answer, text: answer == "answer")
        options.update(kwargs)
        return self.runtime.run_verified(prepared, **options)

    def assert_revocable(self, operation, entered, release):
        with ThreadPoolExecutor(max_workers=2) as pool:
            future = pool.submit(operation)
            try:
                self.assertTrue(entered.wait(5), "callback did not start")
                # On the old runtime this cannot complete until release is set.
                pool.submit(self.runtime.invalidate, "fact").result(timeout=5)
            finally:
                release.set()
            result = future.result(timeout=5)
        self.assertFalse(self.runtime.outcomes())
        return result

    @staticmethod
    def blocker(entered, release, value):
        def callback(*args):
            entered.set()
            if not release.wait(10):
                raise TimeoutError("test callback was not released")
            return value
        return callback

    def test_prepare_callbacks_allow_revocation_and_reject_stale_packets(self):
        for stage in ("choose", "retrieve", "token_count", "verify"):
            with self.subTest(stage=stage):
                self.setUp()
                entered, release = threading.Event(), threading.Event()
                options = {}
                if stage == "choose":
                    options["choose"] = self.blocker(entered, release, "base")
                elif stage == "retrieve":
                    options["experts"] = [ContextExpert("base", self.blocker(entered, release, ["fact"]), 0)]
                elif stage == "token_count":
                    self.runtime._tokens = self.blocker(entered, release, 1)
                else:
                    options["verifier"] = self.blocker(entered, release, Verification(True))
                result = self.assert_revocable(lambda: self.prepare(**options), entered, release)
                self.assertEqual((result.status, result.reason, result.text), ("insufficient", "context_changed", ""))

    def test_resolve_tokenizer_allows_revocation(self):
        prepared = self.prepare()
        entered, release = threading.Event(), threading.Event()
        self.runtime._tokens = self.blocker(entered, release, 1)
        result = self.assert_revocable(
            lambda: self.runtime.resolve(prepared.receipt, budget=RuntimeBudget(tokens=100, milliseconds=10000)),
            entered, release)
        self.assertEqual((result.reason, result.text), ("context_changed", ""))

    def test_model_callbacks_allow_revocation_without_caching(self):
        for stage in ("compute", "verify", "cached_verify"):
            with self.subTest(stage=stage):
                self.setUp()
                prepared = self.prepare()
                if stage == "cached_verify":
                    self.execute(prepared)
                    self.runtime._outcomes.clear()
                entered, release = threading.Event(), threading.Event()
                option = {"compute": self.blocker(entered, release, "answer")} if stage == "compute" else {
                    "verify": self.blocker(entered, release, True)}
                result = self.assert_revocable(lambda: self.execute(prepared, **option), entered, release)
                self.assertEqual(result, {"status": "unverified", "result": "", "reused": False})
                self.assertFalse(self.runtime._pending)

    def test_unrelated_computation_can_finish_while_another_is_blocked(self):
        prepared = self.prepare()
        entered, release = threading.Event(), threading.Event()
        with ThreadPoolExecutor(max_workers=2) as pool:
            first = pool.submit(self.execute, prepared, compute=self.blocker(entered, release, "answer"))
            try:
                self.assertTrue(entered.wait(5))
                second = pool.submit(self.execute, prepared, model="other").result(timeout=5)
                self.assertEqual(second["status"], "verified")
            finally:
                release.set()
            self.assertEqual(first.result(timeout=5)["status"], "verified")

    def test_same_identity_coalesces_and_each_caller_verifies(self):
        prepared = self.prepare()
        entered, release, waiting = threading.Event(), threading.Event(), threading.Event()
        compute = self.blocker(entered, release, "answer")
        real_wait = self.runtime._changed.wait
        verified = []
        def observe_wait(*args, **kwargs):
            waiting.set()
            return real_wait(*args, **kwargs)
        def verify(answer, text):
            verified.append(answer)
            return True
        def unexpected(text):
            self.fail("a concurrent exact request duplicated computation")
        with patch.object(self.runtime._changed, "wait", side_effect=observe_wait), ThreadPoolExecutor(max_workers=2) as pool:
            first = pool.submit(self.execute, prepared, compute=compute, verify=verify)
            try:
                self.assertTrue(entered.wait(5))
                second = pool.submit(self.execute, prepared, compute=unexpected, verify=verify)
                self.assertTrue(waiting.wait(5))
            finally:
                release.set()
            self.assertFalse(first.result(timeout=5)["reused"])
            self.assertTrue(second.result(timeout=5)["reused"])
        self.assertEqual(verified, ["answer", "answer"])
        self.assertFalse(self.runtime._pending)

    def test_waiter_wakes_on_revocation_before_model_returns(self):
        prepared = self.prepare()
        entered, release, waiting = threading.Event(), threading.Event(), threading.Event()
        real_wait = self.runtime._changed.wait
        def observe_wait(*args, **kwargs):
            waiting.set()
            return real_wait(*args, **kwargs)
        with patch.object(self.runtime._changed, "wait", side_effect=observe_wait), ThreadPoolExecutor(max_workers=2) as pool:
            first = pool.submit(self.execute, prepared, compute=self.blocker(entered, release, "answer"))
            try:
                self.assertTrue(entered.wait(5))
                second = pool.submit(self.execute, prepared)
                self.assertTrue(waiting.wait(5))
                self.runtime.invalidate("fact")
                self.assertEqual(second.result(timeout=5)["result"], "")
                self.assertFalse(first.done())
            finally:
                release.set()
            self.assertEqual(first.result(timeout=5)["result"], "")

    def test_exception_cleans_pending_and_permits_retry(self):
        prepared = self.prepare()
        for stage in ("compute", "verify"):
            with self.subTest(stage=stage):
                def broken(*args):
                    raise RuntimeError("adapter failure")
                with self.assertRaisesRegex(RuntimeError, "adapter failure"):
                    self.execute(prepared, model=stage, **{stage: broken})
                self.assertFalse(self.runtime._pending)
                self.assertEqual(self.execute(prepared, model=stage)["status"], "verified")

    def test_owner_exception_releases_waiting_request(self):
        prepared = self.prepare()
        entered, release, waiting = threading.Event(), threading.Event(), threading.Event()
        real_wait = self.runtime._changed.wait
        def observe_wait(*args, **kwargs):
            waiting.set()
            return real_wait(*args, **kwargs)
        def broken(text):
            self.blocker(entered, release, None)()
            raise RuntimeError("adapter failed")
        with patch.object(self.runtime._changed, "wait", side_effect=observe_wait), ThreadPoolExecutor(max_workers=2) as pool:
            first = pool.submit(self.execute, prepared, compute=broken)
            try:
                self.assertTrue(entered.wait(5))
                second = pool.submit(self.execute, prepared)
                self.assertTrue(waiting.wait(5))
            finally:
                release.set()
            with self.assertRaisesRegex(RuntimeError, "adapter failed"):
                first.result(timeout=5)
            self.assertEqual(second.result(timeout=5), {"status": "verified", "result": "answer", "reused": False})
        self.assertFalse(self.runtime._pending)

    def test_recursive_identity_is_rejected_without_deadlock(self):
        prepared = self.prepare()
        with self.assertRaisesRegex(ValueError, "recursive computation"):
            self.execute(prepared, compute=lambda text: self.execute(prepared))
        self.assertFalse(self.runtime._pending)
        self.assertEqual(self.execute(prepared)["status"], "verified")

    def test_inflight_bound_abstains_and_recovers(self):
        self.runtime._max_inflight = 1
        prepared = self.prepare()
        entered, release = threading.Event(), threading.Event()
        with ThreadPoolExecutor(max_workers=1) as pool:
            first = pool.submit(self.execute, prepared, compute=self.blocker(entered, release, "answer"))
            try:
                self.assertTrue(entered.wait(5))
                self.assertEqual(self.execute(prepared, model="other")["status"], "insufficient")
            finally:
                release.set()
            self.assertEqual(first.result(timeout=5)["status"], "verified")
        self.assertEqual(self.execute(prepared, model="other")["status"], "verified")
        for value in (0, 257, True, 1.0):
            with self.assertRaises(ValueError):
                ContextRuntime(tenant="lab", principal="user", role="reader", policy="v1", max_inflight=value)

    def test_receipt_expiry_during_compute_or_verifier_rejects_output(self):
        prepared = self.prepare()
        for stage in ("compute", "verify"):
            with self.subTest(stage=stage):
                def expire(*args):
                    # Advance only the session clock while resolving after the
                    # callback. Do not wait for the default five minute TTL.
                    clock.start()
                    return "answer" if stage == "compute" else True
                clock = patch("context_stamps.session.time.monotonic", return_value=10**12)
                try:
                    self.assertEqual(self.execute(prepared, model=stage, **{stage: expire})["result"], "")
                finally:
                    clock.stop()
                self.assertFalse(self.runtime._pending)
                prepared = self.prepare()


if __name__ == "__main__":
    unittest.main()

import hashlib
import unittest
from dataclasses import replace
from unittest.mock import patch

from context_stamps.computation import ComputationCache, ComputationIdentity, exact_decimal


def identity():
    digest = hashlib.sha256(b"original").hexdigest()
    return ComputationIdentity("tenant-a", "reader", "model-v1", "prompt-v1", "tool-v1",
                               "policy-v1", digest, (("source", "r1", digest),))


class ComputationTests(unittest.TestCase):
    def test_every_computation_input_invalidates(self):
        cache, original = ComputationCache(), identity()
        cache.put(original, "verified result")
        self.assertEqual(cache.get(original), "verified result")
        for field in ("tenant", "principal", "model", "prompt", "tool", "policy"):
            self.assertIsNone(cache.get(replace(original, **{field: "changed"})))
        for changed in (replace(original, request="0" * 64),
                        replace(original, sources=(("source", "r2", original.request),)),
                        replace(original, sources=(("source", "r1", "0" * 64),))):
            self.assertIsNone(cache.get(changed))

    def test_expiry_capacity_and_bytes(self):
        a, b = identity(), replace(identity(), principal="another")
        with patch("context_stamps.computation.time.monotonic", return_value=0):
            cache = ComputationCache(capacity=1, ttl_seconds=1, max_bytes=5)
            cache.put(a, "first")
            cache.put(b, "next")
            self.assertIsNone(cache.get(a))
            self.assertEqual(cache.get(b), "next")
            with self.assertRaises(ValueError):
                cache.put(a, "too long")
        with patch("context_stamps.computation.time.monotonic", return_value=1):
            self.assertIsNone(cache.get(b))

    def test_rejects_ambiguous_bindings_and_limits(self):
        for sources in ([identity().sources[0]], identity().sources * 2, (), (("source", "r", "bad"),)):
            with self.assertRaises(ValueError):
                replace(identity(), sources=sources)
        for kwargs in (dict(capacity=True), dict(ttl_seconds=float("nan")), dict(max_bytes=0)):
            with self.assertRaises(ValueError):
                ComputationCache(**kwargs)

    def test_canonical_source_order(self):
        source = identity().sources[0]
        second = ("other", "v1", "0" * 64)
        self.assertEqual(replace(identity(), sources=(source, second)).digest,
                         replace(identity(), sources=(second, source)).digest)

    def test_exact_numbers_reject_rounding_and_code(self):
        self.assertEqual(exact_decimal("add", "0.1", "0.2"), "0.3")
        self.assertEqual(exact_decimal("compare", "10000000000000001", "10000000000000000"), "1")
        self.assertEqual(exact_decimal("divide", "1", "8"), "0.125")
        for operation, a, b in (("divide", "1", "3"), ("divide", "1", "0"),
                                ("add", "NaN", "1"), ("add", "1e999", "1"),
                                ("add", "__import__('os')", "1"), ("execute", "1", "1")):
            with self.assertRaises(ValueError):
                exact_decimal(operation, a, b)

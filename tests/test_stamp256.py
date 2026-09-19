import unittest

from context_stamps.stamp256 import Stamp256Codec
from stamps import Family, HashingEncoder


class Stamp256Tests(unittest.TestCase):
    def test_mixed_width_roundtrip(self):
        encoder = HashingEncoder(64)
        widths = {"content": 160, "entity": 32, "intent": 32, "task": 32}
        codec = Stamp256Codec({n: Family(encoder.identity, 64, b, 17 + i) for i, (n, b) in enumerate(widths.items())})
        stamp = codec.encode({n: encoder.encode(n + " example") for n in widths})
        raw = codec.pack(stamp)
        self.assertEqual(len(raw), 32)
        self.assertEqual(stamp.bits, 256)
        self.assertEqual(codec.unpack(raw, schema_id=codec.schema.identity), stamp)
        for value in (raw[:-1], raw + b"x", raw.hex(), bytearray(raw)):
            with self.assertRaises(ValueError):
                codec.unpack(value, schema_id=codec.schema.identity)
        with self.assertRaises(ValueError):
            codec.unpack(raw, schema_id="another-schema")

    def test_budget_and_family_mismatch(self):
        with self.assertRaises(ValueError):
            Stamp256Codec({"content": Family("test", 4, 128, 1)})
        first = Stamp256Codec({"content": Family("test", 4, 256, 1)})
        second = Stamp256Codec({"content": Family("test", 4, 256, 2)})
        with self.assertRaises(ValueError):
            first.pack(second.encode({"content": [1, 0, 0, 0]}))


if __name__ == "__main__":
    unittest.main()

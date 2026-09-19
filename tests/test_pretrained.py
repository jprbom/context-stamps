import json
import unittest
from pathlib import Path

from context_stamps import Family, HashingEncoder, SphericalStamp, load_experimental_model


class PretrainedTests(unittest.TestCase):
    def test_packaged_weights_match_recorded_training(self):
        encoder = HashingEncoder(64)
        facets = {"content": "measurement", "entity": "worker", "intent": "review", "task": "timeout"}
        for seed in (17, 41, 83):
            model = load_experimental_model(seed=seed)
            expected = json.loads((Path(__file__).resolve().parents[1] /
                                   f"evidence/pairwise-v1/model-{seed}.json").read_text())
            self.assertEqual(json.loads(model.to_json()), expected)
            stamp = SphericalStamp.encode({k: encoder.encode(v) for k, v in facets.items()},
                {n: Family(encoder.identity, 64, 64, seed + i) for i, n in enumerate(sorted(facets))})
            self.assertGreater(model.score(stamp, stamp), 0)
        with self.assertRaises(ValueError):
            load_experimental_model(seed="../../outside")

"""Optional pair-token preparation and bounded cache tests without downloads."""

import unittest
from types import SimpleNamespace

try:
    import torch
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "optional torch dependency")
class PairTests(unittest.TestCase):
    def test_longest_first_ties_and_truncation(self):
        from context_stamps.efficient_reranker import pair_tokens
        for a, b, qa, pa in ((0, 0, 0, 0), (1000, 2, 507, 2), (3, 1000, 3, 506),
                             (600, 600, 254, 255), (600, 500, 255, 254),
                             (500, 600, 254, 255), (510, 511, 254, 255), (254, 255, 254, 255)):
            q, p = list(range(a)), list(range(b))
            ids, types = pair_tokens(q, p, cls_id=101, sep_id=102)
            self.assertEqual(ids, [101, *q[:qa], 102, *p[:pa], 102])
            self.assertEqual(len(types), len(ids))
        with self.assertRaises(ValueError):
            pair_tokens([], [], cls_id=101, sep_id=102, max_length=1024)

    def test_cache_content_binding_eviction_and_order(self):
        from context_stamps.efficient_reranker import EfficientReranker
        class Tokenizer:
            padding_side = truncation_side = "right"
            cls_token_id, sep_token_id, pad_token_id = 101, 102, 0
            calls = 0
            pair_calls = 0

            def num_special_tokens_to_add(self, pair):
                return 3

            def __call__(self, text, passages=None, **kwargs):
                self.calls += 1
                if passages is not None:
                    from context_stamps.efficient_reranker import pair_tokens
                    self.pair_calls += 1
                    pairs = [pair_tokens([ord(c) % 97 for c in q], [ord(c) % 97 for c in p],
                        cls_id=101, sep_id=102) for q, p in zip(text, passages)]
                    return {"input_ids": [p[0] for p in pairs], "token_type_ids": [p[1] for p in pairs]}
                def encode(s):
                    return [ord(c) % 97 for c in s][:512]
                return {"input_ids": encode(text) if isinstance(text, str) else [encode(t) for t in text]}
        class Model(torch.nn.Module):
            config = SimpleNamespace(model_type="bert")
            device = torch.device("cpu")

            def forward(self, input_ids, attention_mask, token_type_ids):
                return SimpleNamespace(logits=(input_ids * attention_mask).float().sum(1, keepdim=True))
        tok = Tokenizer()
        scorer = EfficientReranker(tok, Model(), precision="fp16", cache_entries=2, cache_bytes=64)
        first = scorer.score("q", ["aaa", "b"])
        second = scorer.score("q", ["b", "aaa"])
        self.assertEqual(first.tolist(), second[::-1].tolist())
        scorer.score("q", ["changed", "third"])
        self.assertLessEqual(scorer._bytes, 64)
        self.assertLessEqual(len(scorer._cache), 2)
        scorer.clear_cache()
        self.assertEqual(scorer._bytes, 0)
        scorer.score("q" * 600, ["p" * 600])
        self.assertEqual(tok.pair_calls, 1)
        self.assertEqual(scorer._bytes, 0)
        with self.assertRaises(ValueError):
            scorer.score("q", ["x"] * 257)


if __name__ == "__main__":
    unittest.main()

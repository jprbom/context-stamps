"""Optional bounded BERT pair preparation and length-aware GPU inference.

Copyright (c) 2026 Prashant Jagtap. MIT License.
This changes execution, not the candidate set or the 512-token evidence limit.
"""

import hashlib
import threading
from array import array
from collections import OrderedDict

import numpy as np
import torch


def pair_tokens(query, passage, *, cls_id, sep_id, max_length=512):
    """BERT longest-first lengths; inputs must contain the full token sequences.

    The original longer sequence retains the odd token; equal original lengths
    give that token to the passage. Clipping both inputs first loses this fact.
    """
    if type(max_length) is not int or not 8 <= max_length <= 512:
        raise ValueError("max_length must be between 8 and 512")
    q, p = len(query), len(passage)
    budget = max_length - 3
    if q + p > budget:
        if min(q, p) <= budget // 2:
            if q < p:
                p = budget - q
            else:
                q = budget - p
        else:
            q, p = ((budget + 1) // 2, budget // 2) if q > p else (budget // 2, (budget + 1) // 2)
    return [cls_id, *query[:q], sep_id, *passage[:p], sep_id], [0] * (q + 2) + [1] * (p + 1)


class EfficientReranker:
    """Own a pinned BERT tokenizer/model; cache only exact passage token IDs.

    Hosts must filter unauthorized documents before calling. Cache keys bind
    exact UTF-8 content; caches belong to this instance and tokenizer. No
    approximate cache hit or score reuse. Concurrent calls are serialized.
    """

    def __init__(self, tokenizer, model, *, precision="autocast_bf16", batch_size=64,
                 cache_bytes=32 * 1024 * 1024, cache_entries=20000, bucket=True):
        if precision not in ("autocast_bf16", "fp16", "bf16"):
            raise ValueError("unsupported precision")
        if (type(batch_size) is not int or not 1 <= batch_size <= 128
                or type(cache_bytes) is not int or not 0 <= cache_bytes <= 256 * 1024 * 1024
                or type(cache_entries) is not int or not 0 <= cache_entries <= 100000
                or type(bucket) is not bool):
            raise ValueError("invalid resource bounds")
        if (model.config.model_type != "bert" or tokenizer.padding_side != "right"
                or tokenizer.truncation_side != "right" or tokenizer.num_special_tokens_to_add(pair=True) != 3
                or any(type(x) is not int for x in (tokenizer.cls_token_id, tokenizer.sep_token_id, tokenizer.pad_token_id))):
            raise ValueError("only explicit right-padded BERT pair encoding is supported")
        self.tokenizer, self.model = tokenizer, model
        self.precision, self.batch_size, self.bucket = precision, batch_size, bucket
        self.cache_bytes, self.cache_entries = cache_bytes, cache_entries
        self._cache, self._bytes, self._lock = OrderedDict(), 0, threading.RLock()
        if precision != "autocast_bf16":
            self.model.to(dtype=torch.float16 if precision == "fp16" else torch.bfloat16)
        self.model.eval()

    def clear_cache(self):
        with self._lock:
            self._cache.clear()
            self._bytes = 0

    def _passages(self, passages):
        keys = [hashlib.sha256(p.encode("utf-8")).digest() for p in passages]
        found = [self._cache.get(key) for key in keys]
        missing = [i for i, value in enumerate(found) if value is None]
        if missing:
            encoded = self.tokenizer([passages[i] for i in missing], add_special_tokens=False,
                                     truncation=True, max_length=512)["input_ids"]
            for i, value in zip(missing, encoded):
                found[i] = array("I", value)
        for key, value in zip(keys, found):
            if key in self._cache:
                self._cache.move_to_end(key)
                continue
            size = len(value) * value.itemsize + len(key)
            if self.cache_entries == 0 or size > self.cache_bytes:
                continue
            while self._cache and (len(self._cache) >= self.cache_entries or self._bytes + size > self.cache_bytes):
                old_key, old = self._cache.popitem(last=False)
                self._bytes -= len(old) * old.itemsize + len(old_key)
            self._cache[key] = value
            self._bytes += size
        return found

    @torch.inference_mode()
    def score(self, query, passages):
        if (not isinstance(query, str) or len(query.encode("utf-8")) > 65536
                or not isinstance(passages, (list, tuple)) or not 1 <= len(passages) <= 256
                or any(not isinstance(p, str) or len(p.encode("utf-8")) > 1048576 for p in passages)
                or sum(len(p.encode("utf-8")) for p in passages) > 16 * 1024 * 1024):
            raise ValueError("bounded query and one to 256 passages required")
        with self._lock:
            q = self.tokenizer(query, add_special_tokens=False, truncation=True, max_length=512)["input_ids"]
            if len(q) > 254:
                # Original sequence lengths affect longest-first ties. Delegate
                # long pairs to the pinned tokenizer instead of cached truncation.
                encoded = self.tokenizer([query] * len(passages), passages, padding=False,
                                         truncation=True, max_length=512)
                pairs = list(zip(encoded["input_ids"], encoded["token_type_ids"]))
            else:
                pairs = [pair_tokens(q, p, cls_id=self.tokenizer.cls_token_id, sep_id=self.tokenizer.sep_token_id)
                         for p in self._passages(passages)]
            order = sorted(range(len(pairs)), key=lambda i: (len(pairs[i][0]), i)) if self.bucket else list(range(len(pairs)))
            scores = np.empty(len(pairs), dtype=np.float32)
            for start in range(0, len(order), self.batch_size):
                ids = order[start:start + self.batch_size]
                length = max(len(pairs[i][0]) for i in ids)
                # Pad to a tensor-core-friendly multiple without changing nonpad tokens.
                length = min(512, ((length + 7) // 8) * 8)
                tokens = torch.full((len(ids), length), self.tokenizer.pad_token_id, dtype=torch.long)
                types = torch.zeros_like(tokens)
                attention = torch.zeros_like(tokens)
                for row, i in enumerate(ids):
                    seq, segments = pairs[i]
                    tokens[row, :len(seq)] = torch.tensor(seq)
                    types[row, :len(seq)] = torch.tensor(segments)
                    attention[row, :len(seq)] = 1
                device = self.model.device
                inputs = dict(input_ids=tokens.to(device), token_type_ids=types.to(device), attention_mask=attention.to(device))
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.precision == "autocast_bf16"):
                    values = self.model(**inputs).logits[:, 0].float().cpu().numpy()
                scores[ids] = values
            if not np.isfinite(scores).all():
                raise ValueError("non-finite reranker output")
            return scores

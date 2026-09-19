"""Optional NumPy packed-code scan; separate from source storage and authorization."""

import math

from .activation import StampSchema
from .security import identifier


class PackedStampIndex:
    """Immutable packed codes, chunked temporary memory, explicit eligible row IDs.

    This is exact linear scan of binary codes, not an approximate sublinear index.
    Source storage, metadata filtering, authentication and concurrent updates belong
    to the host. Build a new snapshot to replace an index.
    """

    def __init__(self, schema: StampSchema, keys, codes):
        import numpy as np

        keys = tuple(keys)
        if not 1 <= len(keys) <= 1000000 or len(set(keys)) != len(keys):
            raise ValueError("one to one million unique keys required")
        for key in keys:
            identifier(key)
        width = sum(bits // 8 for _, _, bits in schema.views)
        array = np.asarray(codes)
        if array.dtype != np.uint8 or array.shape != (len(keys), width):
            raise ValueError("codes must be uint8 rows matching the schema")
        self.schema, self.keys = schema, keys
        self._codes = array.copy()
        self._codes.flags.writeable = False

    @property
    def code_bytes(self):
        return self._codes.nbytes

    def search(self, query, *, eligible, limit=10, weights=None, chunk_size=4096):
        import numpy as np

        if StampSchema.for_stamp(query) != self.schema:
            raise ValueError("query schema mismatch")
        if type(limit) is not int or not 0 <= limit <= 100 or type(chunk_size) is not int or not 1 <= chunk_size <= 65536:
            raise ValueError("invalid search limits")
        if eligible is None:
            raise ValueError("host must explicitly supply authorized, current row indices")
        if len(eligible) > 1000000:
            raise ValueError("eligible row limit exceeded")
        ids = np.asarray(eligible)
        if ids.ndim != 1 or (ids.size and (ids.dtype.kind not in "iu" or ids.min() < 0 or ids.max() >= len(self.keys))):
            raise ValueError("invalid eligible rows")
        ids = np.unique(ids.astype(np.int64))
        names = [n for n, _, _ in self.schema.views]
        weights = dict.fromkeys(names, 1.0) if weights is None else weights
        if set(weights) != set(names) or not all(math.isfinite(w) and w >= 0 for w in weights.values()):
            raise ValueError("invalid weights")
        total = sum(weights.values())
        if not 0 < total < math.inf:
            raise ValueError("positive finite total required")
        if not limit:
            return []
        raw = b"".join(s.value.to_bytes(s.bits // 8, "big") for _, s in query.views)
        q = np.frombuffer(raw, dtype=np.uint8)
        popcount = np.array([i.bit_count() for i in range(256)], dtype=np.uint8)
        best = []
        for start in range(0, len(ids), chunk_size):
            block = ids[start:start + chunk_size]
            disagreements = popcount[np.bitwise_xor(self._codes[block], q)]
            score, pos = np.zeros(len(block), dtype=np.float64), 0
            for name, _, bits in self.schema.views:
                width = bits // 8
                score += (weights[name] / total) * (1 - disagreements[:, pos:pos + width].sum(axis=1) / bits)
                pos += width
            # Stable row order resolves ties and matches the scalar reference.
            top = np.lexsort((block, -score))[:limit]
            best.extend((int(block[i]), float(score[i])) for i in top)
            best = sorted(best, key=lambda row: (-row[1], row[0]))[:limit]
        return [{"source": self.keys[i], "score": score, "row": i} for i, score in best]

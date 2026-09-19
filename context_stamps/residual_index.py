"""Scalar-quantized routing with residual bounds and verified exact refinement.

This optional NumPy index retains full vectors in a host-owned backing store.
It does not make a binary hash lossless or modify an encoder's attention.
"""

import hashlib

from .security import identifier


class ResidualIndex:
    """Read-only int8 snapshot. Explicit eligible rows enforce host filtering.

    For reconstructed vector z and residual radius e >= ||x-z||, Cauchy-Schwarz
    bounds x.q by z.q +/- e*||q||. Every row whose upper bound reaches the kth
    largest lower bound is refined. No fixed shortlist can silently drop a winner.
    Scores use float64 accumulation; conservative numerical slack is included.
    """

    def __init__(self, vectors, *, encoder_id):
        import numpy as np

        identifier(encoder_id)
        values = np.asarray(vectors)
        if values.dtype != np.float32 or values.ndim != 2:
            raise ValueError("float32 matrix required")
        if not 1 <= len(values) <= 1000000 or not 1 <= values.shape[1] <= 4096:
            raise ValueError("matrix exceeds index bounds")
        if not np.isfinite(values).all():
            raise ValueError("finite vectors required")
        self.encoder_id = encoder_id
        self.rows, self.dimension = values.shape
        self._scale = np.max(np.abs(values), axis=1).astype(np.float64) / 127
        if (self._scale == 0).any():
            raise ValueError("zero vectors cannot be indexed")
        self._codes = np.empty(values.shape, dtype=np.int8)
        self._radius = np.empty(self.rows, dtype=np.float64)
        self._digests = np.empty((self.rows, 32), dtype=np.uint8)
        for start in range(0, self.rows, 4096):
            stop = min(start + 4096, self.rows)
            block = values[start:stop].astype(np.float64)
            norms = np.linalg.norm(block, axis=1)
            if not np.all(np.abs(norms - 1) <= 1e-4):
                raise ValueError("unit-normalized vectors required")
            codes = np.rint(block / self._scale[start:stop, None]).clip(-127, 127).astype(np.int8)
            self._codes[start:stop] = codes
            error = np.linalg.norm(block - codes * self._scale[start:stop, None], axis=1)
            self._radius[start:stop] = np.nextafter(error + 1e-12, np.inf)
            for i in range(start, stop):
                digest = hashlib.sha256(values[i].astype("<f4", copy=False).tobytes()).digest()
                self._digests[i] = np.frombuffer(digest, dtype=np.uint8)
        for array in (self._scale, self._codes, self._radius, self._digests):
            array.flags.writeable = False

    @property
    def index_bytes(self):
        """Persistent array bytes, excluding full-vector backing store and Python objects."""
        return sum(a.nbytes for a in (self._scale, self._codes, self._radius, self._digests))

    def search(self, query, *, encoder_id, eligible, fetch, limit=10):
        """Fetch must return float32 vectors in requested row order from this snapshot.

        A stale, reordered or corrupted backing vector aborts the search. The host
        authenticates the index and eligibility list. Hashes are not signatures.
        """
        import numpy as np

        if encoder_id != self.encoder_id:
            raise ValueError("encoder identity mismatch")
        if type(limit) is not int or not 0 <= limit <= 100:
            raise ValueError("limit must be between zero and 100")
        q = np.asarray(query, dtype=np.float64)
        if q.shape != (self.dimension,) or not np.isfinite(q).all():
            raise ValueError("finite matching query vector required")
        norm = np.linalg.norm(q)
        if not abs(norm - 1) <= 1e-4:
            raise ValueError("unit-normalized query required")
        if eligible is None or len(eligible) > self.rows:
            raise ValueError("explicit bounded eligible rows required")
        ids = np.asarray(eligible)
        if ids.ndim != 1 or (ids.size and (ids.dtype.kind not in "iu" or ids.min() < 0 or ids.max() >= self.rows)):
            raise ValueError("invalid eligible rows")
        ids = np.unique(ids.astype(np.int64))
        count = min(limit, len(ids))
        if not count:
            return {"rows": [], "scores": [], "refined_rows": 0, "eligible_rows": len(ids)}
        approx = np.empty(len(ids), dtype=np.float64)
        for start in range(0, len(ids), 4096):
            block = ids[start:start + 4096]
            approx[start:start + len(block)] = np.einsum(
                "ij,j->i", self._codes[block], q, dtype=np.float64) * self._scale[block]
        # Deliberately conservative for the bounded 4096-dimensional unit vectors.
        radius = self._radius[ids] * norm + 1e-10
        threshold = np.partition(approx - radius, len(ids) - count)[len(ids) - count]
        candidates = ids[approx + radius >= threshold]
        scores = np.empty(len(candidates), dtype=np.float64)
        for start in range(0, len(candidates), 4096):
            rows = candidates[start:start + 4096]
            exact = np.asarray(fetch(rows))
            if exact.dtype != np.float32 or exact.shape != (len(rows), self.dimension) or not np.isfinite(exact).all():
                raise ValueError("invalid backing vectors")
            for j, row in enumerate(rows):
                digest = hashlib.sha256(exact[j].astype("<f4", copy=False).tobytes()).digest()
                if digest != self._digests[row].tobytes():
                    raise ValueError("backing vector does not match index snapshot")
            scores[start:start + len(rows)] = np.einsum("ij,j->i", exact, q, dtype=np.float64)
        order = np.lexsort((candidates, -scores))[:count]
        return {"rows": candidates[order].tolist(), "scores": scores[order].tolist(),
                "refined_rows": len(candidates), "eligible_rows": len(ids)}

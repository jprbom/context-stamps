"""Query-driven activation and compact spherical payloads with a shared schema.

Activation selects candidates. It never executes tools or grants access.
"""

import base64
import json
import math
from dataclasses import dataclass

from stamps import Stamp, content_digest

from .spherical import SphericalStamp


@dataclass(frozen=True)
class StampSchema:
    views: tuple[tuple[str, str, int], ...]

    def __post_init__(self):
        object.__setattr__(self, "views", tuple(tuple(v) for v in self.views))
        probe = SphericalStamp(tuple((n, Stamp(f, b, 0)) for n, f, b in self.views))
        if self.views != tuple((n, s.family_id, s.bits) for n, s in probe.views):
            raise ValueError("schema views must be sorted and unique")

    @classmethod
    def for_stamp(cls, stamp):
        return cls(tuple((n, s.family_id, s.bits) for n, s in stamp.views))

    @property
    def identity(self):
        return content_digest(json.dumps(self.views, separators=(",", ":")))

    def pack(self, stamp):
        if StampSchema.for_stamp(stamp) != self:
            raise ValueError("schema mismatch")
        raw = b"".join(s.value.to_bytes(s.bits // 8, "big") for _, s in stamp.views)
        return "scqc1:" + self.identity + ":" + base64.urlsafe_b64encode(raw).decode("ascii")

    def unpack(self, payload):
        if not isinstance(payload, str) or len(payload) > 1024:
            raise ValueError("invalid compact payload")
        try:
            prefix, identity, code = payload.split(":")
            if prefix != "scqc1" or identity != self.identity:
                raise ValueError("unknown shared schema")
            raw = base64.b64decode(code, altchars=b"-_", validate=True)
            if len(raw) != sum(b // 8 for _, _, b in self.views):
                raise ValueError("wrong payload width")
            pos, views = 0, []
            for name, family, bits in self.views:
                size = bits // 8
                views.append((name, Stamp(family, bits, int.from_bytes(raw[pos:pos + size], "big"))))
                pos += size
            return SphericalStamp(tuple(views))
        except (TypeError, UnicodeError) as exc:
            raise ValueError("invalid compact payload") from exc


def activate(query, candidates, *, threshold, limit=5, model=None):
    """Return scored candidate indices above an explicit, application-calibrated threshold.

    Candidates must already be scoped to the current authorized store. A returned
    index is not permission to execute, nor proof that a dependency is available.
    """
    if not math.isfinite(threshold) or type(limit) is not int or not 0 <= limit <= 100:
        raise ValueError("finite threshold and bounded limit required")
    if len(candidates) > 1000:
        raise ValueError("candidate limit exceeded")
    rows = []
    for i, candidate in enumerate(candidates):
        facets = query.compare(candidate)
        score = query.score(candidate) if model is None else model.score(query, candidate)
        if score >= threshold:
            rows.append({"index": i, "score": score, "facets": facets})
    return sorted(rows, key=lambda row: (-row["score"], row["index"]))[:limit]

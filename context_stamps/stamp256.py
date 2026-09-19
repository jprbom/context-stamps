"""Exactly 256 routing bits with an application-selected facet allocation.

Schema/encoder identity, exact content digests, authorization and source data live
outside these 32 bytes. A raw code is meaningful only in a trusted shared schema.
"""

from .activation import StampSchema
from .spherical import SphericalStamp


class Stamp256Codec:
    def __init__(self, families):
        if not families or sum(f.bits for f in families.values()) != 256:
            raise ValueError("facet families must total exactly 256 bits")
        # Probe also validates view names, supported projection method and family types.
        probe = SphericalStamp.encode({n: [1.0] + [0.0] * (f.dim - 1) for n, f in families.items()}, families)
        self._families = dict(families)
        self.schema = StampSchema.for_stamp(probe)

    def encode(self, vectors):
        return SphericalStamp.encode(vectors, self._families)

    def pack(self, stamp):
        if StampSchema.for_stamp(stamp) != self.schema:
            raise ValueError("schema mismatch")
        return b"".join(s.value.to_bytes(s.bits // 8, "big") for _, s in stamp.views)

    def unpack(self, payload, *, schema_id):
        from stamps import Stamp

        if not isinstance(payload, bytes) or len(payload) != 32:
            raise ValueError("exactly 32 bytes required")
        if schema_id != self.schema.identity:
            raise ValueError("shared schema identity mismatch")
        offset, views = 0, []
        for name, family, bits in self.schema.views:
            size = bits // 8
            views.append((name, Stamp(family, bits, int.from_bytes(payload[offset:offset + size], "big"))))
            offset += size
        return SphericalStamp(tuple(views))

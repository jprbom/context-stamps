"""Offline 256-bit routing payload; schema and evidence stay in the application."""

import base64

from context_stamps import Family, HashingEncoder, Stamp256Codec

encoder = HashingEncoder(64)
facets = {"content": "Update timeout configuration", "entity": "worker_alpha",
          "intent": "implement", "task": "timeout"}
codec = Stamp256Codec({name: Family(encoder.identity, 64, 64, 17 + i)
                       for i, name in enumerate(facets)})
stamp = codec.encode({name: encoder.encode(value) for name, value in facets.items()})
payload = codec.pack(stamp)
print("Routing bytes:", len(payload))
print("Shared schema:", codec.schema.identity)
print("Text transport:", base64.b64encode(payload).decode("ascii"))
restored = codec.unpack(payload, schema_id=codec.schema.identity)
assert restored == stamp
print("Facet scores:", stamp.compare(restored))
print("Resolve original authorized evidence before sending context to a model.")

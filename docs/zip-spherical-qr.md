# Zip Spherical QR: 256 bits, not 256 bytes

The routing code is exactly **256 bits / 32 bytes**. It represents supplied facets through angular fingerprints. A schema shared by both endpoints assigns bit ranges to the facets and identifies their encoders. Exact source identity, revisions, authorization and the evidence itself remain outside this code.

The name describes a compact context reference. It is not a claim that ZIP can losslessly compress arbitrary context into 32 bytes. In the recorded sample, zlib expanded the 32-byte code to 41 bytes. Raw binary is used. Base64 text transport uses 44 bytes; the sample tokenized to 32 cl100k tokens. Other codes can tokenize differently. These are transport tokens, not the downstream model's input tokens.

```python
from context_stamps import Family, HashingEncoder, Stamp256Codec

encoder = HashingEncoder(64)
facets = {"content": "Update timeout", "entity": "worker_alpha",
          "intent": "implement", "task": "timeout"}
codec = Stamp256Codec({name: Family(encoder.identity, 64, 64, 17 + i)
                       for i, name in enumerate(facets)})
stamp = codec.encode({name: encoder.encode(value) for name, value in facets.items()})
payload = codec.pack(stamp)
assert len(payload) == 32
restored = codec.unpack(payload, schema_id=codec.schema.identity)
assert restored == stamp
```

`examples/zip_spherical_qr.py` runs offline without optional dependencies. `Stamp256Codec` accepts alternative byte-aligned allocations totaling 256 bits. Both endpoints must use the exact same schema. The code alone cannot detect a wrong schema, establish authenticity, identify an entity exactly, enforce access or reconstruct evidence.

## Allocating the bit budget

Seven allocations were compared in the fixed facet order content/entity/intent/task:

`64/64/64/64`, `128/64/32/32`, `160/32/32/32`, `192/32/16/16`, `32/128/48/48`, `16/160/40/40`, `16/80/80/80`.

Each was trained with the same nonnegative pairwise objective across three projection seeds: 21 training runs, 400 steps each. Selection used only the existing 40-query validation split. The selected allocation remained **64/64/64/64**. Fresh 200-query and shifted-task 200-query splits gave selected-model top-1 accuracy of 0.9950 and 0.9900. Unequal allocation did not improve on the equal-bit learned control. Exact supplied fields still solve these fixtures completely.

Bit entropy, training losses, validation trials and per-query rankings are retained in `evidence/stamp256-v1`. Marginal bit entropy measures balance, not semantic information or joint independence. No claim of maximum possible information follows.

## Quality-preserving retrieval is a separate layer

The original 256-bit public retrieval results remain lossy. The new `ResidualIndex` matches the dense reference by retaining a richer int8 representation, conservative error bounds and access to full vectors. It is **not contained in the 32-byte routing code**. See [retrieval repair and efficiency](retrieval-repair.md) for memory, latency and source-access measurements.

Use the compact code across an established host session, resolve current authorized source evidence and pass that evidence to an ordinary model. When exact entity/task fields already determine the target, skip approximate search and use the exact graph path.

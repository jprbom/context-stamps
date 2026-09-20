"""Inspect automatically observed facets and create a structured 32-byte capsule."""

from context_stamps import FacetCompiler, HashingEncoder, structured_256_codec

text = "Update api/router.py because Router.search() depends on policy.py v2.1."
compiled = FacetCompiler().compile(
    text, metadata={"authority": "maintainer", "policy": "internal", "modality": "code"}
)
for name, value in compiled.facets.items():
    item = compiled.evidence[name]
    print(f"{name:10} {value!r} [{item.source}; {item.rule}]")

# This input supplies all eight document views. Applications must not invent any
# missing values; partial query code should retain only observed views and use
# FacetQuery, as shown in examples/partial_facets.py.
names = (
    "semantic", "task", "entity", "relation", "temporal", "authority", "policy", "modality"
)
assert set(compiled.facets) == set(names)
values = {name: compiled.facets[name] for name in names}
encoder = HashingEncoder(64)
codec = structured_256_codec(encoder)
stamp = codec.encode({name: encoder.encode(value) for name, value in values.items()})
print("capsule_bytes", len(codec.pack(stamp)))
print("schema", codec.schema.identity)

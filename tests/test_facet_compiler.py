import unittest

from context_stamps import (
    STRUCTURED_256_PROFILE,
    FacetCompiler,
    HashingEncoder,
    structured_256_codec,
)


class FacetCompilerTests(unittest.TestCase):
    def test_observed_facets_and_provenance(self):
        compiled = FacetCompiler().compile(
            "Update api/router.py because Router.search() depends on policy.py v2.1 on 2026-09-20.",
            metadata={"authority": "maintainer", "policy": "internal", "modality": "code"},
        )
        self.assertEqual(
            set(compiled.facets),
            {"semantic", "task", "entity", "relation", "temporal", "authority", "policy", "modality"},
        )
        self.assertIn("depends on", compiled.facets["relation"])
        self.assertIn("v2.1", compiled.facets["temporal"])
        self.assertEqual(compiled.evidence["authority"].source, "host-metadata")

    def test_absent_facets_are_omitted_and_result_is_immutable(self):
        compiled = FacetCompiler().compile("A short neutral sentence.")
        self.assertEqual(dict(compiled.facets), {"semantic": "A short neutral sentence."})
        with self.assertRaises(TypeError):
            compiled.facets["policy"] = "invented"

    def test_rejects_unknown_or_unbounded_inputs(self):
        compiler = FacetCompiler()
        for metadata in ({"unknown": "x"}, {"modality": "smell"}, {"policy": ""}):
            with self.assertRaises(ValueError):
                compiler.compile("text", metadata=metadata)
        with self.assertRaises(ValueError):
            compiler.compile("x" * 65537)

    def test_structured_profile_is_exactly_256_bits(self):
        self.assertEqual(sum(STRUCTURED_256_PROFILE.values()), 256)
        encoder = HashingEncoder(64)
        codec = structured_256_codec(encoder)
        vectors = {name: encoder.encode(name) for name in STRUCTURED_256_PROFILE}
        payload = codec.pack(codec.encode(vectors))
        self.assertEqual(len(payload), 32)


if __name__ == "__main__":
    unittest.main()

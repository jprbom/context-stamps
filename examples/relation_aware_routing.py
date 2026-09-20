"""Build a relation view and admit or reject a precise hybrid profile."""

from context_stamps import HybridScoreProfile, RelationEdge, RelationMap, certify_hybrid_scope

relations = RelationMap([
    RelationEdge("implementation", "requirement", "implements", 2.0),
    RelationEdge("implementation", "test", "verified_by", 1.0),
    RelationEdge("test", "fixture", "uses", 1.0),
])
relation_view = relations.encode("implementation", dim=64, hops=3)

# Supply per-query values from a disjoint validation partition. These small repeated
# values demonstrate the API only; they are not benchmark evidence.
dense_validation = [0.30 + (index % 5) / 100 for index in range(120)]
hybrid_validation = [value + 0.02 for value in dense_validation]
profile = HybridScoreProfile(semantic_weight=0.75)
certificate = certify_hybrid_scope(
    "demo-corpus:minilm-revision:schema-revision",
    profile,
    dense_validation,
    hybrid_validation,
    resamples=1_000,
)

print({
    "relation_revision": relations.revision,
    "relation_dimensions": len(relation_view),
    "hybrid_enabled": certificate.enabled,
    "validation_lower_gain": certificate.lower_gain,
})

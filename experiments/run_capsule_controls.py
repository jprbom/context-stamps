"""Generate deterministic control evidence for the v0.4 capsule interfaces."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from context_stamps import (
    STRUCTURED_256_PROFILE,
    FacetCompiler,
    HashingEncoder,
    ProgressiveRouter,
    RoutingPolicy,
    structured_256_codec,
)

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "evidence" / "capsule-controls-v1" / "results.json"


def main():
    cases = [
        ("Review api/router.py and update Router.search() for v2.1.", {"semantic", "task", "entity", "temporal"}),
        ("policy.py depends on schema.py on 2026-09-20.", {"semantic", "entity", "relation", "temporal"}),
        ("A short neutral sentence.", {"semantic"}),
        ("Compare image generation outputs.", {"semantic", "task"}),
        ("train model.py using data/train.json", {"semantic", "task", "entity"}),
        ("The worker requires queue.config v1.4", {"semantic", "entity", "relation", "temporal"}),
        ("Explain why service.run() calls cache.get().", {"semantic", "task", "entity", "relation"}),
        ("Measure audio latency.", {"semantic", "task"}),
    ]
    compiler = FacetCompiler()
    facet_rows = []
    for text, expected in cases:
        observed = set(compiler.compile(text).facets)
        facet_rows.append({"text": text, "expected": sorted(expected), "observed": sorted(observed),
                           "passed": observed == expected})

    encoder = HashingEncoder(64)
    codec = structured_256_codec(encoder)
    roundtrips = 0
    for index in range(100):
        vectors = {name: encoder.encode(f"{name} control {index}") for name in STRUCTURED_256_PROFILE}
        stamp = codec.encode(vectors)
        raw = codec.pack(stamp)
        roundtrips += codec.unpack(raw, schema_id=codec.schema.identity) == stamp and len(raw) == 32

    router = ProgressiveRouter()
    options = {"eligible": ("a", "b", "c"), "scope": "control:v1", "limit": 1,
               "precise": lambda ids, count: [("b", .7)][:count]}
    def compact(ids, count):
        return [("a", .9), ("b", .5)][:count]
    qualified = RoutingPolicy("control:v1", 1, .7, .1, True, .03, 100)
    excessive = RoutingPolicy("control:v1", 1, .7, .1, True, .08, 100)
    routes = [
        router.search(**options, exact_key="a")["route"],
        router.search(**options, exact_key="hidden")["route"],
        router.search(**options, compact=compact)["route"],
        router.search(**options, compact=compact, policy=excessive)["route"],
        router.search(**options, compact=compact, policy=qualified)["route"],
    ]
    expected_routes = ["exact", "abstain", "precise", "precise", "compact"]
    result = {
        "format": "capsule-controls-v1",
        "facet_cases": facet_rows,
        "facet_cases_passed": sum(row["passed"] for row in facet_rows),
        "capsule_profile": STRUCTURED_256_PROFILE,
        "capsule_bits": sum(STRUCTURED_256_PROFILE.values()),
        "roundtrips_passed": roundtrips,
        "roundtrips_total": 100,
        "router_routes": routes,
        "router_expected": expected_routes,
        "router_controls_passed": routes == expected_routes,
        "limitations": [
            "Facet cases are authored controls, not an independently annotated corpus.",
            "The compiler is a deterministic baseline, not a semantic parser.",
            "Router controls validate enforcement, not public retrieval quality.",
        ],
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if result["facet_cases_passed"] != len(cases) or roundtrips != 100 or not result["router_controls_passed"]:
        raise SystemExit("control failure")
    print(json.dumps({"output": str(OUTPUT), "facet_cases": len(cases), "roundtrips": roundtrips,
                      "router_controls": len(routes)}))


if __name__ == "__main__":
    main()

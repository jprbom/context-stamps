"""Fixed-budget synthetic facet ablation with an exact-metadata control."""

import json
import math
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from run_hybrid_retrieval import save, sha  # noqa: E402

from context_stamps import (  # noqa: E402
    FacetQuery,
    HashingEncoder,
    RelationEdge,
    RelationMap,
    SphericalStamp,
    structured_256_codec,
)
from stamps import Family, similarity, stamp_vector  # noqa: E402

OUT = ROOT / "evidence/runtime-v1"


def main():
    encoder = HashingEncoder(64)
    codec = structured_256_codec(encoder)
    semantic_family = Family(encoder.identity, 64, 256, 20260926)
    rng = random.Random(20260926)  # nosec B311 - synthetic experimental ordering
    rows = []
    names = ("semantic", "task", "entity", "relation", "temporal", "authority", "policy", "modality")
    for task in range(80):
        target = dict(semantic="experiment validation result", task="compare", entity=f"project-{task}",
            relation="uses-current-dataset", temporal="revision-2", authority="reviewed", policy="reproducible", modality="table")
        candidates = [dict(target)]
        for facet in names[1:]:
            candidates.append(dict(target, **{facet: "alternative-" + facet}))
        rng.shuffle(candidates)
        expected = candidates.index(target)
        def vectors(item):
            values = {key: encoder.encode(value) for key, value in item.items()}
            relation = RelationMap([RelationEdge("result", f"dataset-{task}", item["relation"])])
            values["relation"] = relation.encode("result", dim=64, hops=2)
            return values
        query = vectors(target)
        document_vectors = [vectors(c) for c in candidates]
        qstamp = codec.encode(query)
        stamps = [codec.encode(d) for d in document_vectors]
        semquery = stamp_vector(query["semantic"], semantic_family)
        rankings = {}
        all_scores = {}
        for method in ("semantic_float", "semantic_256", "facets_float", "facets_256", "without_relation", "without_entity", "exact_metadata"):
            scores = []
            if method.startswith("without_"):
                removed = method.removeprefix("without_")
                partial = FacetQuery(SphericalStamp(tuple((name, value) for name, value in qstamp.views if name != removed)))
            for i, d in enumerate(document_vectors):
                if method == "semantic_float":
                    value = math.fsum(a * b for a, b in zip(query["semantic"], d["semantic"]))
                elif method == "semantic_256":
                    value = similarity(semquery, stamp_vector(d["semantic"], semantic_family))
                elif method == "facets_float":
                    value = math.fsum(math.fsum(a * b for a, b in zip(query[n], d[n])) for n in names) / 8
                elif method == "facets_256":
                    value = qstamp.score(stamps[i])
                elif method.startswith("without_"):
                    value = partial.score(stamps[i])
                else:
                    value = float(candidates[i] == target)
                scores.append(value)
            rankings[method] = max(range(len(scores)), key=lambda i: (scores[i], -i))
            all_scores[method] = scores
        rows.append(dict(task=task, expected=expected, chosen=rankings,
            correct={m: i == expected for m, i in rankings.items()}, scores=all_scores))
    summary = {m: sum(row["correct"][m] for row in rows) for m in rows[0]["correct"]}
    save(OUT / "spherical-ablation.json", dict(tasks=80, candidates=8, top_k=1, results=summary, rows=rows,
        protocol="Fixed seed20260926, 80 fictional projects; eight equal-semantic candidates differ in one declared facet; same candidate set and top1 budget; no training or parameter selection.",
        limits="Deliberately ambiguous metadata fixture. Exact metadata is a strong control and can tie the stamp. This does not establish real-world semantic-retrieval superiority or compression of external evidence/graphs.",
        stamp_bytes=32, float_facets_numeric_bytes=8 * 64 * 4,
        source_sha256={"experiments/ablate_spherical_runtime.py": sha(Path(__file__))}))
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()

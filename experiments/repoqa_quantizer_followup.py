"""Train-only ITQ diagnostic after the frozen stamp route failed validation.

Uses cached embeddings and no final-repository labels or raw-code export.
Copyright (c) 2026 Prashant Jagtap. MIT License.
"""

import argparse
import gzip
import hashlib
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
from repoqa_context_localization import (  # noqa: E402
    MODEL,
    REVISION,
    SOURCE_SHA256,
    SPLITS,
    bit_agreement,
    encode_cache,
    functions,
    rank_scores,
    sha,
)

from context_stamps.learning import fit_family  # noqa: E402
from stamps import Family, _planes  # noqa: E402


def bits(vectors, family):
    x = np.asarray(vectors, dtype=np.float32)
    if family.mean:
        x = x - np.asarray(family.mean, dtype=np.float32)
    projected = x @ np.asarray(_planes(family), dtype=np.float32).T
    return np.packbits(projected >= 0, axis=1, bitorder="little")


def run(source, cache, out):
    if sha(source) != SOURCE_SHA256:
        raise ValueError("source checksum changed")
    data = {r["repo"]: r for r in json.loads(gzip.decompress(source.read_bytes()))["python"]}
    train = []
    for name in SPLITS["train"]:
        repo = data[name]
        docs, _, _, _ = encode_cache(repo, functions(repo), None, cache)
        train.append(docs)
    train = np.concatenate(train)
    rng = np.random.default_rng(20260930)
    sample = train[rng.choice(len(train), 4000, replace=False)]
    t0 = time.perf_counter()
    learned = fit_family(sample, encoder=f"{MODEL}@{REVISION}:function-text-v1", bits=256,
                         method="itq", seed=20260930, iterations=12)
    fit_s = time.perf_counter() - t0
    random = Family(learned.encoder, 384, 256, 20260930)
    rows = []
    for name in SPLITS["validation"]:
        repo = data[name]
        candidates = functions(repo)
        docs, queries, _, _ = encode_cache(repo, candidates, None, cache)
        codes = {"gaussian256": bits(docs, random), "itq256": bits(docs, learned)}
        qcodes = {"gaussian256": bits(queries, random), "itq256": bits(queries, learned)}
        for qid, needle in enumerate(repo["needles"]):
            expected = {i for i, r in enumerate(candidates) if
                        r["path"] == needle["path"] and r["name"] == needle["name"]
                        and r["start"] == needle["start_byte"]}
            if len(expected) != 1:
                raise ValueError("needle index mismatch")
            ranks = {method: rank_scores(bit_agreement(np.unpackbits(qcodes[method][qid],
                                                         bitorder="little"), code, 256), expected)
                     for method, code in codes.items()}
            ranks["dense"] = rank_scores(docs @ queries[qid], expected)
            rows.append(dict(id=f"{name}:{qid}", ranks=ranks))
    methods = ("dense", "gaussian256", "itq256")
    summary = {method: dict(top1=sum(r["ranks"][method] == 1 for r in rows),
                            top10=sum(r["ranks"][method] <= 10 for r in rows),
                            mrr=round(sum(1 / r["ranks"][method] for r in rows) / len(rows), 6))
               for method in methods}
    out.mkdir(parents=True, exist_ok=True)
    result = dict(status="exploratory validation only; no untouched final run", source_sha256=SOURCE_SHA256,
                  model=MODEL, revision=REVISION, train_repositories=SPLITS["train"],
                  validation_repositories=SPLITS["validation"], training_rows=4000,
                  selection_seed=20260930, itq_iterations=12, fit_seconds=fit_s,
                  learned_family_sha256=hashlib.sha256(learned.to_json().encode()).hexdigest(),
                  summary=summary, rows=rows,
                  decision="Do not activate or query the prior final cohort regardless of validation outcome.")
    (out / "quantizer-followup.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n",
                                                   encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=ROOT / "evidence/repoqa-localization-v1")
    args = parser.parse_args()
    run(args.source, args.cache, args.out)

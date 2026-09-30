"""Train-only 256-bit quantization ablation on RepoQA validation repositories.

Compares multi-view ITQ with 32-byte product quantization and an unchanged
float32 dense control. Does not read the previously inspected final group.
Copyright (c) 2026 Prashant Jagtap. MIT License.
"""

import argparse
import gzip
import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import faiss  # noqa: E402
import numpy as np  # noqa: E402
from repoqa_context_localization import (  # noqa: E402
    LEXICAL,
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
from repoqa_quantizer_followup import bits  # noqa: E402

from context_stamps.learning import fit_family  # noqa: E402


def run(source, cache, out):
    if sha(source) != SOURCE_SHA256:
        raise ValueError("source checksum differs from frozen release")
    data = {r["repo"]: r for r in json.loads(gzip.decompress(source.read_bytes()))["python"]}
    train_semantic, train_task = [], []
    for name in SPLITS["train"]:
        repo = data[name]
        rows = functions(repo)
        docs, _, _, _ = encode_cache(repo, rows, None, cache)
        train_semantic.append(docs)
        train_task.extend(LEXICAL.encode(r["body"][:1200]) for r in rows)
    train_semantic = np.concatenate(train_semantic).astype("float32")
    train_task = np.asarray(train_task, dtype="float32")
    rng = np.random.default_rng(20260930)
    sample_ids = rng.choice(len(train_semantic), 4000, replace=False)
    t0 = time.perf_counter()
    semantic = fit_family(train_semantic[sample_ids], encoder=f"{MODEL}@{REVISION}:semantic224-v1",
                          bits=224, method="itq", seed=20260930, iterations=12)
    task = fit_family(train_task[sample_ids], encoder=LEXICAL.identity + ":task32-v1",
                      bits=32, method="itq", seed=20260931, iterations=12)
    itq_fit_s = time.perf_counter() - t0
    faiss.omp_set_num_threads(4)
    t0 = time.perf_counter()
    pq = faiss.IndexPQ(384, 32, 8, faiss.METRIC_INNER_PRODUCT)
    pq.pq.cp.min_points_per_centroid = 16
    pq.train(train_semantic)
    pq_fit_s = time.perf_counter() - t0
    codebook_bytes = len(faiss.serialize_index(pq))
    output, timings = [], []
    for name in SPLITS["validation"]:
        repo = data[name]
        rows = functions(repo)
        docs, queries, _, _ = encode_cache(repo, rows, None, cache)
        task_docs = np.asarray([LEXICAL.encode(r["body"][:1200]) for r in rows], dtype="float32")
        task_queries = np.asarray([LEXICAL.encode(n["description"]) for n in repo["needles"]], dtype="float32")
        sem_codes = bits(docs, semantic)
        task_codes = bits(task_docs, task)
        qsem_codes = bits(queries, semantic)
        qtask_codes = bits(task_queries, task)
        pq.reset()
        pq.add(np.ascontiguousarray(docs, dtype="float32"))
        flat = faiss.IndexFlatIP(384)
        flat.add(np.ascontiguousarray(docs, dtype="float32"))
        request = np.ascontiguousarray(queries, dtype="float32")
        pq_scores, pq_order = pq.search(request, len(rows))
        _, dense_order = flat.search(request, len(rows))
        samples = {"pq32": [], "flat_dense": []}
        for repeat in range(21):
            order = ("pq32", "flat_dense") if repeat % 2 == 0 else ("flat_dense", "pq32")
            for method in order:
                target = pq if method == "pq32" else flat
                start = time.perf_counter()
                target.search(request, len(rows))
                if repeat:
                    samples[method].append((time.perf_counter() - start) * 1000 / len(queries))
        timings.append(dict(repo=name, candidates=len(rows), queries=len(queries),
                            pq_median_ms=statistics.median(samples["pq32"]),
                            flat_dense_median_ms=statistics.median(samples["flat_dense"])))
        if pq.code_size != 32 or pq.ntotal != len(rows):
            raise ValueError("product-quantization code size or count changed")
        for qid, needle in enumerate(repo["needles"]):
            correct = {i for i, r in enumerate(rows) if r["path"] == needle["path"] and
                       r["name"] == needle["name"] and r["start"] == needle["start_byte"]}
            if len(correct) != 1:
                raise ValueError("needle index mismatch")
            sem = bit_agreement(np.unpackbits(qsem_codes[qid], bitorder="little"), sem_codes, 224)
            task_agree = bit_agreement(np.unpackbits(qtask_codes[qid], bitorder="little"), task_codes, 32)
            product_sphere = (224 * sem + 32 * task_agree) / 256
            pq_rank = int(np.where(pq_order[qid] == next(iter(correct)))[0][0]) + 1
            dense_rank = rank_scores(docs @ queries[qid], correct)
            if int(np.where(dense_order[qid] == next(iter(correct)))[0][0]) + 1 != dense_rank:
                raise ValueError("Faiss flat and direct dense ranks disagree")
            output.append(dict(id=f"{name}:{qid}", ranks={
                "dense": dense_rank,
                "itq224_task32": rank_scores(product_sphere, correct),
                "pq32": pq_rank,
            }, pq_target_score=float(pq_scores[qid, pq_rank - 1])))
    summary = {method: dict(top1=sum(r["ranks"][method] == 1 for r in output),
                            top10=sum(r["ranks"][method] <= 10 for r in output),
                            mrr=round(sum(1 / r["ranks"][method] for r in output) / len(output), 6))
               for method in ("dense", "itq224_task32", "pq32")}
    result = dict(status="exploratory validation only; no final-set use", source_sha256=SOURCE_SHA256,
                  model=MODEL, revision=REVISION, training_repositories=SPLITS["train"],
                  validation_repositories=SPLITS["validation"], train_functions=len(train_semantic),
                  itq_training_rows=4000, itq_sample_seed=20260930, itq_fit_s=itq_fit_s,
                  itq_bits={"semantic": 224, "task": 32}, itq_iterations=12,
                  pq_config=dict(dimensions=384, subquantizers=32, bits_per_subquantizer=8,
                                 metric="inner_product", code_bytes=32, codebook_bytes=codebook_bytes,
                                 fit_s=pq_fit_s, faiss_version=faiss.__version__,
                                 threads=4, clustering_seed=int(pq.pq.cp.seed),
                                 min_points_per_centroid=int(pq.pq.cp.min_points_per_centroid)),
                  summary=summary, rows=output, warm_lookup_timings=timings,
                  limitations="The 32-byte figures exclude the shared PQ codebook, full source/resolver, metadata and encoder weights. Validation was already inspected; this does not qualify a route. Query encoding excluded from lookup times.")
    out.mkdir(parents=True, exist_ok=True)
    (out / "quantization-ablation.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n",
                                                   encoding="utf-8")
    print(json.dumps(dict(summary=summary, pq_codebook_bytes=codebook_bytes,
                          pq_fit_s=pq_fit_s, itq_fit_s=itq_fit_s), indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=ROOT / "evidence/repoqa-localization-v1")
    args = parser.parse_args()
    run(args.source, args.cache, args.out)

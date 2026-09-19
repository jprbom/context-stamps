"""Two-view spherical public retrieval; raw corpora and embeddings remain external."""

import argparse
import csv
import hashlib
import json
import math
import statistics
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from stamps import Family, HashingEncoder, _planes, stamp_vector

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "evidence/spherical-public-v1"
POPCOUNT = np.array([i.bit_count() for i in range(256)], dtype=np.uint8)


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8", newline="\n")


def qrels(path):
    result = {}
    with path.open(encoding="utf-8") as stream:
        for row in csv.DictReader(stream, delimiter="\t"):
            result.setdefault(row["query-id"], {})[row["corpus-id"]] = int(row["score"])
    return result


def ndcg(order, labels, ids):
    ideal = sorted((v for v in labels.values() if v > 0), reverse=True)[:10]
    denom = sum(v / math.log2(i + 2) for i, v in enumerate(ideal))
    return sum(max(0, labels.get(ids[i], 0)) / math.log2(j + 2) for j, i in enumerate(order[:10])) / denom if denom else 0


def project(matrix, family):
    # Same angular hyperplanes as the scalar library. Float64 sign computation.
    output = np.packbits(matrix.astype(np.float64) @ np.array(_planes(family)).T >= 0, axis=1, bitorder="little")
    for i in range(min(3, len(matrix))):
        assert int.from_bytes(output[i].tobytes(), "little") == stamp_vector(matrix[i], family).value
    return output


def agreement(query, documents, bits):
    return 1 - POPCOUNT[np.bitwise_xor(documents, query)].sum(axis=1) / bits


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scifact", type=Path, required=True)
    parser.add_argument("--replication-data", type=Path, required=True)
    parser.add_argument("--scifact-cache", type=Path, required=True)
    parser.add_argument("--replication-cache", type=Path, required=True)
    args = parser.parse_args()
    protocol = read(OUT / "protocol.json")
    old_scifact = read(ROOT / "evidence/scifact-v1/manifest.json")
    old_replication = read(ROOT / "evidence/replication-v1/manifest.json")
    all_rows, summary, manifests, validation = [], [], [], []
    chosen_weight = None
    for dataset in protocol["datasets"]:
        root = args.scifact if dataset == "scifact" else args.replication_data / dataset
        corpus = [json.loads(line) for line in (root / "corpus.jsonl").read_text(encoding="utf-8").splitlines()]
        queries = {r["_id"]: r["text"] for r in map(json.loads, (root / "queries.jsonl").read_text(encoding="utf-8").splitlines())}
        ids = [r["_id"] for r in corpus]
        documents = [(r.get("title", "") + " " + r["text"]).strip() for r in corpus]
        labels = qrels(root / "qrels/test.tsv")
        if dataset == "scifact":
            files = old_scifact["data_sha256"]
            train = old_scifact["training_queries"]
            val = old_scifact["validation_queries"]
            query_ids = sorted(set(train + val + list(labels)))
            key = hashlib.sha256(json.dumps([files, protocol["encoder"], protocol["revision"], query_ids], sort_keys=True).encode()).hexdigest()
            cache = args.scifact_cache
        else:
            files = next(x for x in old_replication["datasets"] if x["dataset"] == dataset)["data_sha256"]
            query_ids = sorted(labels)
            key = hashlib.sha256(json.dumps([files, protocol["encoder"], protocol["revision"], query_ids], sort_keys=True).encode()).hexdigest()
            cache = args.replication_cache
        assert all(sha(root / name) == digest for name, digest in files.items())
        dpath, qpath = cache / (key + "-D.npy"), cache / (key + "-Q.npy")
        d, q = np.load(dpath, allow_pickle=False), np.load(qpath, allow_pickle=False)
        assert d.shape == (len(ids), 384) and q.shape == (len(query_ids), 384)
        encoder = HashingEncoder(384)
        lexical_d = np.array([encoder.encode(text) for text in documents])
        lexical_q = np.array([encoder.encode(queries[qid]) for qid in query_ids])
        semantic_family = Family(protocol["encoder"] + "@" + protocol["revision"] + ":normalized-cache", 384, 256, 17)
        sem_d, sem_q = project(d, semantic_family), project(q, semantic_family)
        lex_family = Family(encoder.identity, 384, 128, 18)
        lex_d, lex_q = project(lexical_d, lex_family), project(lexical_q, lex_family)
        positions = {qid: i for i, qid in enumerate(query_ids)}
        doc_positions = {docid: i for i, docid in enumerate(ids)}
        def scores(qid):
            i = positions[qid]
            s = agreement(sem_q[i, :16], sem_d[:, :16], 128)
            l = agreement(lex_q[i], lex_d, 128)
            return s, l
        def rank(values, qid):
            values = values.copy()
            if qid in doc_positions:
                values[doc_positions[qid]] = -np.inf
            return np.argsort(-values, kind="stable")[:10].tolist()
        if dataset == "scifact":
            validation_labels = qrels(root / "qrels/train.tsv")
            for weight in (0.0, .25, .5, .75, 1.0):
                vals = []
                for qid in val:
                    s, l = scores(qid)
                    vals.append(ndcg(rank(weight * s + (1 - weight) * l, qid), validation_labels[qid], ids))
                validation.append({"semantic_weight": weight, "queries": len(vals), "ndcg10": statistics.mean(vals)})
            chosen_weight = max(validation, key=lambda r: r["ndcg10"])["semantic_weight"]
        for qid in sorted(labels):
            i = positions[qid]
            s, l = scores(qid)
            variants = {"dense": d @ q[i], "semantic_256": agreement(sem_q[i], sem_d, 256),
                        "semantic_128_lexical_128": chosen_weight * s + (1 - chosen_weight) * l}
            for method, values in variants.items():
                order = rank(values, qid)
                all_rows.append({"dataset": dataset, "query_id": qid, "method": method,
                                 "ndcg10": ndcg(order, labels[qid], ids), "ranked_ids": [ids[j] for j in order]})
        for method in protocol["methods"]:
            group = [r for r in all_rows if r["dataset"] == dataset and r["method"] == method]
            summary.append({"dataset": dataset, "method": method, "queries": len(group),
                            "ndcg10": statistics.mean(r["ndcg10"] for r in group)})
        manifests.append({"dataset": dataset, "documents": len(ids), "query_ids": query_ids,
                          "data_sha256": files, "embedding_sha256": {"documents": sha(dpath), "queries": sha(qpath)},
                          "semantic_family": json.loads(semantic_family.to_json()),
                          "lexical_family": json.loads(lex_family.to_json())})
        print(dataset, summary[-3:], flush=True)
    save(OUT / "validation.json", validation)
    save(OUT / "results.json", all_rows)
    save(OUT / "summary.json", summary)
    save(OUT / "manifest.json", {"protocol": protocol, "semantic_weight": chosen_weight, "datasets": manifests,
                                 "numpy": np.__version__, "source_sha256": {p: sha(ROOT / p) for p in (
                                     "experiments/run_spherical_public.py", "stamps.py")}})
    save(OUT / "checksums.json", {p.name: sha(p) for p in OUT.glob("*.json") if p.name != "checksums.json"})


if __name__ == "__main__":
    main()

"""Evaluate frozen dense/BM25 fusion, including prospective SciDocs."""

import argparse
import csv
import hashlib
import json
import math
import platform
import statistics
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from context_stamps.hybrid import HybridScoreProfile

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "evidence" / "hybrid-retrieval-v1"


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def save(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
                    encoding="utf-8", newline="\n")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def labels(path):
    result = defaultdict(dict)
    with path.open(encoding="utf-8") as stream:
        for row in csv.DictReader(stream, delimiter="\t"):
            result[row["query-id"]][row["corpus-id"]] = int(row["score"])
    return result


def tokens(text):
    import re
    return re.findall(r"\w+", text.casefold())


class BM25:
    def __init__(self, documents):
        self.size = len(documents)
        self.lengths = np.array([len(tokens(text)) for text in documents])
        self.average = max(1, float(self.lengths.mean()))
        postings = defaultdict(list)
        for index, text in enumerate(documents):
            for token, frequency in Counter(tokens(text)).items():
                postings[token].append((index, frequency))
        self.postings = {key: np.asarray(value) for key, value in postings.items()}

    def score(self, query):
        result = np.zeros(self.size, dtype=np.float64)
        for token in tokens(query):
            values = self.postings.get(token)
            if values is None:
                continue
            indexes, frequencies = values[:, 0], values[:, 1]
            inverse = math.log(1 + (self.size - len(indexes) + 0.5) / (len(indexes) + 0.5))
            result[indexes] += inverse * frequencies * 2.5 / (
                frequencies + 1.5 * (0.25 + 0.75 * self.lengths[indexes] / self.average)
            )
        return result


def metric(order, relevance, ids):
    gains = [max(0, relevance.get(ids[int(index)], 0)) for index in order[:10]]
    ideal = sorted((value for value in relevance.values() if value > 0), reverse=True)[:10]
    denominator = sum(value / math.log2(index + 2) for index, value in enumerate(ideal))
    relevant = sum(value > 0 for value in relevance.values())
    return {
        "ndcg10": sum(value / math.log2(index + 2) for index, value in enumerate(gains)) / denominator
        if denominator else 0,
        "recall10": sum(value > 0 for value in gains) / max(1, relevant),
        "mrr10": next((1 / (index + 1) for index, value in enumerate(gains) if value > 0), 0),
    }


def top(values, excluded):
    values = values.copy()
    if excluded is not None:
        values[excluded] = -np.inf
    return np.argsort(-values, kind="stable")[:10]


def percentile(values, probability):
    return float(np.percentile(np.asarray(values), probability))


def existing_embeddings(name, metadata, scifact_cache, replication_cache):
    files = metadata["data_sha256"]
    protocol = read(ROOT / "evidence/spherical-public-v1/protocol.json")
    key = hashlib.sha256(json.dumps([files, protocol["encoder"], protocol["revision"],
                                     metadata["query_ids"]], sort_keys=True).encode()).hexdigest()
    cache = scifact_cache if name == "scifact" else replication_cache
    documents, queries = cache / (key + "-D.npy"), cache / (key + "-Q.npy")
    assert sha(documents) == metadata["embedding_sha256"]["documents"]
    assert sha(queries) == metadata["embedding_sha256"]["queries"]
    return np.load(documents, allow_pickle=False), np.load(queries, allow_pickle=False), {
        "documents": sha(documents), "queries": sha(queries)
    }


def scidocs_embeddings(documents, queries, args, hashes, query_ids, protocol):
    key = hashlib.sha256(json.dumps([hashes, protocol["encoder"], protocol["encoder_revision"],
                                     query_ids], sort_keys=True).encode()).hexdigest()
    args.scidocs_cache.mkdir(parents=True, exist_ok=True)
    dpath, qpath = args.scidocs_cache / (key + "-D.npy"), args.scidocs_cache / (key + "-Q.npy")
    if not dpath.exists() or not qpath.exists():
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(protocol["encoder"], revision=protocol["encoder_revision"],
                                    device="cpu", trust_remote_code=False,
                                    model_kwargs={"use_safetensors": True})
        np.save(dpath, model.encode(documents, batch_size=64, normalize_embeddings=True,
                                    show_progress_bar=True).astype("float32"))
        np.save(qpath, model.encode([queries[key] for key in query_ids], batch_size=64,
                                    normalize_embeddings=True, show_progress_bar=True).astype("float32"))
    return np.load(dpath, allow_pickle=False), np.load(qpath, allow_pickle=False), {
        "documents": sha(dpath), "queries": sha(qpath)
    }


def main(args):
    protocol = read(OUT / "protocol.json")
    prior = read(ROOT / "evidence/spherical-public-v1/manifest.json")
    prior_by_name = {item["dataset"]: item for item in prior["datasets"]}
    profile = HybridScoreProfile(protocol["semantic_weight"])
    records, summaries, manifests = [], [], []
    for name in protocol["datasets"]:
        source = args.scidocs if name == "scidocs" else args.data / name
        corpus = [json.loads(line) for line in (source / "corpus.jsonl").read_text(encoding="utf-8").splitlines()]
        queries = {row["_id"]: row["text"] for row in map(
            json.loads, (source / "queries.jsonl").read_text(encoding="utf-8").splitlines())}
        relevance = labels(source / "qrels/test.tsv")
        query_ids = sorted(relevance)
        ids = [row["_id"] for row in corpus]
        positions = {value: index for index, value in enumerate(ids)}
        documents = [(row.get("title", "") + " " + row["text"]).strip() for row in corpus]
        hashes = {path: sha(source / path) for path in ("corpus.jsonl", "queries.jsonl", "qrels/test.tsv")}
        if name == "scidocs":
            assert hashes == protocol["scidocs"]["data_sha256"]
            dense_documents, dense_queries, embedding_hashes = scidocs_embeddings(
                documents, queries, args, hashes, query_ids, protocol)
        else:
            metadata = prior_by_name[name]
            assert all(value == metadata["data_sha256"][path] for path, value in hashes.items())
            dense_documents, all_queries, embedding_hashes = existing_embeddings(
                name, metadata, args.scifact_cache, args.replication_cache)
            query_positions = {value: index for index, value in enumerate(metadata["query_ids"])}
            dense_queries = all_queries[[query_positions[value] for value in query_ids]]
        assert dense_documents.shape == (len(ids), 384) and dense_queries.shape == (len(query_ids), 384)
        start = time.perf_counter()
        lexical = BM25(documents)
        build_seconds = time.perf_counter() - start
        timing = defaultdict(list)
        for query_index, query_id in enumerate(query_ids):
            excluded = positions.get(query_id)
            start = time.perf_counter()
            dense = dense_documents @ dense_queries[query_index]
            dense_order = top(dense, excluded)
            timing["dense_minilm"].append((time.perf_counter() - start) * 1000)
            start = time.perf_counter()
            bm25 = lexical.score(queries[query_id])
            bm25_order = top(bm25, excluded)
            timing["bm25"].append((time.perf_counter() - start) * 1000)
            start = time.perf_counter()
            dense_z = (dense - dense.mean()) / max(float(dense.std()), 1e-12)
            bm25_z = (bm25 - bm25.mean()) / max(float(bm25.std()), 1e-12)
            hybrid_order = top(profile.semantic_weight * dense_z + (1 - profile.semantic_weight) * bm25_z,
                               excluded)
            timing["hybrid_z_075"].append(timing["dense_minilm"][-1] + timing["bm25"][-1]
                                           + (time.perf_counter() - start) * 1000)
            for method, order in (("dense_minilm", dense_order), ("bm25", bm25_order),
                                  ("hybrid_z_075", hybrid_order)):
                records.append({"dataset": name, "query_id": query_id, "method": method,
                                "ranked_ids": [ids[int(index)] for index in order],
                                **metric(order, relevance[query_id], ids)})
        groups = {method: [row for row in records if row["dataset"] == name and row["method"] == method]
                  for method in protocol["methods"]}
        rng = np.random.default_rng(20260920)
        differences = np.asarray([hybrid["ndcg10"] - dense["ndcg10"]
                                  for hybrid, dense in zip(groups["hybrid_z_075"], groups["dense_minilm"] )])
        samples = differences[rng.integers(0, len(differences), size=(5000, len(differences)))].mean(axis=1)
        summary = {"dataset": name, "queries": len(query_ids), "documents": len(ids),
                   "hybrid_minus_dense_ndcg10": float(differences.mean()),
                   "hybrid_minus_dense_bootstrap95": [percentile(samples, 2.5), percentile(samples, 97.5)],
                   "hybrid_query_wins": int((differences > 0).sum()),
                   "hybrid_query_ties": int((differences == 0).sum()),
                   "hybrid_query_losses": int((differences < 0).sum()), "methods": {}}
        for method, rows in groups.items():
            summary["methods"][method] = {
                key: statistics.mean(row[key] for row in rows) for key in ("ndcg10", "recall10", "mrr10")
            }
            summary["methods"][method]["median_query_ms"] = statistics.median(timing[method])
            summary["methods"][method]["p95_query_ms"] = percentile(timing[method], 95)
        summaries.append(summary)
        manifests.append({"dataset": name, "data_sha256": hashes, "embedding_sha256": embedding_hashes,
                          "documents": len(ids), "queries": len(query_ids), "bm25_build_seconds": build_seconds})
        print(json.dumps(summary), flush=True)
    save(OUT / "results.json", records)
    save(OUT / "summary.json", summaries)
    save(OUT / "manifest.json", {
        "protocol_sha256": sha(OUT / "protocol.json"), "profile_identity": profile.identity,
        "datasets": manifests, "python": platform.python_version(), "numpy": np.__version__,
        "source_sha256": {path: sha(ROOT / path) for path in (
            "experiments/run_hybrid_retrieval.py", "context_stamps/hybrid.py")},
    })
    save(OUT / "checksums.json", {path.name: sha(path) for path in OUT.glob("*.json")
                                   if path.name != "checksums.json"})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--scidocs", type=Path, required=True)
    parser.add_argument("--scifact-cache", type=Path, required=True)
    parser.add_argument("--replication-cache", type=Path, required=True)
    parser.add_argument("--scidocs-cache", type=Path, required=True)
    main(parser.parse_args())

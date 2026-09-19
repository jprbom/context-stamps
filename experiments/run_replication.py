"""Frozen, no-retuning replication on NFCorpus and ArguAna; no corpus text export."""

import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
import platform
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from context_stamps.baselines import mmr_order
from context_stamps.selection import LinearSelector, rank_candidates, words


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8", newline="\n")


def graded_metrics(order, relevance, ids):
    gains = [max(0, relevance.get(ids[int(i)], 0)) for i in order[:10]]
    ideal = sorted((v for v in relevance.values() if v > 0), reverse=True)[:10]
    denom = sum(v / math.log2(i + 2) for i, v in enumerate(ideal))
    return {
        "ndcg10": sum(v / math.log2(i + 2) for i, v in enumerate(gains)) / denom if denom else 0.0,
        "recall10": sum(v > 0 for v in gains) / max(1, sum(v > 0 for v in relevance.values())),
    }


class BM25Index:
    """Same disclosed BM25 formula, with precomputed term counts for efficiency."""

    def __init__(self, documents):
        self.n = len(documents)
        lengths, postings = [], defaultdict(list)
        for i, text in enumerate(documents):
            tokens = re.findall(r"\w+", text.lower())
            lengths.append(len(tokens))
            for token, frequency in Counter(tokens).items():
                postings[token].append((i, frequency))
        self.lengths = np.array(lengths)
        self.average = max(1, float(self.lengths.mean()))
        self.postings = {key: np.array(value) for key, value in postings.items()}

    def score(self, query):
        result = np.zeros(self.n)
        for token in words(query):
            values = self.postings.get(token)
            if values is None:
                continue
            ids, tf = values[:, 0], values[:, 1]
            idf = math.log(1 + (self.n - len(ids) + 0.5) / (len(ids) + 0.5))
            result[ids] += idf * tf * 2.5 / (tf + 1.5 * (0.25 + 0.75 * self.lengths[ids] / self.average))
        return result


def run(data, out, cache):
    out.mkdir(parents=True, exist_ok=True)
    cache.mkdir(parents=True, exist_ok=True)
    protocol = json.loads(Path("evidence/replication-v1/protocol.json").read_text())
    if sha(Path("context_stamps/selection.py")) != protocol["selector_sha256"]:
        raise ValueError("frozen selector changed")
    model = SentenceTransformer(
        protocol["model"],
        revision=protocol["revision"],
        device="cpu",
        trust_remote_code=False,
        model_kwargs={"use_safetensors": True},
    )
    faiss.omp_set_num_threads(1)
    selector = LinearSelector.load("evidence/scifact-v1/selector-seed-7.json")
    manifests, all_records, summaries = [], [], []
    for name in protocol["datasets"]:
        root = data / name
        corpus = [
            json.loads(line) for line in (root / "corpus.jsonl").read_text(encoding="utf-8").splitlines()
        ]
        queries = {
            r["_id"]: r["text"]
            for r in map(json.loads, (root / "queries.jsonl").read_text(encoding="utf-8").splitlines())
        }
        qrels = defaultdict(dict)
        with (root / "qrels/test.tsv").open(encoding="utf-8") as stream:
            for row in csv.DictReader(stream, delimiter="\t"):
                qrels[row["query-id"]][row["corpus-id"]] = int(row["score"])
        qids = sorted(qrels)
        ids = [r["_id"] for r in corpus]
        positions = {value: i for i, value in enumerate(ids)}
        documents = [(r.get("title", "") + " " + r["text"]).strip() for r in corpus]
        hashes = {p: sha(root / p) for p in ["corpus.jsonl", "queries.jsonl", "qrels/test.tsv"]}
        key = hashlib.sha256(
            json.dumps([hashes, protocol["model"], protocol["revision"], qids], sort_keys=True).encode()
        ).hexdigest()
        start = time.perf_counter()
        hit = (cache / (key + "-D.npy")).exists() and (cache / (key + "-Q.npy")).exists()
        if hit:
            d = np.load(cache / (key + "-D.npy"), allow_pickle=False)
            q = np.load(cache / (key + "-Q.npy"), allow_pickle=False)
        else:
            print(f"{name}: encoding {len(documents)} documents and {len(qids)} queries", flush=True)
            d = model.encode(
                documents, batch_size=32, normalize_embeddings=True, show_progress_bar=True
            ).astype("float32")
            q = model.encode(
                [queries[qid] for qid in qids],
                batch_size=32,
                normalize_embeddings=True,
                show_progress_bar=True,
            ).astype("float32")
            np.save(cache / (key + "-D.npy"), d)
            np.save(cache / (key + "-Q.npy"), q)
        encode_seconds = time.perf_counter() - start
        start = time.perf_counter()
        lexical = BM25Index(documents)
        index = faiss.IndexFlatIP(d.shape[1])
        index.add(d)
        build_seconds = time.perf_counter() - start
        records = []
        for qi, qid in enumerate(qids):
            query = queries[qid]
            start = time.perf_counter()
            values, indexes = index.search(q[qi : qi + 1], 101)
            pool = [int(i) for i in indexes[0] if ids[int(i)] != qid][:100]
            dense_ms = (time.perf_counter() - start) * 1000
            scores = {int(i): float(s) for i, s in zip(indexes[0], values[0])}
            start = time.perf_counter()
            lex = lexical.score(query)
            if qid in positions:
                lex[positions[qid]] = -np.inf
            bm_order = np.argsort(-lex, kind="stable")[:10].tolist()
            bm_ms = (time.perf_counter() - start) * 1000
            start = time.perf_counter()
            selected = rank_candidates(
                query,
                [documents[i] for i in pool],
                [scores[i] for i in pool],
                coverage_weight=protocol["coverage_weight"],
                diversity_weight=protocol["diversity_weight"],
                pair_similarity=lambda i, j: float(d[pool[i]] @ d[pool[j]]),
            )
            coverage_ms = (time.perf_counter() - start) * 1000 + dense_ms
            start = time.perf_counter()
            mmr = mmr_order(
                [scores[i] for i in pool], d[pool], relevance_weight=protocol["mmr_relevance_weight"]
            )
            mmr_ms = (time.perf_counter() - start) * 1000 + dense_ms
            start = time.perf_counter()
            learned = sorted(pool, key=lambda i: -selector.score(query, documents[i], scores[i]))[:10]
            learned_ms = (time.perf_counter() - start) * 1000 + dense_ms
            orderings = {
                "dense": (pool[:10], dense_ms),
                "bm25": (bm_order, bm_ms),
                "coverage_diversity": ([pool[i] for i in selected], coverage_ms),
                "mmr": ([pool[i] for i in mmr], mmr_ms),
                "frozen_linear": (learned, learned_ms),
            }
            qw = words(query)
            missing_positive_ids = [
                did for did, rel in qrels[qid].items() if rel > 0 and did not in positions
            ]
            overlap = max(
                (
                    len(qw & words(documents[positions[did]]))
                    / max(1, len(qw | words(documents[positions[did]])))
                    for did, rel in qrels[qid].items()
                    if rel > 0 and did in positions
                ),
                default=0,
            )
            for method, (order, elapsed) in orderings.items():
                records.append(
                    {
                        "dataset": name,
                        "query_id": qid,
                        "method": method,
                        **graded_metrics(order, qrels[qid], ids),
                        "retrieval_selection_ms": elapsed,
                        "low_lexical_overlap": not missing_positive_ids and overlap <= 0.1,
                        "max_positive_jaccard": None if missing_positive_ids else overlap,
                        "missing_positive_ids": missing_positive_ids,
                        "ranked_ids": [ids[i] for i in order],
                    }
                )
            if qi % 100 == 0:
                print(f"{name}: {qi + 1}/{len(qids)} evaluated", flush=True)
        rng = np.random.default_rng(protocol["bootstrap_seed"])
        for stratum in ["all", "low_lexical_overlap"]:
            chosen = [r for r in records if stratum == "all" or r["low_lexical_overlap"]]
            dense_values = np.array([r["ndcg10"] for r in chosen if r["method"] == "dense"])
            if not len(dense_values):
                continue
            boot = rng.integers(0, len(dense_values), (protocol["bootstrap_samples"], len(dense_values)))
            for method in orderings:
                rows = [r for r in chosen if r["method"] == method]
                values = np.array([r["ndcg10"] for r in rows])
                summaries.append(
                    {
                        "dataset": name,
                        "stratum": stratum,
                        "method": method,
                        "n": len(rows),
                        "ndcg10": float(values.mean()),
                        "recall10": float(np.mean([r["recall10"] for r in rows])),
                        "retrieval_selection_ms": float(np.mean([r["retrieval_selection_ms"] for r in rows])),
                        "paired_delta_vs_dense": float((values - dense_values).mean()),
                        "paired_ci95": np.quantile(
                            (values - dense_values)[boot].mean(axis=1), [0.025, 0.975]
                        ).tolist(),
                    }
                )
        all_records.extend(records)
        manifests.append(
            {
                "dataset": name,
                "documents": len(ids),
                "queries": len(qids),
                "data_sha256": hashes,
                "encoding_seconds_or_cache_load": encode_seconds,
                "cache_hit": hit,
                "index_build_seconds": build_seconds,
            }
        )
    save(out / "summary.json", summaries)
    (out / "per-query.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in all_records), encoding="utf-8", newline="\n"
    )
    save(
        out / "manifest.json",
        {
            "protocol": protocol,
            "datasets": manifests,
            "python": platform.python_version(),
            "packages": {
                p: importlib.metadata.version(p)
                for p in ["numpy", "faiss-cpu", "sentence-transformers", "torch"]
            },
            "source_sha256": {
                p: sha(Path(p))
                for p in [
                    "experiments/run_replication.py",
                    "context_stamps/selection.py",
                    "context_stamps/baselines.py",
                ]
            },
            "selector_weights_sha256": sha(Path("evidence/scifact-v1/selector-seed-7.json")),
            "notes": "CPU; encoder cost excluded from per-query latency; no new fitting or tuning; query-bootstrap intervals are exploratory and not multiplicity adjusted",
        },
    )
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("evidence/replication-v1"))
    args = parser.parse_args()
    run(args.data, args.out, args.cache)

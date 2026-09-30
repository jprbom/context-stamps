"""Repository-disjoint, retrieval-only RepoQA-derived localization study.

Copyright (c) 2026 Prashant Jagtap. MIT License.
The external benchmark and model are downloaded separately. No benchmark source
or raw descriptions are written to the research evidence directory.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import re
import sys
import time
from functools import cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
from sklearn.feature_extraction.text import TfidfVectorizer  # noqa: E402

from stamps import Family, HashingEncoder, _planes  # noqa: E402

MODEL = "sentence-transformers/all-MiniLM-L6-v2"
REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
SOURCE_SHA256 = "c050a2ad90a7df89d9dc1f1c3b3b20683edd20a56293b35fcaae43dec115d681"
SOURCE_COMMIT = "e3a571033de99d0b9dcaccd25577a75d4b1c70b1"
SPLITS = {
    "train": ("psf/black", "python-poetry/poetry", "locustio/locust", "pyg-team/pytorch_geometric"),
    "validation": ("openai/openai-python", "mlc-ai/mlc-llm", "reactive-python/reactpy"),
    "final": ("marshmallow-code/marshmallow", "ethereum/web3.py", "Ciphey/Ciphey"),
}
BITS = {"semantic": 128, "task": 64, "entity": 32, "relation": 32}
LEXICAL = HashingEncoder(64)
FAMILIES = {
    "semantic": Family(f"{MODEL}@{REVISION}:function-text-v1", 384, 128, 202609301),
    "task": Family(LEXICAL.identity + ":body-v1", 64, 64, 202609302),
    "entity": Family(LEXICAL.identity + ":identifier-v1", 64, 32, 202609303),
    "relation": Family(LEXICAL.identity + ":dependency-v1", 64, 32, 202609304),
}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def identifier_words(text):
    return " ".join(re.findall(r"[A-Za-z][A-Za-z0-9_]*", text.replace("_", " "))) or "unknown"


def projection(name, values):
    return (np.asarray(values, dtype=np.float32) @ planes(name).T >= 0).astype(np.uint8)


@cache
def planes(name):
    return np.asarray(_planes(FAMILIES[name]), dtype=np.float32)


POPCOUNT = np.array([x.bit_count() for x in range(256)], dtype=np.uint8)


def bit_agreement(query_bits, candidate_bytes, bits):
    packed = np.packbits(query_bits, bitorder="little")
    mismatches = POPCOUNT[np.bitwise_xor(candidate_bytes, packed)].sum(axis=1)
    return 1.0 - mismatches / bits


def functions(repo):
    rows = []
    for path, items in sorted(repo["functions"].items()):
        content = repo["content"][path].encode("utf-8")
        for item in items:
            body = content[item["start_byte"]:item["end_byte"]].decode("utf-8")
            if not body.strip():
                continue
            if item["name"] not in body[:200]:
                raise ValueError(f"function offset does not match {repo['repo']}:{path}:{item['name']}")
            rows.append(dict(path=path, name=item["name"], start=item["start_byte"],
                             end=item["end_byte"], body=body))
    return rows


def encode_cache(repo, rows, model, cache):
    repo_key = hashlib.sha256((repo["repo"] + repo["commit_sha"] + SOURCE_SHA256 + REVISION +
                               "utf8-byte-offsets-v2").encode()).hexdigest()
    target = cache / (repo_key + ".npz")
    if target.exists():
        with np.load(target, allow_pickle=False) as saved:
            return saved["documents"], saved["queries"], True, 0.0
    text = [f"{r['path']}\n{r['name']}\n{r['body'][:1600]}" for r in rows]
    questions = [n["description"] for n in repo["needles"]]
    start = time.perf_counter()
    documents = model.encode(text, batch_size=64, normalize_embeddings=True,
                             convert_to_numpy=True, show_progress_bar=False).astype("float32")
    queries = model.encode(questions, batch_size=32, normalize_embeddings=True,
                           convert_to_numpy=True, show_progress_bar=False).astype("float32")
    duration = time.perf_counter() - start
    np.savez_compressed(target, documents=documents, queries=queries)
    return documents, queries, False, duration


def rank_scores(scores, correct):
    order = np.argsort(-scores, kind="stable")
    ranks = np.flatnonzero(np.isin(order, list(correct)))
    return int(ranks[0]) + 1 if len(ranks) else None


def run(source, out, cache, device):
    if sha(source) != SOURCE_SHA256:
        raise ValueError("RepoQA source checksum does not match the frozen release")
    source_data = json.loads(gzip.decompress(source.read_bytes()))
    repos = {r["repo"]: r for r in source_data["python"]}
    if set(repos) != {name for names in SPLITS.values() for name in names}:
        raise ValueError("repository split differs from frozen protocol")
    from sentence_transformers import SentenceTransformer

    cache.mkdir(parents=True, exist_ok=True)
    out.mkdir(parents=True, exist_ok=True)
    model = SentenceTransformer(MODEL, revision=REVISION, device=device,
                                trust_remote_code=False, model_kwargs={"use_safetensors": True})
    records, index = [], []
    for split, names in SPLITS.items():
        for name in names:
            repo = repos[name]
            rows = functions(repo)
            embeddings, query_embeddings, hit, embed_s = encode_cache(repo, rows, model, cache)
            texts = [f"{r['path']} {r['name']} {r['body'][:1600]}" for r in rows]
            vectorizer = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True,
                                         token_pattern=r"(?u)\b\w+\b", max_features=250000)
            t0 = time.perf_counter()
            lexical = vectorizer.fit_transform(texts)
            query_lexical = vectorizer.transform([n["description"] for n in repo["needles"]])
            tfidf_s = time.perf_counter() - t0
            t0 = time.perf_counter()
            candidate_bits = {
                "semantic": np.packbits(projection("semantic", embeddings), axis=1, bitorder="little"),
                "task": np.packbits(projection("task", [LEXICAL.encode(r["body"][:1200]) for r in rows]), axis=1, bitorder="little"),
                "entity": np.packbits(projection("entity", [LEXICAL.encode(identifier_words(r["path"] + " " + r["name"])) for r in rows]), axis=1, bitorder="little"),
                "relation": np.packbits(projection("relation", [LEXICAL.encode(identifier_words(" ".join(repo["dependency"].get(r["path"], [])[:12]))) for r in rows]), axis=1, bitorder="little"),
            }
            stamp_s = time.perf_counter() - t0
            stamp_bytes = sum(b.nbytes for b in candidate_bits.values())
            index.append(dict(repo=name, split=split, commit=repo["commit_sha"], functions=len(rows),
                              files=len(repo["content"]), cache_hit=hit, dense_embed_s=embed_s,
                              tfidf_index_s=tfidf_s, stamp_encode_s=stamp_s,
                              dense_vector_bytes=embeddings.nbytes, stamp_payload_bytes=int(stamp_bytes)))
            for task_i, needle in enumerate(repo["needles"]):
                correct = {i for i, r in enumerate(rows) if r["path"] == needle["path"]
                           and r["name"] == needle["name"] and r["start"] == needle["start_byte"]}
                if len(correct) != 1:
                    raise ValueError(f"needle is not uniquely indexed: {name}:{task_i}")
                q = needle["description"]
                t0 = time.perf_counter()
                dense = embeddings @ query_embeddings[task_i]
                dense_ms = 1000 * (time.perf_counter() - t0)
                t0 = time.perf_counter()
                lexical_scores = (lexical @ query_lexical[task_i].T).toarray().ravel()
                lexical_ms = 1000 * (time.perf_counter() - t0)
                t0 = time.perf_counter()
                observed = {
                    "semantic": projection("semantic", query_embeddings[task_i:task_i + 1])[0],
                    "task": projection("task", [LEXICAL.encode(q)])[0],
                }
                # An entity view is admitted only when the query explicitly quotes an identifier.
                quoted = re.findall(r"`([A-Za-z_][A-Za-z0-9_]*)`", q)
                if quoted:
                    observed["entity"] = projection("entity", [LEXICAL.encode(" ".join(quoted))])[0]
                stamped = sum(bit_agreement(bits, candidate_bits[facet], BITS[facet]) * BITS[facet]
                              for facet, bits in observed.items()) / sum(BITS[f] for f in observed)
                stamp_ms = 1000 * (time.perf_counter() - t0)
                # Scores only; selector weights are fitted on TRAIN after all retrieval calls.
                records.append(dict(id=f"{name}:{task_i}", repo=name, split=split,
                                    target=next(iter(correct)), candidates=len(rows),
                                    dense_rank=rank_scores(dense, correct),
                                    tfidf_rank=rank_scores(lexical_scores, correct),
                                    stamp_rank=rank_scores(stamped, correct),
                                    dense_ms=dense_ms, tfidf_ms=lexical_ms, stamp_ms=stamp_ms,
                                    observed_facets=sorted(observed),
                                    target_dense=float(dense[next(iter(correct))]),
                                    target_tfidf=float(lexical_scores[next(iter(correct))]),
                                    target_stamp=float(stamped[next(iter(correct))]),
                                    dense_top10=np.argsort(-dense, kind="stable")[:10].tolist(),
                                    tfidf_top10=np.argsort(-lexical_scores, kind="stable")[:10].tolist(),
                                    stamp_top10=np.argsort(-stamped, kind="stable")[:10].tolist()))
            print(f"{split}: {name}: {len(rows)} functions / {len(repo['needles'])} queries", flush=True)
    # The frozen methods are not modified using validation or final labels.
    methods = ("dense", "tfidf", "stamp")
    def summarize(group):
        result = {}
        for method in methods:
            ranks = [r[f"{method}_rank"] for r in group]
            result[method] = dict(top1=sum(x == 1 for x in ranks), top5=sum(x is not None and x <= 5 for x in ranks),
                                  top10=sum(x is not None and x <= 10 for x in ranks),
                                  mrr=round(sum(1 / x for x in ranks if x) / len(ranks), 6),
                                  median_lookup_ms=round(float(np.median([r[f"{method}_ms"] for r in group])), 4))
        return result
    summary = {split: dict(queries=len(group), methods=summarize(group)) for split in SPLITS
               if (group := [r for r in records if r["split"] == split])}
    payload = dict(protocol="RepoQA-derived Python function localization; exact indexed path/name/start-byte; not native RepoQA code-similarity or patch pass.",
                   source=dict(url="https://github.com/evalplus/repoqa_release", commit=SOURCE_COMMIT,
                               file="repoqa-2024-06-23.json.gz", sha256=SOURCE_SHA256),
                   model=dict(id=MODEL, revision=REVISION, device=device),
                   preprocessing=dict(document="path + function name + first 1600 code characters; model max sequence applies",
                                      query="official needle description; model max sequence applies",
                                      tfidf="word unigram/bigram, sublinear tf; fit per repository",
                                      stamp="256-bit four-view document; query observes semantic/task and explicitly quoted entity only"),
                   splits=SPLITS, stamp_bits=BITS, summary=summary, index=index, records=records,
                   limitations="Candidate source and queries excluded from Git. Same-repository source is indexed; repository split separates tuning/evaluation, not indexed corpus. No reader, patch test, authorization treatment, cold model load, or end-to-end agent timing in this study.")
    (out / "results.json").write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=ROOT / "evidence/repoqa-localization-v1")
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    run(args.source, args.out, args.cache, args.device)


if __name__ == "__main__":
    main()

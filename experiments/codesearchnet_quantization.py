"""Frozen, leakage-controlled CodeSearchNet Python quantization study.

This is a derived, repository-local function-localization task, not the
official CodeSearchNet challenge. Third-party snippets stay outside Git.
Copyright (c) 2026 Prashant Jagtap. MIT License.
"""

import argparse
import ast
import hashlib
import json
import re
import statistics
import textwrap
import time
import warnings
from collections import Counter, defaultdict
from pathlib import Path

import faiss
import numpy as np
import pyarrow.parquet as pq
from sentence_transformers import SentenceTransformer

MODEL = "sentence-transformers/all-MiniLM-L6-v2"
REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
SOURCE_SHA = {
    "train": "ad9e3a4ab10c2c1d8926d2b26ca2bfcc3aadda1477ba29a933391f93806b9fed",
    "test": "3167e79ee7f081d825bf97b96d3a6b2d96428b00f6a98125be943384d8afae5f",
    "validation": "22eaacb46ed7e74d582409b85692ef63f5a43e99f9395c2eb736b5c8451422bb",
}
SEED = 20260930


def digest(value):
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def file_sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


class StripDocstrings(ast.NodeTransformer):
    def visit_FunctionDef(self, node):
        self.generic_visit(node)
        self.strip(node)
        return node

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_ClassDef(self, node):
        self.generic_visit(node)
        self.strip(node)
        return node

    @staticmethod
    def strip(node):
        if node.body and isinstance(node.body[0], ast.Expr) and isinstance(node.body[0].value, ast.Constant) and isinstance(node.body[0].value.value, str):
            node.body.pop(0)
        if not node.body:
            node.body.append(ast.Pass())


def clean(code, description):
    if not code or not description or not description.strip():
        return None
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", SyntaxWarning)
            tree = ast.parse(textwrap.dedent(code))
        tree = StripDocstrings().visit(tree)
        rendered = ast.unparse(tree)
    except (SyntaxError, ValueError, TypeError, RecursionError):
        return None
    def normalized(s):
        return re.sub(r"\s+", " ", s).strip().casefold()
    if len(normalized(description)) < 20 or len(normalized(description)) > 400:
        return None
    if normalized(description) in normalized(rendered):
        return None
    return rendered[:3000]


def rows(path, train=False):
    if file_sha(path) != SOURCE_SHA[path.name.split("-")[1].split(".")[0]]:
        raise ValueError(f"source SHA changed: {path}")
    pf = pq.ParquetFile(path)
    result = []
    for batch in pf.iter_batches(batch_size=2048, columns=["repository_name", "func_code_string", "func_documentation_string", "func_code_url"]):
        data = batch.to_pydict()
        for repo, code, desc, url in zip(*(data[k] for k in ["repository_name", "func_code_string", "func_documentation_string", "func_code_url"])):
            if train and int(digest(url)[:8], 16) % 25 != 0:
                continue
            rendered = clean(code, desc)
            if rendered is None:
                continue
            result.append({"repo": repo, "code": rendered, "query": desc.strip(), "id": digest(url)[:20]})
    return result


def cohort(data, label, max_repos=30, queries_per_repo=3):
    grouped = defaultdict(list)
    for row in data:
        grouped[row["repo"]].append(row)
    eligible = []
    for repo, items in grouped.items():
        if not 50 <= len(items) <= 1000:
            continue
        counts = Counter(re.sub(r"\s+", " ", r["query"].casefold()) for r in items)
        possible = [i for i, row in enumerate(items) if counts[re.sub(r"\s+", " ", row["query"].casefold())] == 1]
        if len(possible) >= queries_per_repo:
            eligible.append(repo)
    chosen = sorted(eligible, key=lambda x: digest(f"{SEED}:{label}:{x}"))[:max_repos]
    output = []
    for repo in chosen:
        items = sorted(grouped[repo], key=lambda x: x["id"])
        counts = Counter(re.sub(r"\s+", " ", r["query"].casefold()) for r in items)
        possible = [i for i, row in enumerate(items) if counts[re.sub(r"\s+", " ", row["query"].casefold())] == 1]
        targets = sorted(possible, key=lambda i: digest(f"{SEED}:{label}:{items[i]['id']}"))[:queries_per_repo]
        output.append({"repo": repo, "items": items, "targets": targets})
    return output


def prepare(source, cache):
    cache.mkdir(parents=True, exist_ok=True)
    meta = {"dataset": "code-search-net/code_search_net", "task": "derived repo-local function localization", "seed": SEED, "model": MODEL, "revision": REVISION, "splits": {}}
    for split in ("train", "validation", "test"):
        path = source / f"python-{split}.parquet"
        if not path.exists():
            raise FileNotFoundError(path)
        data = rows(path, train=(split == "train"))
        if split == "train":
            data = sorted(data, key=lambda r: digest(f"{SEED}:{r['id']}"))[:16000]
            chosen = data
            info = {"clean_train_pairs": len(data), "repositories": len({r['repo'] for r in data})}
        else:
            chosen = cohort(data, split)
            info = {"clean_functions": len(data), "eligible_repositories": len({r['repo'] for r in data}), "selected_repositories": [r["repo"] for r in chosen], "selected_queries": sum(len(r["targets"]) for r in chosen), "selected_candidates": sum(len(r["items"]) for r in chosen)}
        out = cache / f"{split}.json"
        out.write_text(json.dumps(chosen, ensure_ascii=False), encoding="utf-8")
        info["source_sha256"] = file_sha(path)
        info["prepared_sha256"] = file_sha(out)
        meta["splits"][split] = info
    (cache / "manifest.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(meta, indent=2))


def embed(cache):
    model = SentenceTransformer(MODEL, revision=REVISION, device="cuda")
    for split in ("train", "validation", "test"):
        data = json.loads((cache / f"{split}.json").read_text(encoding="utf-8"))
        if split == "train":
            docs, queries = [r["code"] for r in data], [r["query"] for r in data]
        else:
            docs = [r["code"] for repo in data for r in repo["items"]]
            queries = [repo["items"][i]["query"] for repo in data for i in repo["targets"]]
        t0 = time.perf_counter()
        d = model.encode(docs, batch_size=128, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False).astype("float32")
        q = model.encode(queries, batch_size=128, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False).astype("float32")
        np.savez_compressed(cache / f"{split}-vectors.npz", docs=d, queries=q)
        print(split, len(d), len(q), round(time.perf_counter() - t0, 1), "seconds", flush=True)


def metric(ranks):
    return {"n": len(ranks), "top1": sum(r == 1 for r in ranks), "top10": sum(r <= 10 for r in ranks), "mrr": round(sum(1 / r for r in ranks) / len(ranks), 6)}


def evaluate(cache, split, methods):
    cohorts = json.loads((cache / f"{split}.json").read_text(encoding="utf-8"))
    vectors = np.load(cache / f"{split}-vectors.npz")
    docs, queries = vectors["docs"], vectors["queries"]
    doc_offset = query_offset = 0
    ranks = {name: [] for name in methods}
    timing = {name: [] for name in methods}
    per_repo = []
    for cohort in cohorts:
        count, nq = len(cohort["items"]), len(cohort["targets"])
        d, q = np.ascontiguousarray(docs[doc_offset:doc_offset+count]), np.ascontiguousarray(queries[query_offset:query_offset+nq])
        actual = np.asarray(cohort["targets"])
        row = {"repo": cohort["repo"], "candidates": count, "ranks": {}}
        for name, (template, doc_matrix, query_matrix) in methods.items():
            index = faiss.clone_index(template)
            index.reset()
            indexed = np.ascontiguousarray(d @ doc_matrix) if doc_matrix is not None else d
            index.add(indexed)
            # Warm search. Include neither encoding nor index construction.
            request = np.ascontiguousarray(q @ query_matrix) if query_matrix is not None else q
            index.search(request, count)
            samples = []
            for _ in range(11):
                start = time.perf_counter()
                request = np.ascontiguousarray(q @ query_matrix) if query_matrix is not None else q
                _, order = index.search(request, count)
                samples.append((time.perf_counter() - start) * 1000 / nq)
            found = [int(np.flatnonzero(order[i] == target)[0]) + 1 for i, target in enumerate(actual)]
            ranks[name].extend(found)
            row["ranks"][name] = found
            timing[name].append(statistics.median(samples))
        per_repo.append(row)
        doc_offset += count
        query_offset += nq
    assert doc_offset == len(docs) and query_offset == len(queries)
    return {"summary": {name: metric(r) for name, r in ranks.items()}, "warm_lookup_median_ms_per_query": {name: round(statistics.median(t), 6) for name, t in timing.items()}, "rows": per_repo}


def fit(cache, out, run_test=False):
    faiss.omp_set_num_threads(4)
    train = np.load(cache / "train-vectors.npz")
    x = np.ascontiguousarray(train["docs"])
    flat = faiss.IndexFlatIP(384)
    pq_index = faiss.IndexPQ(384, 32, 8, faiss.METRIC_INNER_PRODUCT)
    pq_index.pq.cp.min_points_per_centroid = 16
    pq_index.train(x)
    opq = faiss.OPQMatrix(384, 32)
    opq.niter = 12
    opq.niter_pq = 4
    opq.pq = faiss.ProductQuantizer(384, 32, 8)
    opq.pq.cp.min_points_per_centroid = 16
    opq_index = faiss.IndexPQ(384, 32, 8, faiss.METRIC_INNER_PRODUCT)
    opq_index.pq.cp.min_points_per_centroid = 16
    transformed = faiss.IndexPreTransform(opq, opq_index)
    t0 = time.perf_counter()
    transformed.train(x)
    fit_seconds = time.perf_counter() - t0
    # Expected inner-product error depends on the query distribution. A
    # reversible covariance transform weights document quantization toward
    # directions used by training queries while preserving exact float scores:
    # (d A) . (q A^-T) == d . q. Only quantization makes the routes differ.
    qtrain = np.ascontiguousarray(train["queries"])
    covariance = (qtrain.T @ qtrain) / len(qtrain)
    eigenvalues, basis = np.linalg.eigh(covariance)
    floor = max(float(np.mean(eigenvalues)) * 0.1, 1e-7)
    eigenvalues = np.maximum(eigenvalues, floor)
    methods = {"dense": (flat, None, None), "pq32": (pq_index, None, None), "opq32": (transformed, None, None)}
    fit_times = {}
    for alpha in (0.25, 0.5):
        scale = (eigenvalues / np.mean(eigenvalues)) ** alpha
        doc_matrix = np.ascontiguousarray((basis * scale) @ basis.T, dtype="float32")
        query_matrix = np.ascontiguousarray((basis / scale) @ basis.T, dtype="float32")
        index = faiss.IndexPQ(384, 32, 8, faiss.METRIC_INNER_PRODUCT)
        index.pq.cp.min_points_per_centroid = 16
        start = time.perf_counter()
        index.train(np.ascontiguousarray(x @ doc_matrix))
        fit_times[f"query_weighted_pq32_alpha{alpha}"] = time.perf_counter() - start
        methods[f"query_weighted_pq32_alpha{alpha}"] = (index, doc_matrix, query_matrix)
    result = {"method": "frozen CodeSearchNet Python derived task", "source_manifest": json.loads((cache / "manifest.json").read_text(encoding="utf-8")), "faiss_version": faiss.__version__, "threads": 4, "opq_fit_seconds": fit_seconds, "query_weighted_fit_seconds": fit_times, "codebook_bytes": {name: len(faiss.serialize_index(index)) + (0 if dm is None else dm.nbytes + qm.nbytes) for name, (index, dm, qm) in methods.items()}, "validation": evaluate(cache, "validation", methods)}
    if run_test:
        result["test"] = evaluate(cache, "test", methods)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({k: result[k]["summary"] for k in ("validation", "test") if k in result}, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("phase", choices=["prepare", "embed", "fit"])
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--cache", type=Path, required=True)
    p.add_argument("--out", type=Path, default=Path("evidence/codesearchnet-quantization-v1/results.json"))
    p.add_argument("--run-test", action="store_true")
    args = p.parse_args()
    {"prepare": lambda: prepare(args.source, args.cache), "embed": lambda: embed(args.cache), "fit": lambda: fit(args.cache, args.out, args.run_test)}[args.phase]()

"""Freeze dense and 256-bit multi-facet source packets for a coding pilot.

Evaluation material is read locally and is never used for fitting. The stamp is
only a retrieval handle; source text and hashes remain in the host index.
Copyright (c) 2026 Prashant Jagtap. MIT License.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
from sentence_transformers import SentenceTransformer  # noqa: E402
from stamps import Family, HashingEncoder, _planes  # noqa: E402

MODEL = "sentence-transformers/all-MiniLM-L6-v2"
REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
BITS = {"semantic": 96, "task": 32, "entity": 32, "relation": 32,
        "temporal": 16, "authority": 16, "policy": 16, "modality": 16}
LEXICAL = HashingEncoder(64)


def sections(source: str):
    tree = ast.parse(source)
    lines = source.splitlines()
    rows = []

    def add(node, parent=""):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            name = parent + node.name
            body = "\n".join(lines[node.lineno - 1:node.end_lineno])
            if body.strip():
                rows.append({"name": name, "start": node.lineno, "end": node.end_lineno,
                             "text": body[:2600]})
            if isinstance(node, ast.ClassDef):
                for child in node.body:
                    if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        add(child, name + ".")

    for node in tree.body:
        add(node)
    return rows


def words(text):
    return " ".join(re.findall(r"[A-Za-z][A-Za-z0-9_]*", text.replace("_", " ")))


def bit_codes(vectors, family):
    planes = np.asarray(_planes(family), dtype=np.float32)
    array = np.asarray(vectors, dtype=np.float32)
    return np.packbits(array @ planes.T >= 0, axis=1, bitorder="little")


def packet(rows, ordering, source_hash):
    chosen, used = [], 0
    for index in ordering:
        row = rows[int(index)]
        snippet = row["text"][:2200]
        if used + len(snippet) > 4200:
            continue
        chosen.append({"path": "/app/bottle.py", "symbol": row["name"],
                       "lines": [row["start"], row["end"]],
                       "source_sha256": source_hash, "text": snippet})
        used += len(snippet)
        if len(chosen) == 3:
            break
    # Only the selected, bounded snippets may enter the model prompt.
    text = "\n\n".join(f"/app/bottle.py:{x['lines'][0]}-{x['lines'][1]} {x['symbol']}\n{x['text']}"
                       for x in chosen)
    return {"text": text, "sources": [{k: v for k, v in x.items() if k != "text"} for x in chosen]}


def run(source, instruction, out, device):
    raw = source.read_bytes()
    source_hash = hashlib.sha256(raw).hexdigest()
    rows = sections(raw.decode("utf-8"))
    query = instruction.read_text(encoding="utf-8")
    model = SentenceTransformer(MODEL, revision=REVISION, device=device,
                                trust_remote_code=False, model_kwargs={"use_safetensors": True})
    started = time.perf_counter()
    doc_text = [f"bottle.py {r['name']}\n{r['text']}" for r in rows]
    documents = model.encode(doc_text, batch_size=64, normalize_embeddings=True,
                             convert_to_numpy=True, show_progress_bar=False).astype(np.float32)
    question = model.encode([query], normalize_embeddings=True, convert_to_numpy=True,
                            show_progress_bar=False).astype(np.float32)
    embed_seconds = time.perf_counter() - started
    dense_scores = (documents @ question[0]).ravel()
    dense_order = np.argsort(-dense_scores, kind="stable")
    families = {name: Family(f"{MODEL}@{REVISION}" if name == "semantic" else LEXICAL.identity,
                             384 if name == "semantic" else 64, bits, 202610020 + i)
                for i, (name, bits) in enumerate(BITS.items())}
    # All eight views are packed for documents. Only semantic and observed task
    # text participate in this query; absent policy/time/etc. are not invented.
    features = {"semantic": documents,
                "task": [LEXICAL.encode(r["text"][:1200]) for r in rows],
                "entity": [LEXICAL.encode(words(r["name"])) for r in rows],
                "relation": [LEXICAL.encode(words(r["text"][:500])) for r in rows],
                "temporal": [LEXICAL.encode("undated") for _ in rows],
                "authority": [LEXICAL.encode("repository source") for _ in rows],
                "policy": [LEXICAL.encode("local coding") for _ in rows],
                "modality": [LEXICAL.encode("python code") for _ in rows]}
    started = time.perf_counter()
    codes = {name: bit_codes(values, families[name]) for name, values in features.items()}
    query_codes = {"semantic": bit_codes(question, families["semantic"])[0],
                   "task": bit_codes([LEXICAL.encode(query)], families["task"])[0]}
    pop = np.asarray([i.bit_count() for i in range(256)], dtype=np.uint8)
    mismatch = sum(pop[np.bitwise_xor(codes[name], q)].sum(axis=1).astype(np.float32)
                   for name, q in query_codes.items())
    stamp_scores = 1 - mismatch / sum(BITS[name] for name in query_codes)
    stamp_order = np.argsort(-stamp_scores, kind="stable")
    stamp_seconds = time.perf_counter() - started
    overlap_top3 = len(set(map(int, dense_order[:3])) & set(map(int, stamp_order[:3])))
    overlap_top10 = len(set(map(int, dense_order[:10])) & set(map(int, stamp_order[:10])))
    route_status = "abstain" if overlap_top3 == 0 else "exploratory_only"
    out.mkdir(parents=True, exist_ok=False)
    result = {"schema": 1, "task": "fix-code-vulnerability", "task_revision":
              "7131e4375048a0e408a8fb404b5f499d726b695b", "source_sha256": source_hash,
              "instruction_sha256": hashlib.sha256(instruction.read_bytes()).hexdigest(),
              "encoder": f"{MODEL}@{REVISION}", "functions": len(rows), "facet_bits": BITS,
              "index_bytes": {"dense_float32": int(documents.nbytes),
                              "stamp_payload": int(sum(c.nbytes for c in codes.values()))},
              "build_seconds": {"embedding": embed_seconds, "stamp": stamp_seconds},
              "dense": packet(rows, dense_order, source_hash),
              "stamp": packet(rows, stamp_order, source_hash),
              "top10": {"dense": [rows[int(i)]["name"] for i in dense_order[:10]],
                        "stamp": [rows[int(i)]["name"] for i in stamp_order[:10]]},
              "agreement": {"top3_overlap": overlap_top3, "top10_overlap": overlap_top10,
                            "route_status": route_status,
                            "meaning": "disagreement is a veto, agreement alone is not relevance proof"}}
    (out / "packets.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"functions": len(rows), "index_bytes": result["index_bytes"],
                      "top3": {arm: [s["symbol"] for s in result[arm]["sources"]]
                               for arm in ("dense", "stamp")},
                      "agreement": result["agreement"]}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--instruction", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    run(args.source, args.instruction, args.out, args.device)

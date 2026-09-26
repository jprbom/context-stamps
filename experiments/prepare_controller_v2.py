"""Pinned public-data preparation and local teacher inference for controller-v2."""

import argparse
import hashlib
import json
import os
import shutil
import sys
import time
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))
from run_hybrid_retrieval import BM25, labels, metric, read, save, sha  # noqa: E402

OUT = ROOT / "evidence/controller-v2"
NAMES = ("scifact", "nfcorpus", "arguana", "scidocs", "fiqa")


def normalize(text):
    return " ".join(text.casefold().split())


def acquire_fiqa(work, protocol):
    archive = work / "fiqa.zip"
    if not archive.exists():
        request = urllib.request.Request(protocol["fiqa_url"], headers={"User-Agent": "context-stamps-research"})
        with urllib.request.urlopen(request, timeout=60) as response, archive.open("wb") as target:  # nosec B310
            total = 0
            while chunk := response.read(1024 * 1024):
                total += len(chunk)
                if total > 128 * 1024 * 1024:
                    raise ValueError("archive download exceeds budget")
                target.write(chunk)
    # Published legacy archive fingerprint plus a newly recorded SHA256; not authentication.
    if hashlib.md5(archive.read_bytes(), usedforsecurity=False).hexdigest() != protocol["fiqa_archive_md5"]:
        raise ValueError("FiQA archive differs from published registry")
    destination = work / "public-validation/fiqa"
    with zipfile.ZipFile(archive) as handle:
        for name in ("corpus.jsonl", "queries.jsonl", "qrels/dev.tsv", "qrels/test.tsv"):
            member = handle.getinfo("fiqa/" + name)
            if member.file_size > 512 * 1024 * 1024:
                raise ValueError("unexpected archive member size")
            path = destination / name
            path.parent.mkdir(parents=True, exist_ok=True)
            # Fixed whitelist destinations, no archive-controlled path extraction.
            with handle.open(member) as source, path.open("wb") as target:
                shutil.copyfileobj(source, target)
    return destination


def encoder_model(protocol):
    from transformers import AutoModel, AutoTokenizer
    kwargs = dict(revision=protocol["encoder_revision"], local_files_only=True, trust_remote_code=False)
    return (AutoTokenizer.from_pretrained(protocol["encoder"], **kwargs),
            AutoModel.from_pretrained(protocol["encoder"], use_safetensors=True, **kwargs).cuda().eval())


def encode(texts, tokenizer, model):
    rows = []
    with torch.inference_mode():
        for offset in range(0, len(texts), 128):
            inputs = tokenizer(texts[offset:offset + 128], padding=True, truncation=True,
                               max_length=256, return_tensors="pt").to("cuda")
            hidden = model(**inputs).last_hidden_state.float()
            mask = inputs["attention_mask"].unsqueeze(-1)
            pooled = (hidden * mask).sum(1) / mask.sum(1).clamp_min(1)
            rows.append(torch.nn.functional.normalize(pooled, dim=-1).cpu().numpy())
    return np.concatenate(rows).astype(np.float32)


def teacher_model(protocol):
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    kwargs = dict(revision=protocol["teacher_revision"], local_files_only=True, trust_remote_code=False)
    return (AutoTokenizer.from_pretrained(protocol["teacher"], **kwargs),
            AutoModelForSequenceClassification.from_pretrained(protocol["teacher"], use_safetensors=True,
                **kwargs).cuda().eval())


def teacher_scores(queries, passages, tokenizer, model, max_length=512):
    values = []
    with torch.inference_mode():
        for offset in range(0, len(queries), 64):
            inputs = tokenizer(queries[offset:offset + 64], passages[offset:offset + 64], padding=True,
                               truncation=True, max_length=max_length, return_tensors="pt").to("cuda")
            with torch.autocast("cuda", dtype=torch.bfloat16):
                prediction = model(**inputs).logits[:, 0].float()
            values.extend(prediction.cpu().tolist())
    return np.asarray(values, dtype=np.float32)


def standardized(values):
    return (values - values.mean()) / max(float(values.std()), 1e-9)


def main(work):
    protocol = read(OUT / "protocol.json")
    destination = work / "controller-v2-data"
    destination.mkdir(parents=True, exist_ok=True)
    acquire_fiqa(work, protocol)
    old_split = read(ROOT / "evidence/controller-v1/split-manifest.json")
    old_data = read(ROOT / "evidence/controller-v1/data-preparation.json")
    old_checksums = {(r["dataset"], r["split"]): r["array_sha256"] for r in old_data["partitions"]}
    raw, enc = {}, None
    for name in NAMES:
        source = work / ("public-validation/" + name if name in ("scidocs", "fiqa") else "replication-data/" + name)
        docs = [json.loads(line) for line in (source / "corpus.jsonl").read_text(encoding="utf-8").splitlines()]
        queries = {r["_id"]: r["text"] for r in map(json.loads, (source / "queries.jsonl").read_text(encoding="utf-8").splitlines())}
        texts = [(r.get("title", "") + " " + r["text"]).strip() for r in docs]
        judgments = {split: labels(source / ("qrels/" + split + ".tsv")) for split in ("train", "dev", "test")
                     if (source / ("qrels/" + split + ".tsv")).exists() and not (name == "fiqa" and split == "train")}
        if name == "fiqa":
            parts = {"calibration": sorted(judgments["dev"]), "test": sorted(judgments["test"])}
            if enc is None:
                enc = encoder_model(protocol)
            dpath = destination / "fiqa-D.npy"
            if not dpath.exists():
                print("Encoding new FiQA corpus", len(texts), flush=True)
                np.save(dpath, encode(texts, *enc), allow_pickle=False)
            D = np.load(dpath, allow_pickle=False)
            qids = sorted(set(parts["calibration"] + parts["test"]))
            qvectors = dict(zip(qids, encode([queries[q] for q in qids], *enc)))
        else:
            parts = {split: ids for split, ids in old_split["assignments"][name].items() if ids}
            dpath = work / ("controller-v1-data/" + name + "-D.npy")
            if sha(dpath) != old_data["inherited_embeddings"][name]["documents"]:
                raise ValueError("document cache changed")
            D, qvectors = np.load(dpath, allow_pickle=False), {}
            for split, qids in parts.items():
                path = work / ("controller-v1-data/" + name + "-" + split + ".npz")
                if sha(path) != old_checksums[(name, split)]:
                    raise ValueError("old query cache changed")
                meta = read(path.with_suffix(".json"))
                if meta["query_ids"] != qids:
                    raise ValueError("query ordering differs from split manifest")
                qvectors.update(zip(qids, np.load(path, allow_pickle=False)["Q"]))
            for filename, expected in old_split["data_sha256"][name].items():
                if sha(source / filename) != expected:
                    raise ValueError("old raw data changed")
        raw[name] = dict(D=D, dpath=dpath, queries=queries, qvectors=qvectors, parts=parts,
                         docs=docs, texts=texts, source=source, judgments=judgments)
    # New-domain calibration and test must also be disjoint by normalized text.
    holdouts = {normalize(raw["fiqa"]["queries"][q]) for ids in raw["fiqa"]["parts"].values() for q in ids}
    seen = {split: set() for split in ("train", "tune", "calibration", "test")}
    exclusions = []
    for name, r in raw.items():
        for split, qids in r["parts"].items():
            keep = []
            for qid in qids:
                text = normalize(r["queries"][qid])
                if split == "train" and name != "fiqa" and text in holdouts:
                    exclusions.append(dict(dataset=name, query_id=qid, reason="new_holdout_overlap"))
                    continue
                seen[split].add(text)
                keep.append(qid)
            r["parts"][split] = keep
    for a, b in (("train", "tune"), ("train", "calibration"), ("train", "test"),
                 ("tune", "calibration"), ("tune", "test"), ("calibration", "test")):
        if seen[a] & seen[b]:
            raise ValueError("query overlap between " + a + " and " + b)
    if enc is not None:
        del enc
        torch.cuda.empty_cache()
    tokenizer, teacher = teacher_model(protocol)
    metadata = dict(protocol_sha256=sha(OUT / "protocol.json"),
                    assignments={n: r["parts"] for n, r in raw.items()}, exclusions=exclusions,
                    fiqa_archive_sha256=sha(work / "fiqa.zip"), data_sha256={}, partitions=[])
    for name, r in raw.items():
        metadata["data_sha256"][name] = {str(p.relative_to(r["source"])).replace("\\", "/"): sha(p)
            for p in r["source"].rglob("*") if p.is_file() and p.suffix in (".tsv", ".jsonl")}
        lexical = BM25(r["texts"])
        doc_ids = [d["_id"] for d in r["docs"]]
        positions = {key: i for i, key in enumerate(doc_ids)}
        judgments = {q: value for split in r["judgments"].values() for q, value in split.items()}
        for split, qids in r["parts"].items():
            target = destination / (name + "-" + split + ".npz")
            records_path = target.with_suffix(".json")
            # Checkpoint each expensive teacher partition for interrupted local runs.
            marker = target.with_suffix(".ready.json")
            if marker.exists():
                info = read(marker)
                if info["protocol_sha256"] != metadata["protocol_sha256"] or sha(target) != info["array_sha256"] or sha(records_path) != info["metadata_sha256"]:
                    raise ValueError("prepared partition changed")
                metadata["partitions"].append(info)
                print("Reused", name, split, flush=True)
                continue
            start = time.perf_counter()
            Q = np.asarray([r["qvectors"][q] for q in qids], dtype=np.float32)
            candidate = np.zeros((len(qids), 256), dtype=np.int32)
            features = np.zeros((len(qids), 256, 6), dtype=np.float32)
            relevance = np.zeros((len(qids), 256), dtype=np.float32)
            mask = np.zeros((len(qids), 256), dtype=bool)
            teacher_values = np.zeros((len(qids), 256), dtype=np.float32)
            records = []
            for i, qid in enumerate(qids):
                ds, ls = (r["D"] @ Q[i]).astype(np.float64), lexical.score(r["queries"][qid])
                eligible = np.ones(len(ds), dtype=bool)
                if qid in positions:
                    eligible[positions[qid]] = False
                dz = (ds - ds[eligible].mean()) / max(float(ds[eligible].std()), 1e-9)
                lz = (ls - ls[eligible].mean()) / max(float(ls[eligible].std()), 1e-9)
                hs = .75 * dz + .25 * lz
                orders = [np.argsort(-np.where(eligible, scores, -np.inf), kind="stable") for scores in (ds, hs, ls)]
                old = sorted(set(orders[0][:48]) | set(orders[1][:48]) | set(orders[2][:32]))
                chosen = sorted(set(orders[0][:96]) | set(orders[1][:96]) | set(orders[2][:64]))
                count = len(chosen)
                candidate[i, :count], mask[i, :count] = chosen, True
                ranks = [np.empty(len(ds), dtype=np.int32), np.empty(len(ds), dtype=np.int32)]
                ranks[0][orders[0]], ranks[1][orders[2]] = np.arange(len(ds)), np.arange(len(ds))
                features[i, :count] = np.stack((ds[chosen], dz[chosen], hs[chosen], lz[chosen],
                                   1 / (1 + ranks[0][chosen]), 1 / (1 + ranks[1][chosen])), axis=1)
                rel = judgments[qid]
                relevance[i, :count] = [max(0, rel.get(doc_ids[c], 0)) for c in chosen]
                ideal = sorted((v for v in rel.values() if v > 0), reverse=True)[:10]
                denom = sum(v / np.log2(j + 2) for j, v in enumerate(ideal))
                def coverage(indices):
                    gains = sorted((max(0, rel.get(doc_ids[c], 0)) for c in indices), reverse=True)
                    return dict(recall=sum(v > 0 for v in gains) / max(1, sum(v > 0 for v in rel.values())),
                                oracle_ndcg10=sum(v / np.log2(j + 2) for j, v in enumerate(gains[:10])) / max(denom, 1e-12))
                records.append(dict(query_id=qid, dataset=name, split=split, ideal=float(denom),
                    total_relevant=sum(v > 0 for v in rel.values()), old_candidates=coverage(old), new_candidates=coverage(chosen),
                    metrics={m: metric(o, rel, doc_ids) for m, o in zip(("dense", "hybrid", "bm25"), orders)},
                    ranked_ids={m: [doc_ids[c] for c in o[:10]] for m, o in zip(("dense", "hybrid", "bm25"), orders)}))
            # Flatten batches over many short query lists for higher teacher utilization.
            pairs = [(i, j) for i in range(len(qids)) for j in np.where(mask[i])[0]]
            for offset in range(0, len(pairs), 1024):
                chunk = pairs[offset:offset + 1024]
                logits = teacher_scores([r["queries"][qids[i]] for i, j in chunk],
                    [r["texts"][candidate[i, j]] for i, j in chunk], tokenizer, teacher)
                for (i, j), value in zip(chunk, logits):
                    teacher_values[i, j] = value
                if offset % 32768 == 0:
                    print(name, split, "teacher pairs", offset, "/", len(pairs), flush=True)
            if not np.isfinite(teacher_values).all():
                raise ValueError("teacher produced non-finite score")
            np.savez(target, Q=Q, candidates=candidate, F=features, Y=relevance, mask=mask, teacher=teacher_values)
            save(records_path, dict(records=records, query_ids=qids, document_ids=doc_ids))
            info = dict(dataset=name, split=split, queries=len(qids), pairs=len(pairs), seconds=time.perf_counter() - start,
                        protocol_sha256=metadata["protocol_sha256"], array_sha256=sha(target), metadata_sha256=sha(records_path),
                        document_sha256=sha(r["dpath"]))
            save(marker, info)
            metadata["partitions"].append(info)
            print("Prepared", name, split, round(info["seconds"], 1), "seconds", flush=True)
    metadata["source_sha256"] = {"experiments/prepare_controller_v2.py": sha(Path(__file__))}
    save(OUT / "data-manifest.json", metadata)
    print("Prepared all partitions; no test outcomes selected a model.", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work", required=True, type=Path)
    torch.set_num_threads(4)
    os.environ["HF_HUB_OFFLINE"] = "1"
    main(parser.parse_args().work)

"""Prepare source-disjoint local MuSiQue development inputs and scoring keys.

Copyright (c) 2026 Prashant Jagtap. MIT License.
MuSiQue data stays outside this repository under its original CC BY 4.0 terms.
The filter checks known local overlap, not unknown base-model pretraining.
"""

import argparse
import collections
import hashlib
import json
import re
import unicodedata
from pathlib import Path

from ruler_native import outside_repo, sha, write_new

REVISION = "922ac98f19a201998dbdae6d7f2887a5258dbdeb"
DATA_HASHES = {
    "musique_full_v1.0_train.jsonl": "b1cd998f7e0e2838d6fda024e4ad1eb0e7fc3edefdadb0bd9b5b10b0907f2034",
    "musique_full_v1.0_dev.jsonl": "8cab31d56a3a1c4ef491b205a8dab3f1ac9c66e472098c6cf1de4e20294f7a4a",
}


def normalized(text):
    return " ".join(re.findall(r"\w+", unicodedata.normalize("NFKC", text).casefold()))


def digest(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def rows(path):
    with path.open(encoding="utf-8") as stream:
        for index, line in enumerate(stream):
            if index >= 50000 or len(line.encode()) > 2*1024**2:
                raise ValueError("bounded dataset rows required")
            yield json.loads(line)


def public_input(row):
    """Allowlist inputs; no decomposition, aliases, support labels or truth flag."""
    if not isinstance(row.get("question"), str) or not 1 <= len(row["question"]) <= 16384:
        raise ValueError("bounded question required")
    paragraphs = row["paragraphs"]
    if type(paragraphs) is not list or not 1 <= len(paragraphs) <= 32:
        raise ValueError("bounded paragraph list required")
    result = []
    for paragraph in paragraphs:
        idx, title, text = paragraph["idx"], paragraph["title"], paragraph["paragraph_text"]
        if (type(idx) is not int or not 0 <= idx < 32 or not isinstance(title, str)
                or len(title) > 2048 or not isinstance(text, str) or not 1 <= len(text) <= 65536):
            raise ValueError("invalid source paragraph")
        result.append(dict(idx=idx, title=title, text=text, sha256=digest(title+"\0"+text)))
    if len({p["idx"] for p in result}) != len(result):
        raise ValueError("duplicate source index")
    # Canonical source order prevents fitting to native supporting-index order.
    result.sort(key=lambda p: (p["sha256"], p["idx"]))
    payload = dict(question=row["question"], paragraphs=result)
    payload["key"] = digest(json.dumps(payload, sort_keys=True, ensure_ascii=False))
    return payload


def prior_inventory(root):
    titles, texts, questions = set(), set(), set()
    for item in json.loads((root/"squad.json").read_bytes())["data"]:
        titles.add(normalized(item["title"]))
        for paragraph in item["paragraphs"]:
            texts.add(digest(normalized(paragraph["context"])))
            questions.update(normalized(q["question"]) for q in paragraph["qas"])
    for item in json.loads((root/"hotpotqa.json").read_bytes()):
        questions.add(normalized(item["question"]))
        for title, sentences in item["context"]:
            titles.add(normalized(title))
            texts.add(digest(normalized("".join(sentences))))
    return titles, texts, questions


def inventory(path, prior):
    groups = {}
    for row in rows(path):
        public_input(row)
        if type(row["answerable"]) is not bool:
            raise ValueError("typed target required")
        group = groups.setdefault(row["id"], dict(labels=[], seeds=set(), titles=set(), texts=set(), overlap=False))
        titles = {normalized(p["title"]) for p in row["paragraphs"]}
        texts = {digest(normalized(p["paragraph_text"])) for p in row["paragraphs"]}
        group["labels"].append(row["answerable"])
        group["seeds"].update(str(d["id"]) for d in row["question_decomposition"])
        group["titles"].update(titles)
        group["texts"].update(texts)
        group["overlap"] |= bool(titles & prior[0] or texts & prior[1] or normalized(row["question"]) in prior[2])
    if any(sorted(g["labels"]) != [False, True] for g in groups.values()):
        raise ValueError("every question must have one answerable and one unanswerable variant")
    return groups


def select_groups(groups, *, name, count, excluded=None, separate_each=False):
    excluded = {field: set(values) for field, values in (excluded or dict(titles=(), texts=(), seeds=())).items()}
    selected = []
    for key in sorted(groups, key=lambda k: digest("musique-local-v1:"+name+":"+k)):
        group = groups[key]
        if group["overlap"] or any(group[f] & excluded[f] for f in excluded):
            continue
        selected.append(key)
        if separate_each:
            for f in excluded:
                excluded[f].update(group[f])
        if len(selected) == count:
            break
    if len(selected) != count:
        raise ValueError(f"insufficient source-separated {name} groups: {len(selected)}/{count}")
    return selected


def union_metadata(groups, keys):
    return {f: set().union(*(groups[k][f] for k in keys)) for f in ("titles", "texts", "seeds")}


def prepare(data, prior, source, output):
    data, prior, source, output = map(outside_repo, (data, prior, source, output))
    if any(sha(data/name) != expected for name, expected in DATA_HASHES.items()):
        raise ValueError("pinned MuSiQue bytes changed")
    output.mkdir(parents=True, exist_ok=False)
    previous = prior_inventory(prior)
    meta = {split: inventory(data/f"musique_full_v1.0_{split}.jsonl", previous) for split in ("train", "dev")}
    heldout = select_groups(meta["dev"], name="evaluation", count=64, separate_each=True)
    evaluation_sources = union_metadata(meta["dev"], heldout)
    calibration = select_groups(meta["train"], name="calibration", count=64,
                                excluded=evaluation_sources, separate_each=True)
    calibration_sources = union_metadata(meta["train"], calibration)
    excluded = {f: evaluation_sources[f] | calibration_sources[f] for f in evaluation_sources}
    training = select_groups(meta["train"], name="training", count=400, excluded=excluded)
    partitions = dict(training=training, calibration=calibration, evaluation=heldout)
    audit = dict(native_groups={s: len(g) for s, g in meta.items()},
                 no_prior_overlap_groups={s: sum(not g["overlap"] for g in gs.values()) for s, gs in meta.items()},
                 selected_groups={s: len(ids) for s, ids in partitions.items()}, overlap={})
    groups_by_part = {s: union_metadata(meta["dev" if s == "evaluation" else "train"], ids)
                      for s, ids in partitions.items()}
    for a, b in (("training", "calibration"), ("training", "evaluation"), ("calibration", "evaluation")):
        audit["overlap"][a+":"+b] = {f: len(groups_by_part[a][f] & groups_by_part[b][f]) for f in excluded}
    if any(n for pair in audit["overlap"].values() for n in pair.values()):
        raise ValueError("source or seed overlap survived partitioning")
    write_new(output/"registration.json", dict(
        revision=REVISION, data_hashes=DATA_HASHES, preparation_sha256=sha(__file__),
        prior_hashes={p: sha(prior/p) for p in ("squad.json", "hotpotqa.json")},
        scorer_hashes={p.relative_to(source).as_posix(): sha(p) for p in source.rglob("*.py")},
        partition_groups=partitions, audit=audit,
        boundary="Fresh local source-filtered development subset. No base-pretraining contamination or domain qualification claim.",
        isolation="Selectors/readers receive input files only. Decompositions and target labels stay in separate scoring files."))
    for split in ("train", "dev"):
        available = {name: set(ids) for name, ids in partitions.items() if (name == "evaluation") == (split == "dev")}
        chosen = collections.defaultdict(list)
        for row in rows(data/f"musique_full_v1.0_{split}.jsonl"):
            for name, ids in available.items():
                if row["id"] in ids:
                    clean = public_input(row)
                    chosen[name].append((clean, dict(key=clean["key"], group=row["id"], original=row)))
        for name, pairs in chosen.items():
            pairs.sort(key=lambda pair: digest("case-order:"+pair[0]["key"]))
            with (output/f"{name}-inputs.jsonl").open("x", encoding="utf-8", newline="\n") as inputs, \
                    (output/f"{name}-keys.jsonl").open("x", encoding="utf-8", newline="\n") as keys:
                for public, private in pairs:
                    inputs.write(json.dumps(public, ensure_ascii=False)+"\n")
                    keys.write(json.dumps(private, ensure_ascii=False)+"\n")
    write_new(output/"prepared.json", dict(files={p.name: sha(p) for p in output.glob("*.jsonl")}, audit=audit))
    print(json.dumps(audit), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("data", "prior", "source", "output"):
        parser.add_argument("--"+name, type=Path, required=True)
    args = parser.parse_args()
    prepare(args.data, args.prior, args.source, args.output)

"""Prepare native JSON/PNG ChartQA sources without a Parquet reader.

Copyright (c) 2026 Prashant Jagtap. MIT License for this code.
Publisher data and notices remain external under their own terms. No model calls.
"""

import argparse
import collections
import hashlib
import importlib.metadata
import json
import platform
import re
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from chartqa_prepare import (
    COUNTS,
    MAX_IMAGE_BYTES,
    ROOT,
    SPLITS,
    file_digest,
    fingerprint,
    image_identity,
    normalized_question,
    outside,
    select_groups,
    write_new,
)

REVISION = "044eabfc306abfe9340c5741f0093aefc5973d06"
IMAGE_NAME = r"[^/\\:\x00-\x1f]{1,160}\.png"


def git_identity(raw):
    # Git's upstream object identifier; SHA-256 also binds our retained bytes.
    return hashlib.sha1(f"blob {len(raw)}\0".encode() + raw, usedforsecurity=False).hexdigest()


def native_inventory(source):
    manifest = json.loads((source / "manifest.json").read_bytes())
    if manifest["revision"] != REVISION or file_digest(source / "git-tree.json") != manifest["tree_sha256"]:
        raise ValueError("native source registration changed")
    tree = json.loads((source / "git-tree.json").read_bytes())
    if tree["truncated"] or len({r["path"] for r in tree["tree"]}) != len(tree["tree"]):
        raise ValueError("complete unique Git source tree required")
    index = {r["path"]: r for r in tree["tree"] if r["type"] == "blob"}
    records, labels = [], {}
    for split in SPLITS:
        for origin, name in enumerate(("human", "augmented")):
            filename = f"{split}_{name}.json"
            record = manifest["files"][filename]
            path = source / filename
            raw = path.read_bytes()
            if fingerprint(raw) != record["sha256"] or git_identity(raw) != index[record["source"]]["sha"]:
                raise ValueError("native question source changed")
            rows = json.loads(raw)
            if len(rows) != record["rows"]:
                raise ValueError("native record count changed")
            for i, row in enumerate(rows):
                if set(row) != {"imgname", "query", "label"}:
                    raise ValueError("unexpected native question fields")
                name = row["imgname"]
                if type(name) is not str or not re.fullmatch(IMAGE_NAME, name):
                    raise ValueError("flat source PNG name required")
                image_path = f"ChartQA Dataset/{split}/png/{name}"
                image = index[image_path]
                if not 0 < image["size"] <= MAX_IMAGE_BYTES:
                    raise ValueError("bounded PNG source required")
                normalized_question(row["query"])
                row_id = f"{split}-{'h' if origin == 0 else 'a'}-{i:05d}"
                records.append(dict(id=row_id, split=split, origin=origin, question=row["query"],
                                    source_questions=record["source"], source_image=image_path,
                                    git_blob=image["sha"], declared_bytes=image["size"]))
                # Supplied labels are mechanically separated, never used to select cases.
                labels[row_id] = row["label"]
    if collections.Counter(r["split"] for r in records) != {"train": 28299, "val": 1920, "test": 2500}:
        raise ValueError("native split totals differ")
    return records, labels


def fetch_png(row, image_folder):
    path = row["source_image"]
    if not re.fullmatch(r"ChartQA Dataset/(train|val|test)/png/" + IMAGE_NAME, path):
        raise ValueError("publisher image path required")
    maximum = row["declared_bytes"]
    if type(maximum) is not int or not 0 < maximum <= MAX_IMAGE_BYTES:
        raise ValueError("bounded publisher image required")
    url = f"https://raw.githubusercontent.com/vis-nlp/ChartQA/{REVISION}/" + urllib.parse.quote(path, safe="/")
    with urllib.request.urlopen(url, timeout=30) as response:
        raw = response.read(maximum + 1)
    if len(raw) != maximum or git_identity(raw) != row["git_blob"]:
        raise ValueError("downloaded PNG differs from pinned Git source")
    identity = image_identity(raw)
    filename = identity["image_sha256"] + ".png"
    with (image_folder / filename).open("xb") as stream:
        stream.write(raw)
    return dict(**identity, image_file="images/" + filename, download_url=url)


def prepare(source, output):
    source, output = outside(source), outside(output)
    output.mkdir(exist_ok=False)
    (output / "source").mkdir()
    hashes = {}
    for name in ("chartqa_prepare.py", "chartqa_native_prepare.py"):
        raw = (ROOT / "experiments" / name).read_bytes()
        (output / "source" / name).write_bytes(raw)
        hashes[name] = fingerprint(raw)
    write_new(output / "registration.json", dict(schema=1, revision=REVISION, source_hashes=hashes,
        input_manifest_sha256=file_digest(source / "manifest.json"), counts=COUNTS, seed=101,
        minimum_questions=2, maximum_questions=4, selection_group="publisher Git blob identity",
        decoded_pixel_gate="selected cross-cohort equality aborts preparation; no silent replacement",
        purpose="repeated-chart workflow subset, not full ChartQA leaderboard score",
        python=platform.python_version(), pillow=importlib.metadata.version("Pillow"),
        parquet_reader_used=False, model_calls=0))
    records, labels = native_inventory(source)
    plan = select_groups(records, group_key="git_blob")
    write_new(output / "selection-plan.json", plan)
    write_new(output / "input-inventory.json", records)
    chosen = {row_id for groups in plan["selected"].values() for group in groups for row_id in group["ids"]}
    chosen_rows = [r for r in records if r["id"] in chosen]
    by_blob = {r["git_blob"]: r for r in chosen_rows}
    (output / "images").mkdir()
    # Four bounded public downloads; this does not load or contend for a GPU.
    with ThreadPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(lambda row: fetch_png(row, output / "images"), by_blob.values()))
    images = dict(zip(by_blob, results))
    inputs, keys = {}, {}
    for split in SPLITS:
        inputs[split], keys[split] = [], []
        for row in chosen_rows:
            if row["split"] != split:
                continue
            label = labels[row["id"]]
            if type(label) is not str or not label.strip() or len(label) > 4096:
                raise ValueError("bounded native string reference required")
            inputs[split].append(dict(**row, **images[row["git_blob"]]))
            keys[split].append(dict(id=row["id"], labels=[label]))
    sets = {s: {r["pixel_group"] for r in inputs[s]} for s in SPLITS}
    overlap = {a + "/" + b: len(sets[a] & sets[b]) for a, b in (("train", "val"), ("train", "test"), ("val", "test"))}
    write_new(output / "pixel-overlap-audit.json", overlap)
    if any(overlap.values()):
        raise ValueError("selected cohorts have identical decoded pixels; retain failed plan")
    for split in SPLITS:
        write_new(output / f"{split}.inputs.json", inputs[split])
        write_new(output / f"{split}.keys.json", keys[split])
    write_new(output / "manifest.json", dict(schema=1, revision=REVISION,
        counts={s: dict(images=len(sets[s]), questions=len(inputs[s])) for s in SPLITS},
        cross_split_exact_pixel_overlap=overlap,
        limits=["Near-duplicates and base-model pretraining exposure remain possible.",
                "Image groups with 2-4 questions are selected; this is a workflow subset.",
                "No ground-truth table, chart annotation or answer is supplied to readers.",
                "ChartQA sources retain their GPL-3.0 notice; no dataset redistribution is implied."],
        files={p.relative_to(output).as_posix(): file_digest(p) for p in output.rglob("*") if p.is_file()}))
    print(json.dumps({s: dict(images=len(sets[s]), questions=len(inputs[s])) for s in SPLITS}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    prepare(args.source, args.output)

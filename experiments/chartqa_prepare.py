"""Prepare image-disjoint ChartQA workflow cohorts without answer-based selection.

Copyright (c) 2026 Prashant Jagtap. MIT License for this preparation code.
ChartQA data retain their own GPL-3.0 notice; keep all data outside this repo.
No model is loaded. Reference answers are written separately from reader inputs.
"""

import argparse
import collections
import hashlib
import importlib.metadata
import io
import json
import platform
import re
import unicodedata
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATASET = "HuggingFaceM4/ChartQA"
DATA_REVISION = "b605b6e08b57faf4359aeb2fe6a3ca595f99b6c5"
SPLITS = ("train", "val", "test")
COUNTS = {"train": 64, "val": 32, "test": 64}
MAX_IMAGE_BYTES = 16 * 1024 * 1024
MAX_PIXELS = 16 * 1024 * 1024


def fingerprint(raw):
    return hashlib.sha256(raw).hexdigest()


def file_digest(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def write_new(path, value):
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def outside(path):
    value = Path(path).resolve()
    if value == ROOT or ROOT in value.parents:
        raise ValueError("benchmark data must remain outside the source repository")
    return value


def normalized_question(text):
    if type(text) is not str or not text.strip() or len(text.encode("utf-8")) > 16384:
        raise ValueError("bounded nonempty question required")
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def image_identity(raw):
    from PIL import Image

    if type(raw) is not bytes or not 1 <= len(raw) <= MAX_IMAGE_BYTES:
        raise ValueError("bounded immutable image required")
    with warnings.catch_warnings():
        warnings.simplefilter("error", Image.DecompressionBombWarning)
        with Image.open(io.BytesIO(raw)) as source:
            if source.format not in ("PNG", "JPEG") or getattr(source, "n_frames", 1) != 1:
                raise ValueError("single-frame PNG or JPEG required")
            width, height = source.size
            if width * height > MAX_PIXELS or width <= 0 or height <= 0:
                raise ValueError("decoded image exceeds pixel limit")
            media_type = "image/png" if source.format == "PNG" else "image/jpeg"
            source.verify()
        with Image.open(io.BytesIO(raw)) as source:
            # Decode identity catches different encodings of the same exact pixels.
            # It does not establish absence of near-duplicates or pretraining exposure.
            pixels = source.convert("RGBA").tobytes()
        group = fingerprint(f"RGBA:{width}:{height}:".encode() + pixels)
    return dict(image_sha256=fingerprint(raw), pixel_group=group, width=width, height=height,
                byte_length=len(raw), media_type=media_type)


def select_groups(records, counts=COUNTS, *, seed=101, group_key="pixel_group"):
    """Input-only repeated-chart task selection; no reference answer is consulted."""
    if set(counts) != set(SPLITS) or any(type(v) is not int or not 1 <= v <= 256 for v in counts.values()):
        raise ValueError("bounded count for every native split required")
    if group_key not in ("pixel_group", "git_blob"):
        raise ValueError("declared source grouping required")
    groups = collections.defaultdict(list)
    memberships = collections.defaultdict(set)
    ids = set()
    for row in records:
        if row["split"] not in SPLITS or row["id"] in ids:
            raise ValueError("invalid split or duplicate row identity")
        ids.add(row["id"])
        memberships[row[group_key]].add(row["split"])
        groups[(row["split"], row[group_key])].append(row)
    exclusions, eligible, selected = [], {}, {}
    for split in SPLITS:
        candidates = []
        for (owner, group), members in sorted(groups.items()):
            if owner != split:
                continue
            later = [s for s in memberships[group] if SPLITS.index(s) > SPLITS.index(split)]
            if later:
                exclusions.append(dict(split=split, group=group, reason="image_in_later_native_split", rows=len(members)))
                continue
            unique = {}
            for row in sorted(members, key=lambda r: r["id"]):
                key = normalized_question(row["question"])
                if key in unique:
                    exclusions.append(dict(split=split, id=row["id"], reason="duplicate_question_for_identical_pixels"))
                else:
                    unique[key] = row
            if len(unique) < 2:
                exclusions.append(dict(split=split, group=group, reason="fewer_than_two_distinct_questions", rows=len(unique)))
                continue
            order = sorted(unique.values(), key=lambda r: fingerprint(f"{seed}|question|{r['id']}".encode()))
            candidates.append((fingerprint(f"{seed}|chart|{group}".encode()), group, order[:4]))
        eligible[split] = len(candidates)
        if len(candidates) < counts[split]:
            raise ValueError(f"insufficient repeated-chart groups in {split}: {len(candidates)}")
        selected[split] = [dict(group=group, ids=[r["id"] for r in rows])
                           for _, group, rows in sorted(candidates)[:counts[split]]]
    return dict(seed=seed, group_key=group_key, requested_counts=dict(counts), eligible_groups=eligible,
                selected=selected, exclusions=exclusions)


def shards(data, registration):
    if registration["dataset"] != DATASET or registration["revision"] != DATA_REVISION:
        raise ValueError("unexpected dataset revision")
    if "README.md" not in registration["files"] or len(registration["files"]) != 6:
        raise ValueError("dataset card and all five shards required")
    card = (data / "README.md").read_bytes()
    expected_card = registration["files"]["README.md"]
    git_identity = hashlib.sha1(f"blob {len(card)}\0".encode() + card, usedforsecurity=False).hexdigest()
    if len(card) != expected_card["size"] or git_identity != expected_card["git_blob"]:
        raise ValueError("dataset card differs from Hub revision")
    result = []
    for name, record in sorted(registration["files"].items()):
        if name == "README.md":
            continue
        match = re.fullmatch(r"data/(train|val|test)-\d{5}-of-\d{5}-[a-f0-9]{16}\.parquet", name)
        if not match:
            raise ValueError("unexpected dataset member")
        path = data / name
        if path.stat().st_size != record["size"] or file_digest(path) != record["sha256"]:
            raise ValueError("downloaded shard differs from Hub identity: " + name)
        result.append((match[1], name, path))
    if collections.Counter(split for split, _, _ in result) != {"train": 3, "val": 1, "test": 1}:
        raise ValueError("complete five-shard dataset required")
    return result


def rows_from(shard_list, columns):
    import pyarrow.parquet as pq

    counts = collections.Counter()
    for split, filename, path in shard_list:
        parquet = pq.ParquetFile(path)
        for batch in parquet.iter_batches(batch_size=32, columns=columns, use_threads=False):
            for row in batch.to_pylist():
                row_id = f"{split}-{counts[split]:05d}"
                yield split, row_id, filename, row
                counts[split] += 1


def prepare(data, download_registration, output):
    data, download_registration, output = map(outside, (data, download_registration, output))
    registration = json.loads(download_registration.read_bytes())
    output.mkdir(exist_ok=False)
    (output / "source").mkdir()
    source = Path(__file__).read_bytes()
    (output / "source/chartqa_prepare.py").write_bytes(source)
    write_new(output / "registration.json", dict(dataset=DATASET, revision=DATA_REVISION,
        source_sha256=fingerprint(source), download_registration_sha256=file_digest(download_registration),
        counts=COUNTS, minimum_questions=2, maximum_questions=4, seed=101,
        purpose="repeated-chart local context workflow; image-group subset, not full benchmark",
        packages={n: importlib.metadata.version(n) for n in ("Pillow", "pyarrow")}, python=platform.python_version(),
        label_boundary="input-only selection first; supplied reference labels copied separately in second pass"))
    (output / "download-registration.json").write_bytes(download_registration.read_bytes())
    shard_list = shards(data, registration)
    cache, records = {}, []
    for split, row_id, filename, row in rows_from(shard_list, ["image", "query", "human_or_machine"]):
        raw = row["image"]["bytes"]
        encoded = fingerprint(raw)
        if encoded not in cache:
            cache[encoded] = image_identity(raw)
        normalized_question(row["query"])
        origin = row["human_or_machine"]
        if type(origin) is not int or origin not in (0, 1):
            raise ValueError("unexpected question origin")
        records.append(dict(id=row_id, split=split, shard=filename, question=row["query"],
                            origin=origin, **cache[encoded]))
    if collections.Counter(r["split"] for r in records) != {"train": 28299, "val": 1920, "test": 2500}:
        raise ValueError("native row counts differ")
    plan = select_groups(records)
    write_new(output / "selection-plan.json", plan)
    write_new(output / "input-inventory.json", records)
    chosen = {row_id for groups in plan["selected"].values() for group in groups for row_id in group["ids"]}
    lookup = {r["id"]: r for r in records if r["id"] in chosen}
    (output / "images").mkdir()
    inputs, keys = collections.defaultdict(list), collections.defaultdict(list)
    for split, row_id, _, row in rows_from(shard_list, ["image", "label"]):
        if row_id not in chosen:
            continue
        record, labels = lookup[row_id], row["label"]
        if type(labels) is not list or not labels or any(type(t) is not str or not t.strip() or len(t) > 4096 for t in labels):
            raise ValueError("invalid benchmark reference label")
        raw = row["image"]["bytes"]
        if fingerprint(raw) != record["image_sha256"]:
            raise ValueError("source changed between preparation passes")
        asset_path = "images/" + record["image_sha256"] + ".bin"
        target = output / asset_path
        if not target.exists():
            target.write_bytes(raw)
        inputs[split].append(dict(**record, image_file=asset_path))
        keys[split].append(dict(id=row_id, labels=labels))
    if sum(map(len, inputs.values())) != len(chosen):
        raise ValueError("missing selected inputs")
    for split in SPLITS:
        write_new(output / f"{split}.inputs.json", inputs[split])
        write_new(output / f"{split}.keys.json", keys[split])
    sets = {s: {r["pixel_group"] for r in inputs[s]} for s in SPLITS}
    overlap = {a + "/" + b: len(sets[a] & sets[b]) for a, b in (("train", "val"), ("train", "test"), ("val", "test"))}
    if any(overlap.values()):
        raise ValueError("selected cohorts share exact decoded image pixels")
    write_new(output / "manifest.json", dict(schema=1, dataset=DATASET, data_revision=DATA_REVISION,
        counts={s: dict(images=len(plan["selected"][s]), questions=len(inputs[s])) for s in SPLITS},
        cross_split_exact_pixel_overlap=overlap,
        limits=["Near-duplicate charts and base-model pretraining exposure are not ruled out.",
                "Repeated-chart subset requires 2-4 distinct questions per image; not a full native benchmark.",
                "Source data retain their GPL-3.0 notice. No reference table or answer enters reader inputs."],
        files={p.relative_to(output).as_posix(): file_digest(p) for p in output.rglob("*") if p.is_file()}))
    print(json.dumps({s: dict(images=len(plan["selected"][s]), questions=len(inputs[s])) for s in SPLITS}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--download-registration", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    prepare(args.data, args.download_registration, args.output)

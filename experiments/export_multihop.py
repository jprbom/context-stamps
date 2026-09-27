"""Export reviewed short predictions, aggregate results and reproducible policies.

Raw questions, paragraphs and prompts stay outside the publishing repository.
"""

import base64
import gzip
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

from multihop_data import digest, normalized
from relation_ranker import features
from ruler_native import sha, write_new

ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT.parent


def read(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def compressed(path, value):
    with path.open("xb") as stream:
        stream.write(gzip.compress(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode(), mtime=0))


def export():
    output = ROOT/"evidence/multihop-v1"
    output.mkdir(exist_ok=False)
    data = WORK/"multihop-prepared-v2"
    training = WORK/"relation-ranker-training-v1"
    shutil.copyfile(data/"registration.json", output/"data-registration.json")
    shutil.copyfile(data/"prepared.json", output/"data-prepared.json")
    shutil.copyfile(WORK/"musique-data-v1/download.json", output/"download.json")
    shutil.copyfile(training/"registration.json", output/"training-registration.json")
    for name in ("training.json", "pointwise.json", "diffusion.json"):
        shutil.copyfile(training/name, output/name)
    shutil.copyfile(WORK/"multihop-native-canaries-v1/canaries.json", output/"native-canaries.json")
    keys = read(data/"evaluation-keys.jsonl")
    write_new(output/"targets.json", [dict(key=k["key"], group=k["group"], answer=k["original"]["answer"],
                                         aliases=k["original"]["answer_aliases"], answerable=k["original"]["answerable"],
                                         support=[p["idx"] for p in k["original"]["paragraphs"] if p["is_supporting"]]) for k in keys])
    feature_rows = {}
    for case in read(data/"evaluation-inputs.jsonl"):
        x, p, b = features(case)
        feature_rows[case["key"]] = dict(x=x, transition=p, bm25=b, indices=[p["idx"] for p in case["paragraphs"]],
                                       hashes=[p["sha256"] for p in case["paragraphs"]])
    compressed(output/"evaluation-features.json.gz", feature_rows)
    inventory = {}
    for part in ("training", "calibration", "evaluation"):
        groups = {}
        for key in read(data/f"{part}-keys.jsonl"):
            g = groups.setdefault(key["group"], dict(titles=set(), texts=set(), seeds=set()))
            for p in key["original"]["paragraphs"]:
                g["titles"].add(digest(normalized(p["title"])))
                g["texts"].add(digest(normalized(p["paragraph_text"])))
            g["seeds"].update(str(d["id"]) for d in key["original"]["question_decomposition"])
        inventory[part] = {k: {f: sorted(v) for f, v in g.items()} for k, g in groups.items()}
    compressed(output/"partition-inventory.json.gz", inventory)
    maps = dict(json.loads((training/"registration.json").read_bytes())["source_hashes"])
    for name, run_name, analysis_name in (("small", "multihop-reader-v2", "multihop-analysis-v1"),
                                          ("reference", "multihop-reference-v1", "multihop-reference-analysis-v1")):
        target = output/name
        target.mkdir()
        run, analysis = WORK/run_name, WORK/analysis_name
        for filename in ("registration.json", "prepared.json", "complete.json", "warmup.json"):
            shutil.copyfile(run/filename, target/filename)
        if (run/"reference-protocol.json").exists():
            shutil.copyfile(run/"reference-protocol.json", target/"reference-protocol.json")
        maps.update(json.loads((run/"registration.json").read_bytes())["source_hashes"])
        compressed(target/"generations.json.gz", read(run/"generations.jsonl"))
        for filename in ("summary.json", "case-scores.json", "group-scores.json"):
            shutil.copyfile(analysis/filename, target/filename)
        shutil.copyfile(analysis/"registration.json", target/"scoring-registration.json")
        maps["experiments/eval_multihop.py"] = json.loads((analysis/"registration.json").read_bytes())["scorer_sha256"]
    # Preserve both preparation variants before any potential source changes.
    first = WORK/"multihop-prepared-v1"
    first_failure = json.loads((first/"failure.json").read_bytes())
    prior_source = (first/"multihop_data.py").read_bytes()
    if hashlib.sha256(prior_source).hexdigest() != first_failure["source_sha256"]:
        raise ValueError("first failed preparation source changed")
    earlier = WORK/"multihop-reader-v1"
    contents = json.loads(gzip.decompress((earlier/"sources.json.gz").read_bytes()))["content_by_sha256"]
    contents[first_failure["source_sha256"]] = base64.b64encode(prior_source).decode()
    archive = ROOT/"evidence/engineering-sources-v1/source-multihop-preparation-v1.json.gz"
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    compressed(archive, dict(schema=1, source_base_commit=commit, worktree_dirty=True,
                             content_by_sha256=contents, manifest_source_maps={}))
    index = archive.parent/"index.json"
    listing = json.loads(index.read_bytes())
    listing["archives"].append(dict(file=archive.name, sha256=sha(archive), source_base_commit=commit, worktree_dirty=True))
    index.write_text(json.dumps(listing, indent=2)+"\n", encoding="utf-8", newline="\n")
    write_new(output/"preparation-history.json", dict(first_failure=first_failure,
                                                     superseded_reader=json.loads((earlier/"superseded.json").read_bytes()),
                                                     archive=archive.relative_to(ROOT).as_posix(), archive_sha256=sha(archive)))
    for name in ("export_multihop.py", "verify_multihop.py", "test_relation_ranker.py"):
        maps["experiments/"+name] = sha(ROOT/"experiments"/name)
    write_new(output/"manifest.json", dict(source_hashes=maps, model_requests=1024, warmups=2,
                                           candidate_active=False,
                                           files={p.relative_to(output).as_posix(): sha(p) for p in output.rglob("*") if p.is_file()}))


if __name__ == "__main__":
    export()

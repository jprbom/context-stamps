"""Fetch pinned publisher JSON metadata; image selection happens separately.

Copyright (c) 2026 Prashant Jagtap. MIT License for this code.
The downloaded benchmark retains its original licence and attribution.
"""

import argparse
import json
import urllib.parse
import urllib.request
from pathlib import Path

from chartqa_native_prepare import REVISION, git_identity
from chartqa_prepare import fingerprint, outside, write_new


def read_url(url, limit):
    request = urllib.request.Request(url, headers={"User-Agent": "context-stamps-research"})
    with urllib.request.urlopen(request, timeout=30) as response:
        raw = response.read(limit + 1)
    if len(raw) > limit:
        raise ValueError("source response exceeds size limit")
    return raw


def fetch(output):
    output = outside(output)
    output.mkdir(exist_ok=False)
    source = Path(__file__).read_bytes()
    (output / "fetch-source.py.txt").write_bytes(source)
    write_new(output / "registration.json", dict(revision=REVISION, source_sha256=fingerprint(source),
                                                dataset="vis-nlp/ChartQA native JSON and PNG", model_calls=0))
    raw = read_url(f"https://api.github.com/repos/vis-nlp/ChartQA/git/trees/{REVISION}?recursive=1", 64*1024*1024)
    tree = json.loads(raw)
    if tree["truncated"]:
        raise ValueError("complete Git tree required")
    (output / "git-tree.json").write_bytes(raw)
    index = {r["path"]: r for r in tree["tree"] if r["type"] == "blob"}
    if len(index) != sum(r["type"] == "blob" for r in tree["tree"]):
        raise ValueError("duplicate source tree member")
    records = {}
    names = [(f"ChartQA Dataset/{split}/{split}_{origin}.json", f"{split}_{origin}.json")
             for split in ("train", "val", "test") for origin in ("human", "augmented")]
    names += [("README.md", "README.md"), ("LICENSE", "LICENSE")]
    for name, target in names:
        item = index[name]
        if type(item["size"]) is not int or not 0 < item["size"] < 16*1024*1024:
            raise ValueError("source file size limit")
        url = f"https://raw.githubusercontent.com/vis-nlp/ChartQA/{REVISION}/" + urllib.parse.quote(name, safe="/")
        data = read_url(url, item["size"])
        if len(data) != item["size"] or git_identity(data) != item["sha"]:
            raise ValueError("pinned source object differs")
        (output / target).write_bytes(data)
        record = dict(source=name, url=url, sha256=fingerprint(data))
        if target.endswith(".json"):
            content = json.loads(data)
            record.update(rows=len(content), fields=sorted(content[0]))
        records[target] = record
    write_new(output / "manifest.json", dict(revision=REVISION, tree_sha256=fingerprint(raw), files=records))
    print(json.dumps(dict(files=len(records), questions=sum(r.get("rows", 0) for r in records.values()),
                          revision=REVISION, model_calls=0)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    fetch(parser.parse_args().output)

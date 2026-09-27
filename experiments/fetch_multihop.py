"""Download pinned public MuSiQue assets without executing upstream installers."""

import argparse
import hashlib
import json
import urllib.request
import zipfile
from pathlib import Path

from multihop_data import DATA_HASHES, REVISION
from ruler_native import outside_repo, write_new

ROOT = Path(__file__).resolve().parents[1]
URL = "https://drive.usercontent.google.com/download?id=1tGdADlNjWFaHLeZZGShh2IRcpO6Lv24h&export=download&confirm=t"
ZIP_HASH = "98f839bf2fd5319f5c688aed77901a6d5c30b3b9f9f691ab9a8ecafb045ee0cd"


def fetch_file(url, path, limit, expected=None):
    checksum, size = hashlib.sha256(), 0
    with urllib.request.urlopen(url, timeout=45) as response, path.open("xb") as stream:
        if not response.geturl().startswith("https://"):
            raise ValueError("HTTPS downgrade refused")
        while True:
            chunk = response.read(1024*1024)
            if not chunk:
                break
            size += len(chunk)
            if size > limit:
                raise ValueError("download size limit; partial file retained")
            checksum.update(chunk)
            stream.write(chunk)
    if expected is not None and checksum.hexdigest() != expected:
        raise ValueError("pinned download changed; file retained, do not use")
    return dict(bytes=size, sha256=checksum.hexdigest(), url=url)


def fetch(source, data):
    source, data = map(outside_repo, (source, data))
    source.mkdir(parents=True, exist_ok=False)
    data.mkdir(parents=True, exist_ok=False)
    (source/"metrics").mkdir()
    expected = json.loads((ROOT/"evidence/multihop-v1/data-registration.json").read_bytes())["scorer_hashes"]
    names = {"evaluate_v1.0.py", *("metrics/"+n+".py" for n in
              ("answer", "support", "group", "metric", "group_answer_sufficiency", "group_support_sufficiency"))}
    if set(expected) != names:
        raise ValueError("unexpected scorer inventory")
    records = {}
    for name in sorted(names | {"README.md", "LICENSE", "download_data.sh"}):
        url = f"https://raw.githubusercontent.com/StonyBrookNLP/musique/{REVISION}/{name}"
        records[name] = fetch_file(url, source/name, 2*1024**2, expected.get(name))
    write_new(source/"download.json", dict(revision=REVISION, files=records))
    archive = data/"musique_data_v1.0.zip"
    record = fetch_file(URL, archive, 300*1024**2, ZIP_HASH)
    write_new(data/"download.json", record)
    hashes = dict(DATA_HASHES)
    hashes["dev_test_singlehop_questions_v1.0.json"] = "013be00ab914799891e5ae40de6cdc1baf03aaffd1f3ce4bf0de796020069613"
    with zipfile.ZipFile(archive) as zipped:
        for name, expected_hash in hashes.items():
            member = zipped.getinfo("data/"+name)
            if member.file_size > 512*1024**2:
                raise ValueError("extracted member size limit")
            checksum = hashlib.sha256()
            with zipped.open(member) as raw, (data/name).open("xb") as output:
                while True:
                    chunk = raw.read(1024*1024)
                    if not chunk:
                        break
                    checksum.update(chunk)
                    output.write(chunk)
            if checksum.hexdigest() != expected_hash:
                raise ValueError("extracted file integrity failed")
    print(json.dumps(dict(source=str(source), data=str(data), native_scripts_executed=False)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    args = parser.parse_args()
    fetch(args.source, args.data)

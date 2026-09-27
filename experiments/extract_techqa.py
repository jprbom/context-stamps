"""Verify and selectively extract a retained pinned archive after download.

Copyright (c) 2026 Prashant Jagtap. MIT License.
The original 128 MiB streaming attempt is retained as a failed preparation.
This attempt allows at most 512 MiB per selected plaintext JSON document store.
"""

import argparse
import hashlib
import json
import tarfile
import time
import urllib.request
from pathlib import Path

from fetch_techqa import DIGEST, MEMBERS, REVISION, SIZE, SOURCE_REVISION
from ruler_native import outside_repo, sha, write_new


def extract(archive_path, output):
    archive_path, output = outside_repo(archive_path), outside_repo(output)
    output.mkdir(parents=True, exist_ok=False)
    write_new(output/"registration.json", dict(revision=REVISION, bytes=SIZE, sha256=DIGEST,
                                               archive_name=archive_path.name, members=MEMBERS,
                                               selected_member_limit=512*1024**2,
                                               source_revision=SOURCE_REVISION, source_sha256=sha(__file__),
                                               constants_source_sha256=sha(Path(__file__).with_name("fetch_techqa.py"))))
    tick = time.perf_counter()
    if archive_path.stat().st_size != SIZE or sha(archive_path) != DIGEST:
        raise ValueError("complete pinned archive required before extraction")
    inventory, selected = [], {}
    with tarfile.open(archive_path, mode="r|gz") as archive:
        for member in archive:
            if len(inventory) >= 1000 or member.size > 16*1024**3:
                raise ValueError("archive inventory bound exceeded")
            metadata = dict(name=member.name, bytes=member.size, regular=member.isfile())
            inventory.append(metadata)
            if member.name not in MEMBERS:
                continue
            print(json.dumps(metadata), flush=True)
            if member.name in selected or not member.isfile() or not 1 <= member.size <= 512*1024**2:
                write_new(output/"failed-inventory.json", inventory)
                raise ValueError("selected member outside fixed bounds")
            name, copied, checksum = MEMBERS[member.name], 0, hashlib.sha256()
            with archive.extractfile(member) as source, (output/name).open("xb") as target:
                while raw := source.read(1024*1024):
                    copied += len(raw)
                    if copied > member.size:
                        raise ValueError("archive member size exceeded")
                    target.write(raw)
                    checksum.update(raw)
            if copied != member.size:
                raise ValueError("incomplete selected member")
            selected[member.name] = dict(file=name, bytes=copied, sha256=checksum.hexdigest())
    if set(selected) != set(MEMBERS):
        raise ValueError("missing selected members")
    for name in ("README.md", "LICENSE.md", "techqa_evaluation.py", "techqa_metrics.py", "techqa_processor.py"):
        url = f"https://raw.githubusercontent.com/IBM/techqa/{SOURCE_REVISION}/{name}"
        with urllib.request.urlopen(url, timeout=45) as response:
            raw = response.read(2*1024**2+1)
        if len(raw) > 2*1024**2:
            raise ValueError("bounded upstream source required")
        with (output/name).open("xb") as target:
            target.write(raw)
    write_new(output/"complete.json", dict(compressed_bytes=SIZE, archive_sha256=DIGEST,
                                           seconds=time.perf_counter()-tick, selected=selected, inventory=inventory,
                                           files={p.name: sha(p) for p in output.iterdir() if p.is_file()},
                                           full_corpus_retained=False, compressed_archive_retained=True,
                                           data_license="CDLA-Permissive-1.0", upstream_code_executed=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    extract(args.archive, args.output)

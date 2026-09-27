"""Stream pinned TechQA data, retaining only the small train/dev collections.

Copyright (c) 2026 Prashant Jagtap. MIT License.
The dataset itself is CDLA-Permissive-1.0, as stated in the archive README;
the Hub's Apache metadata does not replace that dataset notice.
No archive paths are extracted automatically and no upstream code is run.
"""

import argparse
import hashlib
import io
import json
import tarfile
import time
import urllib.request
from pathlib import Path

from ruler_native import outside_repo, sha, write_new

REVISION = "60437bc79ab217679682217598a3693cab78365b"
SOURCE_REVISION = "f0cf8ce11c6ef778c6bc064ee6c1d9b3eca76faf"
URL = f"https://huggingface.co/datasets/PrimeQA/TechQA/resolve/{REVISION}/TechQA.tar.gz"
SIZE = 2959973525
DIGEST = "6b094ef9a69718f727ce8d7e15c4d961e51032cefaa952e0d6af9d176d7ba118"
MEMBERS = {
    "TechQA/README.txt": "README.txt",
    "TechQA/CDLA-Permissive-v1.0.pdf": "CDLA-Permissive-v1.0.pdf",
    "TechQA/training_and_dev/training_Q_A.json": "training_Q_A.json",
    "TechQA/training_and_dev/dev_Q_A.json": "dev_Q_A.json",
    "TechQA/training_and_dev/training_dev_technotes.json": "training_dev_technotes.json",
}


class CheckedStream(io.RawIOBase):
    def __init__(self, source):
        self.source = source
        self.size = 0
        self.digest = hashlib.sha256()
        self.reported = 0

    def read(self, count=-1):
        raw = self.source.read(min(count if count >= 0 else 1024*1024, SIZE+1-self.size))
        self.size += len(raw)
        if self.size > SIZE:
            raise ValueError("pinned compressed size exceeded")
        self.digest.update(raw)
        if self.size-self.reported >= 256*1024**2:
            self.reported = self.size
            print(json.dumps(dict(compressed_bytes=self.size, expected=SIZE)), flush=True)
        return raw


def fetch(output):
    output = outside_repo(output)
    output.mkdir(parents=True, exist_ok=False)
    write_new(output/"registration.json", dict(url=URL, revision=REVISION, bytes=SIZE, sha256=DIGEST,
                                               source_revision=SOURCE_REVISION, script_sha256=sha(__file__),
                                               license="CDLA-Permissive-1.0", members=MEMBERS))
    tick = time.perf_counter()
    selected, inventory = {}, []
    with urllib.request.urlopen(URL, timeout=120) as response:
        if not response.geturl().startswith("https://"):
            raise ValueError("HTTPS downgrade refused")
        stream = CheckedStream(response)
        with tarfile.open(fileobj=stream, mode="r|gz") as archive:
            for member in archive:
                if len(inventory) >= 1000 or member.size > 16*1024**3:
                    raise ValueError("archive inventory limit")
                inventory.append(dict(name=member.name, bytes=member.size, regular=member.isfile()))
                if member.name not in MEMBERS:
                    continue
                if member.name in selected or not member.isfile() or not 1 <= member.size <= 128*1024**2:
                    raise ValueError("unexpected selected archive member")
                raw = archive.extractfile(member).read(128*1024**2+1)
                if len(raw) != member.size:
                    raise ValueError("incomplete archive member")
                name = MEMBERS[member.name]
                with (output/name).open("xb") as file:
                    file.write(raw)
                selected[member.name] = dict(file=name, bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())
                print(json.dumps(dict(selected_member=member.name, bytes=len(raw))), flush=True)
        while stream.read(1024*1024):
            pass
    if stream.size != SIZE or stream.digest.hexdigest() != DIGEST or set(selected) != set(MEMBERS):
        raise ValueError("pinned complete archive/member integrity failed; retain attempt")
    for name in ("README.md", "LICENSE.md", "techqa_evaluation.py", "techqa_metrics.py", "techqa_processor.py"):
        url = f"https://raw.githubusercontent.com/IBM/techqa/{SOURCE_REVISION}/{name}"
        with urllib.request.urlopen(url, timeout=45) as response:
            raw = response.read(2*1024**2+1)
        if len(raw) > 2*1024**2:
            raise ValueError("upstream source size limit")
        with (output/name).open("xb") as file:
            file.write(raw)
    write_new(output/"complete.json", dict(compressed_bytes=stream.size, archive_sha256=stream.digest.hexdigest(),
                                           seconds=time.perf_counter()-tick, selected=selected, inventory=inventory,
                                           files={p.name: sha(p) for p in output.iterdir() if p.is_file()},
                                           upstream_code_executed=False, full_corpus_retained=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    fetch(parser.parse_args().output)

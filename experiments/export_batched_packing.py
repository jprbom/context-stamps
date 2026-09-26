"""Export CPU measurements and retain all three exploratory failures as data."""

import argparse
import base64
import gzip
import json
import shutil
import subprocess
from pathlib import Path

from ruler_native import sha, write_new
from source_evidence import checked_path

ROOT = Path(__file__).resolve().parents[1]


def export(run, probes, output):
    output = output.resolve()
    if not output.is_relative_to(ROOT/"evidence"):
        raise ValueError("public evidence directory required")
    output.mkdir(parents=True, exist_ok=False)
    names = ("registration.json", "initialization.json", "tokenizer-canaries.json", "records.jsonl", "summary.json", "completion.json")
    for name in names:
        shutil.copyfile(run/name, output/name)
    (output/"probes").mkdir()
    contents, maps = {}, {}
    for i in (1, 2, 3):
        source = probes/f"relation-optimization-attempt-v{i}"
        report = json.loads((source/"probe.json").read_bytes())
        target = output/"probes"/f"attempt-{i}.json"
        write_new(target, report)
        maps[target.relative_to(ROOT).as_posix()] = report["source_hashes"]
        for name, expected in report["source_hashes"].items():
            path = checked_path(Path(name).name, source)
            if sha(path) != expected:
                raise ValueError("previous exploratory source changed")
            contents[expected] = base64.b64encode(path.read_bytes()).decode("ascii")
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    payload = dict(schema=1, source_base_commit=commit, worktree_dirty=True,
                   manifest_source_maps=maps, content_by_sha256=contents)
    archive = ROOT/"evidence/engineering-sources-v1/source-batched-packing-probes-v1.json.gz"
    with archive.open("xb") as stream:
        stream.write(gzip.compress(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode(), mtime=0))
    index_path = archive.parent/"index.json"
    index = json.loads(index_path.read_bytes())
    index["archives"].append(dict(file=archive.name, sha256=sha(archive), source_base_commit=commit, worktree_dirty=True))
    with index_path.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(index, stream, indent=2)
        stream.write("\n")
    write_new(output/"manifest.json", dict(
        files={p.relative_to(output).as_posix(): sha(p) for p in output.rglob("*") if p.is_file()},
        tools={"experiments/"+n: sha(ROOT/"experiments"/n) for n in
               ("export_batched_packing.py", "verify_batched_packing.py")},
        archive=archive.relative_to(ROOT).as_posix(), archive_sha256=sha(archive)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("run", "probes", "output"):
        parser.add_argument("--"+name, type=Path, required=True)
    args = parser.parse_args()
    export(args.run, args.probes, args.output)

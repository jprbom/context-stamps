"""Preserve exact currently available source bytes named by engineering manifests."""

import argparse
import base64
import gzip
import json
import re
import subprocess

from source_evidence import ARCHIVES, ROOT, checked_path, sha


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True)
    parser.add_argument("--manifests", nargs="+", required=True)
    args = parser.parse_args()
    if re.fullmatch(r"[a-z0-9][a-z0-9-]{0,63}", args.name) is None:
        raise ValueError("bounded archive name required")
    target = ARCHIVES / (args.name + ".json.gz")
    if target.exists():
        raise ValueError("source archives cannot be overwritten")
    contents, maps = {}, {}
    for manifest in args.manifests:
        source = json.loads(checked_path(manifest).read_text(encoding="utf-8"))
        hashes = source["source_hashes"]
        maps[manifest] = hashes
        for name, expected in hashes.items():
            raw = checked_path(name).read_bytes()
            if sha(raw) != expected:
                raise ValueError("current source does not match the recorded run: " + name)
            contents[expected] = base64.b64encode(raw).decode("ascii")
    payload = {"schema": 1, "source_base_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
               "worktree_dirty": bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip()),
               "manifest_source_maps": maps, "content_by_sha256": contents}
    compressed = gzip.compress(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode(), mtime=0)
    ARCHIVES.mkdir(parents=True, exist_ok=True)
    index_path = ARCHIVES / "index.json"
    index = json.loads(index_path.read_text()) if index_path.exists() else {"schema": 1, "archives": []}
    target.write_bytes(compressed)
    index["archives"].append({"file": target.name, "sha256": sha(compressed),
                             "source_base_commit": payload["source_base_commit"], "worktree_dirty": payload["worktree_dirty"]})
    with index_path.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(index, stream, indent=2)
        stream.write("\n")
    print(json.dumps({"archive": target.relative_to(ROOT).as_posix(), "distinct_source_contents": len(contents),
                      "compressed_bytes": len(compressed)}))


if __name__ == "__main__":
    main()

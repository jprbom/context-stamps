"""Prepare private multi-format packets for one frozen Terminal-Bench task.

Public input rows and task source remain outside Git. No verifier test or
reference solution is read by this script.
Copyright (c) 2026 Prashant Jagtap. MIT License.
"""

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from context_stamps import CodingSchemaIndex, compile_coding_contract, structured_256_codec  # noqa: E402
from stamps import HashingEncoder  # noqa: E402

REVISION = "7131e4375048a0e408a8fb404b5f499d726b695b"
TASK = "multi-source-data-merger"
SOURCES = ("source_a/users.json", "source_b/users.csv", "source_c/users.parquet")


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def build(source: Path, out: Path):
    source, out = source.resolve(), out.resolve()
    if out.is_relative_to(Path(__file__).resolve().parents[1]):
        raise ValueError("benchmark materials must remain outside Git")
    revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=source,
                              check=True, capture_output=True, text=True).stdout.strip()
    if revision != REVISION:
        raise ValueError("benchmark revision changed")
    task = source / "tasks" / TASK
    root = task / "environment" / "data"
    instruction = (task / "instruction.md").read_text(encoding="utf-8")
    contract = compile_coding_contract(root, SOURCES, ("merged_users.parquet", "conflicts.json"),
                                       runtime_root="/data", output_root="/app")
    if not contract.verify(root):
        raise ValueError("source revision mismatch")
    started = time.perf_counter()
    schema = contract.data_schema_hints(root, allow_parquet=True)
    encoder = HashingEncoder(64)
    codec = structured_256_codec(encoder, seed=20261002)
    index = CodingSchemaIndex(contract, allow_parquet=True)
    stamps = []
    manifest = {}
    for item in contract.sources:
        view = contract.data_schema_hints(root, selected=(item.relative,), allow_parquet=True)
        facets = {"semantic": view, "task": instruction, "entity": item.relative,
                  "relation": "input to merged dataset", "temporal": REVISION,
                  "authority": "pinned public task", "policy": "offline role-scoped coding",
                  "modality": Path(item.relative).suffix}
        stamp = codec.pack(codec.encode({name: encoder.encode(text) for name, text in facets.items()}))
        index.bind(stamp, item.relative, roles=frozenset({"coding_agent"}))
        stamps.append(stamp)
        manifest[item.relative] = {"sha256": item.sha256, "bytes": item.bytes, "stamp_hex": stamp.hex()}
    activated = index.activate_data(tuple(stamps), role="coding_agent", root=root)
    if activated.status != "complete" or activated.text != schema:
        raise ValueError("stamp-bound schema view differs from direct schema")
    build_seconds = time.perf_counter() - started
    full = []
    for item in contract.sources:
        path = root / item.relative
        if path.suffix == ".parquet":
            body = json.dumps(pq.read_table(path).to_pylist(), ensure_ascii=False)
        else:
            body = path.read_text(encoding="utf-8")
        full.append(f"{item.runtime}\n{body}")
    packets = {"task": TASK, "revision": REVISION,
               "instruction_sha256": sha((task / "instruction.md").read_bytes()),
               "contract": contract.render(), "contract_sha256": sha(contract.render().encode()),
               "manifest": manifest, "codec_schema": codec.schema.identity,
               "build_seconds": build_seconds, "stamp_payload_bytes": sum(map(len, stamps)),
               "views": {"contract_only": "No source body supplied by the host.",
                         "schema_direct": schema, "stamp_schema": activated.text,
                         "full_source": "\n\n".join(full)}}
    out.mkdir(parents=True, exist_ok=False)
    (out / "packets.json").write_text(json.dumps(packets, indent=2, ensure_ascii=False) + "\n",
                                      encoding="utf-8")
    print(json.dumps({"source_count": len(SOURCES), "stamp_payload_bytes": packets["stamp_payload_bytes"],
                      "view_bytes": {key: len(value.encode()) for key, value in packets["views"].items()},
                      "build_seconds": build_seconds}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    build(args.source, args.out)

"""Exact-file and dependency-closure packets for a local coding-agent pilot.

The source inventory is the benchmark's public environment, not its hidden
tests or reference solution. Relationships come from the visible task contract.
Copyright (c) 2026 Prashant Jagtap. MIT License.
"""

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from context_stamps.stamp256 import structured_256_codec  # noqa: E402
from context_stamps.workflow import ContextGraph, ContextNode  # noqa: E402
from stamps import HashingEncoder  # noqa: E402

FILES = ("analyze_climate.py", "config.ini", "sample_data/climate_data.csv")
SOURCE = "analyze_climate.py"


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def run(source, instruction, out):
    root = source / "tasks" / "modernize-scientific-stack" / "environment" / "climate_analyzer"
    task = source / "tasks" / "modernize-scientific-stack" / "instruction.md"
    if instruction.resolve() != task.resolve():
        raise ValueError("instruction path differs from selected task")
    prompt = task.read_text(encoding="utf-8")
    if "/app/climate_analyzer/analyze_climate.py" not in prompt:
        raise ValueError("source file is not explicitly requested")
    for relative in FILES:
        if not (root / relative).is_file() or (root / relative).is_symlink():
            raise ValueError("required source missing or symlinked")
    encoder = HashingEncoder(64)
    codec = structured_256_codec(encoder, seed=20261002)
    graph = ContextGraph()
    manifest = {}
    started = time.perf_counter()
    for relative in FILES:
        raw = (root / relative).read_bytes()
        if len(raw) > 20000:
            raise ValueError("unexpected source size")
        body = raw.decode("utf-8")
        digest = sha(raw)
        views = {"semantic": body, "task": prompt if relative == SOURCE else relative,
                 "entity": relative, "relation": "dependency of " + SOURCE,
                 "temporal": "initial revision", "authority": "pinned public repository",
                 "policy": "offline evaluation", "modality": Path(relative).suffix or "text"}
        stamp = codec.encode({name: encoder.encode(value) for name, value in views.items()})
        packed = codec.pack(stamp)
        if len(packed) != 32 or codec.unpack(packed, schema_id=codec.schema.identity) != stamp:
            raise ValueError("256-bit roundtrip failed")
        graph.put(ContextNode(relative, f"/app/climate_analyzer/{relative}\n{body}", digest,
                              frozenset({"coding_agent"}), stamp))
        manifest[relative] = {"sha256": digest, "bytes": len(raw), "stamp_hex": packed.hex()}
    for dependency in FILES[1:]:
        graph.link(SOURCE, dependency, "depends_on", provenance="public_task_contract")
    revisions = {name: value["sha256"] for name, value in manifest.items()}
    handoff = graph.handoff([SOURCE], role="coding_agent", revisions=revisions, budget_bytes=8192)
    if handoff.status != "complete" or set(handoff.sources) != set(FILES):
        raise ValueError("dependency closure incomplete")
    # Direct-file baseline does not inherit the treatment's declared edges.
    dense_text = f"/app/climate_analyzer/{SOURCE}\n" + (root / SOURCE).read_text(encoding="utf-8")
    out.mkdir(parents=True, exist_ok=False)
    record = {"schema": 1, "task": "modernize-scientific-stack",
              "task_revision": "7131e4375048a0e408a8fb404b5f499d726b695b",
              "instruction_sha256": sha(task.read_bytes()), "codec_schema": codec.schema.identity,
              "manifest": manifest, "build_seconds": time.perf_counter() - started,
              "baseline": {"text": dense_text, "sources": [SOURCE]},
              "stamp": {"text": handoff.text, "sources": list(handoff.sources)},
              "scope": "visible task-declared files; exact path activation; no learned ranking"}
    (out / "packets.json").write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"files": len(FILES), "stamp_bytes": 32 * len(FILES),
                      "baseline_packet_bytes": len(dense_text.encode()),
                      "closure_packet_bytes": len(handoff.text.encode())}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--instruction", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    run(args.source, args.instruction, args.out)

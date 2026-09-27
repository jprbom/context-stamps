"""Same frozen context treatments with the installed 7B reader.

Copyright (c) 2026 Prashant Jagtap. MIT License.
This diagnostic reference was added after inspecting the 1.5B result. It is
not a second untouched sample or an independent candidate-selection split.
The shared reader implementation remains byte-for-byte unchanged.
"""

import argparse
from pathlib import Path

import multihop_reader as reader
from ruler_native import outside_repo, sha, write_new

# This wrapper changes only the registered reader identity. Source registration
# includes these exact assignments as well as the unchanged shared runner.
reader.MODEL = "qwen2.5-coder:7b"
reader.DIGEST = "dae161e27b0e90dd1856c8bb3209201fd6736d8eb66298e75ed87571486f4364"
_original_sources = reader.sources


def reference_sources():
    return _original_sources() | {"experiments/multihop_reference.py": sha(__file__)}


reader.sources = reference_sources


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "run"))
    for name in ("data", "trained", "tokenizer", "output", "previous_analysis"):
        parser.add_argument("--"+name.replace("_", "-"), type=Path, required=name == "output")
    args = parser.parse_args()
    if args.command == "prepare":
        reader.prepare(args.data, args.trained, args.tokenizer, args.output)
        write_new(args.output/"reference-protocol.json", dict(
            reason="Diagnose reader limitations after small-reader analysis; no selector update",
            previous_analysis_sha256=sha(args.previous_analysis/"summary.json"),
            prospective_model_selection=False, fresh_additional_questions=False))
    else:
        reader.run(outside_repo(args.output))

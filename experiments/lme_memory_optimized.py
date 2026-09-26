"""Same local memory protocol with separately pinned, tie-preserving fast search.

Use a new output directory. The original lme_memory.py protocol stays frozen.
No previous quality or generation-latency score is relabeled as an optimized run.
"""

import argparse
from pathlib import Path

import lme_memory
from lme_rank_selective import ranked_selective
from ruler_native import sha

original_sources = lme_memory.code_sources


def optimized_sources():
    base = Path(__file__).resolve().parent
    return original_sources() | {"experiments/"+p: sha(base/p) for p in
                                 ("lme_memory_optimized.py", "lme_rank_fast.py", "lme_rank_selective.py")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "run"))
    parser.add_argument("--data", type=Path)
    parser.add_argument("--tokenizer", type=Path)
    parser.add_argument("--scorer", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    lme_memory.code_sources = optimized_sources
    lme_memory.ranked = ranked_selective
    if args.command == "prepare":
        lme_memory.prepare(args.data, args.tokenizer, args.scorer, args.output)
    else:
        lme_memory.run(args.output, args.scorer)


if __name__ == "__main__":
    main()

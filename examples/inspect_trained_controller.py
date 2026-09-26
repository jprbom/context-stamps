"""Run the published experimental controller on one verified local cache query.

This is an inference/reproduction example, not a deployment recommendation.
Requires the optional controller dependencies and prepared public-data caches.
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))
from train_controller import batch  # noqa: E402
from train_controller_v2 import load_partitions  # noqa: E402

from context_stamps.contractive_controller import load_contractive  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work", required=True, type=Path)
    parser.add_argument("--dataset", choices=("scifact", "nfcorpus", "arguana", "scidocs", "fiqa"), default="scifact")
    parser.add_argument("--row", type=int, default=0)
    args = parser.parse_args()
    selection = json.loads((ROOT / "evidence/controller-v2/selection.json").read_text(encoding="utf-8"))
    model = load_contractive(ROOT / ("evidence/controller-v2/models/" + selection["selected_student"] + ".safetensors"))
    data = next(d for d in load_partitions(args.work, "test") if d["name"] == args.dataset)
    if not 0 <= args.row < len(data["Q"]):
        parser.error("row outside this collection")
    # Cached public candidates are eligible in this experiment. A real application
    # must derive eligibility from its trusted access/version checks before scoring.
    inputs = batch(data, [args.row], "cpu")
    scores = model.score_candidates(*inputs)[0]
    order = scores.argsort(descending=True, stable=True)[:10].tolist()
    ids = [data["meta"]["document_ids"][data["candidates"][args.row, i]] for i in order]
    print(json.dumps(dict(query_id=data["meta"]["query_ids"][args.row], experimental_student=selection["selected_student"],
         ranked_ids=ids, actual_scope_decision=selection["decisions"][args.dataset]), indent=2))


if __name__ == "__main__":
    main()

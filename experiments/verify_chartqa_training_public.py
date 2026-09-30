"""Check consistency of the public aggregate without requiring GPL dataset bytes.

Copyright (c) 2026 Prashant Jagtap. MIT License.
This verifies the published projection, not the external raw run. For raw replay,
use export_chartqa_training_summary.py with the retained local artifacts.
"""

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def verify():
    data = json.loads((ROOT/"evidence/chartqa-training-v1/summary.json").read_bytes())
    source = ROOT/"experiments/export_chartqa_training_summary.py"
    if data["summary_source_sha256"] != hashlib.sha256(source.read_bytes()).hexdigest():
        raise ValueError("aggregate exporter source changed")
    if (data["schema"] != 1 or data["model_calls"] != 493 or data["paired_records"] != 429
            or data["policy_candidates"] != 3 or data["retained_fold_policies"] != 12
            or data["folds"] != 4 or not data["model_unchanged"]
            or data["candidate_active"] or data["independent_validation_complete"]
            or data["selected_penalty"] is not None):
        raise ValueError("published training scope or inactive decision changed")
    controls = data["controls"]
    expected = {"direct": (112, 94, 39, 143, 84405),
                "memory": (79, 68, 21, 207, 207561),
                "program": (51, 39, 13, 207, 241129)}
    for arm, (correct, exact, charts, calls, tokens) in expected.items():
        record = controls[arm]
        if (record["questions"], record["correct"], record["stripped_exact"],
                record["complete_charts"], record["costs"]["calls"],
                record["costs"]["tokens"]) != (143, correct, exact, charts, calls, tokens):
            raise ValueError("published arm metrics changed")
        if sum(group["questions"] for group in record["subgroup"].values()) != 143:
            raise ValueError("published subgroup denominator changed")
        if sum(group["correct"] for group in record["subgroup"].values()) != correct:
            raise ValueError("published subgroup numerator changed")
    if controls["cv_selected_fixed"] != controls["direct"]:
        raise ValueError("fixed fold control does not equal direct")
    if any(record != controls["direct"] for record in data["cv"].values()):
        raise ValueError("cross-validated route differs from reported direct control")
    cells = data["cells"].values()
    if (sum(cell["questions"] for cell in cells) != 143
            or any(sum(cell[arm] for cell in data["cells"].values()) != correct
                   for arm, (correct, _, _, _, _) in expected.items())):
        raise ValueError("question-cell aggregate does not match arm totals")
    if (data["paired_comparisons"]["any_arm_correct"] != 115
            or data["paired_comparisons"]["all_arms_wrong"] != 28
            or data["paired_comparisons"]["memory_rescues_direct"] != 1
            or data["paired_comparisons"]["program_rescues_direct"] != 2):
        raise ValueError("paired failure aggregate changed")
    if any(key in data for key in ("keys", "rows", "question", "answer", "image_sha256")):
        raise ValueError("case-level field entered public aggregate")
    print("Published ChartQA TRAIN summary is internally consistent; candidate inactive.")


if __name__ == "__main__":
    verify()

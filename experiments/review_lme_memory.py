"""Retain every regression and a limited evidence-presence diagnostic post-run."""

import argparse
import collections
import json
from pathlib import Path

from lme_memory import native_scorer, score
from ruler_native import write_new


def review(run, scorer, output):
    ns = native_scorer(scorer)
    rows = json.loads((run/"scores.json").read_bytes())
    inputs = {r["id"]: r for r in json.loads((run/"inputs.json").read_bytes())}
    keys = {r["id"]: r for r in json.loads((run/"keys.json").read_bytes())}
    lookup = {(r["id"], r["arm"]): r for r in rows if r["phase"] == "holdout"}
    report = dict(status="inactive", interpretation="Native-credit development diagnosis, not semantic truth certification",
                  assumptions_not_met=["independent retention population", "complete deployed retrieval/policy/model resource measurements",
                                       "adversarial and multimodal memory qualification", "reliable local answer verification"],
                  training_model_tokens=sum(r["input_tokens"]+r.get("output_tokens", 0) for r in rows if r["phase"] == "train" and r["arm"] != "none"),
                  all_measured_model_tokens=sum(r["input_tokens"]+r.get("output_tokens", 0) for r in rows if r["arm"] != "learned"),
                  arms={})
    for arm in ("none", "state", "linked", "learned"):
        group = [r for r in rows if r["phase"] == "holdout" and r["arm"] == arm]
        gains = [r["id"] for r in group if r["correct"] and not lookup[r["id"], "state"]["correct"]]
        losses = [r["id"] for r in group if not r["correct"] and lookup[r["id"], "state"]["correct"]]
        cells = []
        for domain, category in sorted({(r["domain"], r["category"]) for r in group}):
            subset = [r for r in group if (r["domain"], r["category"]) == (domain, category)]
            cells.append(dict(domain=domain, category=category, n=len(subset), correct=sum(r["correct"] for r in subset)))
        # Presence of reference phrases can be incidental. Never call it evidence recall.
        presence = []
        for r in group:
            key = keys[r["id"]]
            if not key["eval_function"].startswith("norm_phrase_"):
                continue
            chosen = r.get("selected_arm", arm)
            text = inputs[r["id"]]["prompts"][chosen]
            body = text.split("Memory records (JSON array):\n", 1)[1].rsplit("\n\nQuestion:\n", 1)[0]
            memory = "\n".join(json.loads(body))
            presence.append(dict(id=r["id"], native_correct=r["correct"],
                                 reference_phrase_match_in_selected_text=score(ns, memory, key)))
        report["arms"][arm] = dict(gain_ids=gains, regression_ids=losses, cells=cells,
            exact_unknown=sum(ns["is_unknown"](ns["extract_boxed_answer"](r["prediction"])) for r in group),
            phrase_presence_diagnostic=presence,
            errors=[dict(id=r["id"], error=r["error"]) for r in group if r["error"]])
    report["learned_choices"] = dict(collections.Counter(r["selected_arm"] for r in rows if r["arm"] == "learned"))
    report["all_truncations"] = [dict(id=r["id"], phase=r["phase"], arm=r["arm"], native_correct=r["correct"]) for r in rows if r["truncated"]]
    report["regressions"] = [{k: lookup[i, "learned"][k] for k in ("id", "domain", "category", "selected_arm", "prediction")}
                             | dict(reference=keys[i]["answer"], baseline_prediction=lookup[i, "state"]["prediction"])
                             for i in report["arms"]["learned"]["regression_ids"]]
    write_new(output, report)
    print(json.dumps(dict(learned_choices=report["learned_choices"], gains=len(report["arms"]["learned"]["gain_ids"]),
                          regressions=len(report["regressions"]), status=report["status"])))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("run", "scorer", "output"):
        parser.add_argument("--"+name, type=Path, required=True)
    args = parser.parse_args()
    review(args.run, args.scorer, args.output)

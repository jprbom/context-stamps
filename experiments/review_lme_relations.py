"""Post-run paired quality and measured request-cost diagnostics."""

import argparse
import collections
import json
import statistics
from pathlib import Path

from lme_memory import native_scorer, score
from lme_relations import ARMS, MODELS
from ruler_native import write_new


def strict_reference_agreement(prediction, reference, namespace):
    """Secondary diagnostic: exact phrase lists in the native extracted answer.

    Post-hoc, conservative reference agreement; not semantic factuality. This
    can reject valid paraphrases and does not inspect text outside a final box.
    Native benchmark scores remain unchanged.
    """
    if not score(namespace, prediction, reference):
        return False
    name = reference["eval_function"].split("|", 1)[0].strip()
    if name in ("mc_choice_match", "mc_choice_set_match"):
        return True
    _, options = namespace["parse_eval_function_spec"](reference["eval_function"])
    options = {k: v for k, v in options.items() if k != "require_non_empty"}
    candidate = namespace["split_phrases"](namespace["extract_boxed_answer"](prediction), **options)
    expected = namespace["split_phrases"](reference["answer"], **options)
    if name == "norm_phrase_set_match_ordered":
        return candidate == expected
    return set(candidate) == set(expected)


def diagnostics(rows, namespace, references):
    result = {}
    for model in MODELS:
        lookup = {a: {r["id"]: r for r in rows if r["model"] == model and r["arm"] == a} for a in ARMS}
        arms, pairs = {}, {}
        for arm, by_id in lookup.items():
            group = list(by_id.values())
            cells = collections.defaultdict(lambda: dict(n=0, correct=0))
            for r in group:
                cell = cells[r["domain"]+":"+r["category"]]
                cell["n"] += 1
                cell["correct"] += int(r["correct"])
            durations = [r["request_wall_seconds"] for r in group]
            arms[arm] = dict(total_request_seconds=sum(durations), median_request_seconds=statistics.median(durations),
                p95_request_seconds=statistics.quantiles(durations, n=100, method="inclusive")[94],
                median_retrieval_seconds=statistics.median(r["retrieval_seconds"] for r in group),
                median_pack_seconds=statistics.median(r["pack_seconds"] for r in group),
                median_selected_views=statistics.median(len(r["sources"]) for r in group),
                exact_unknown=sum(namespace["is_unknown"](namespace["extract_boxed_answer"](r["prediction"])) for r in group),
                literal_unknown=sum("UNKNOWN" in r["prediction"].upper() for r in group), cells=dict(cells))
            arms[arm]["strict_reference_agreement"] = sum(
                not r["error"] and strict_reference_agreement(r["prediction"], references[r["id"]], namespace)
                for r in group)
            arms[arm]["native_pass_strict_fail"] = [r["id"] for r in group if r["correct"]
                and not strict_reference_agreement(r["prediction"], references[r["id"]], namespace)]
            if arm != "structure":
                baseline = lookup["structure"]
                pairs[arm] = dict(
                    gains=[i for i, r in by_id.items() if r["correct"] and not baseline[i]["correct"]],
                    regressions=[i for i, r in by_id.items() if baseline[i]["correct"] and not r["correct"]],
                    total_model_token_change=sum(r["input_tokens"]+r["output_tokens"] for r in group)/sum(r["input_tokens"]+r["output_tokens"] for r in baseline.values())-1,
                    total_request_time_change=sum(durations)/sum(r["request_wall_seconds"] for r in baseline.values())-1)
        result[model] = dict(arms=arms, versus_structure=pairs)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ("run", "scorer", "output"):
        parser.add_argument("--"+arg, type=Path, required=True)
    args = parser.parse_args()
    references = {r["id"]: r for r in json.loads((args.run/"references.json").read_bytes())}
    report = diagnostics(json.loads((args.run/"scores.json").read_bytes()), native_scorer(args.scorer), references)
    write_new(args.output, dict(models=report, activation_qualified=False,
        timing="Actual timed retrieval + compilation + local generation; warm indexes/models; excludes index building and model loading",
        qualification="Descriptive development comparison on 72 previously inspected questions, not independent generalization or local-learning qualification",
        strict_diagnostic="Post-hoc exact normalized phrase set/order agreement within the native extracted answer; choices retain native scoring. Can reject valid paraphrases; not semantic factuality or a replacement for native scores."))

"""Post-run diagnostics; never used for selection, fitting or native scoring."""

import argparse
import collections
import json
import re
import statistics
from pathlib import Path

from lme_memory import native_scorer
from ruler_native import write_new


def review(run, scorer):
    rows = json.loads((run/"scores.json").read_bytes())
    inputs = {r["id"]: r for r in json.loads((run/"inputs.json").read_bytes())}
    keys = {r["id"]: r for r in json.loads((run/"keys.json").read_bytes())}
    summaries = json.loads((run/"summary.json").read_bytes())
    namespace = native_scorer(scorer)
    results = {}
    for model, saved in summaries.items():
        model_rows = [r for r in rows if r["model"] == model]
        arms = {}
        for arm in ("raw", "structure"):
            subset = [r for r in model_rows if r["arm"] == arm]
            groups = collections.defaultdict(lambda: dict(n=0, correct=0))
            for r in subset:
                cell = groups[r["domain"]+":"+r["category"]]
                cell["n"] += 1
                cell["correct"] += int(r["correct"])
            arms[arm] = dict(
                exact_unknown=sum(namespace["is_unknown"](namespace["extract_boxed_answer"](r["prediction"])) for r in subset),
                literal_unknown=sum("UNKNOWN" in r["prediction"].upper() for r in subset),
                groups=dict(groups),
                median_retrieval_seconds=statistics.median(r["retrieval_seconds"] for r in subset),
                median_pack_seconds=statistics.median(r["pack_seconds"] for r in subset),
                median_sum_of_stages_seconds=statistics.median(r["retrieval_seconds"]+r["pack_seconds"]+r["generation_wall_seconds"] for r in subset))
        regressions = []
        for i in saved["regressions"]:
            rr = {r["arm"]: r for r in model_rows if r["id"] == i}
            page_counts = {a: len(set(re.findall(r"RootWebArea '([^']+)'", inputs[i]["prompts"][a]))) for a in rr}
            regressions.append(dict(id=i, reference=keys[i]["answer"], eval_function=keys[i]["eval_function"],
                                    predictions={a: r["prediction"] for a, r in rr.items()}, selected_page_title_counts=page_counts))
        results[model] = dict(arms=arms, regressions=regressions,
            model_token_change=(saved["structure"]["input_tokens"]+saved["structure"]["output_tokens"])/(saved["raw"]["input_tokens"]+saved["raw"]["output_tokens"])-1,
            generation_time_change=saved["structure"]["generation_wall_seconds"]/saved["raw"]["generation_wall_seconds"]-1)
    return dict(models=results, activation_qualified=False,
                interpretation="Posthoc descriptive diagnostics; page-title count is not evidence sufficiency or a causal explanation. Stage sum is not measured end-to-end latency.",
                observations=[
                    "The 1.5B edb69441 regression emits all reference columns inside boxed{...} without a backslash. Native scoring is retained unchanged.",
                    "The raw 1.5B answer on 98b62f3d gets native credit despite including Body as an extra mandatory field. Native credit is not exact factuality.",
                    "Both readers regress on 0738c1ac and d5b9fda4. Multiple selected pages expose a scope/selection hypothesis, not a demonstrated cause.",
                    "Structure alone does not preserve action-to-result relationships. Neither variant proves task success or causal effects."])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ("run", "scorer", "output"):
        parser.add_argument("--"+arg, type=Path, required=True)
    args = parser.parse_args()
    write_new(args.output, review(args.run, args.scorer))

"""Load a reviewed experimental predictor; demonstrate conservative fallback.

Copyright (c) 2026 Prashant Jagtap. MIT License.
The three retrieval-score rows are fictional. No model quality claim is made.
"""

import argparse
import json
from dataclasses import asdict

from context_stamps.acquisition import AcquisitionModel, StopPolicy, plan_prefix


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Reviewed acquisition model JSON, not a language model")
    args = parser.parse_args()
    model = AcquisitionModel.load(args.model)
    policy = StopPolicy(model.revision, .9)
    rows = ((.2, 1., 2., 3., .5, 1.), (.7, 2., 4., 2., 1., .5), (.1, 0., 1., 0., .1, .1))
    plan = plan_prefix(rows, model, policy, scope="fictional-unqualified")
    if plan.status != "unqualified_full_pool" or set(plan.indices) != {0, 1, 2}:
        raise RuntimeError("unqualified selector must retain the available pool")
    limited = plan_prefix(rows, model, policy, scope="fictional-unqualified", max_items=1)
    if limited.status != "budget_exhausted":
        raise RuntimeError("resource exhaustion must not be labelled sufficiency")
    print(json.dumps(dict(plan=asdict(plan), limited=asdict(limited),
        note="Indices are suggestions only. Host authorization, dependency closure, conflict checking, full-packet token budgets and answer verification remain required."), indent=2))


if __name__ == "__main__":
    main()

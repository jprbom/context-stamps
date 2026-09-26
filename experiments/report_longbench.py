"""Derived public-reader pilot report; no inference, tuning or new observations."""

import collections
import json
from pathlib import Path

from local_eval import write_new
from longbench_eval import METHODS, file_hash, summarize

ROOT = Path(__file__).resolve().parents[1]
DIRECTORY = ROOT / "evidence/longbench-v2-pilot/run-v2"


def derive(directory=DIRECTORY):
    plan = json.loads((directory / "plan.json").read_text())
    summary = json.loads((directory / "summary.json").read_text())
    if summary["status"] != "completed":
        raise ValueError("A partial run is not a completed benchmark report")
    rows = [json.loads((directory / name).read_text()) for name in sorted(summary["attempt_hashes"])]
    by_key = {(r["id"], r["method"]): r for r in rows}
    strata = []
    for axis in ("domain", "difficulty", "length"):
        for value in sorted({r[axis] for r in rows}):
            group = [r for r in rows if r[axis] == value]
            subset = [r for r in plan["samples"] if r[axis] == value]
            strata.append(dict(axis=axis, value=value, methods=summarize({"samples": subset}, group)))
    comparisons = []
    for method in ("bm25", "no_context"):
        outcomes = collections.Counter()
        for sample in plan["samples"]:
            baseline = by_key[sample["id"], "full"]["correct"]
            treatment = by_key[sample["id"], method]["correct"]
            label = "both_correct" if baseline and treatment else "both_wrong"
            if baseline != treatment:
                label = "full_only" if baseline else "alternative_only"
            outcomes[label] += 1
        comparisons.append(dict(alternative=method, versus="full", counts=dict(outcomes), tasks=len(plan["samples"])))
    telemetry = json.loads((directory / "telemetry.json").read_text())
    samples = [[float(v.strip()) for v in r["sample"].split(",")]
               for r in telemetry if r.get("exit_code") == 0 and r.get("sample")]
    tokens = {m: sum(r["usage"]["prompt_eval_count"] for r in rows if r["method"] == m) for m in METHODS}
    return dict(schema=1, summary_sha256=file_hash(directory/"summary.json"),
        source_hashes={"experiments/report_longbench.py": file_hash(Path(__file__))},
        corpus_tasks=len(plan["eligibility"]), eligible=sum(r["eligible"] for r in plan["eligibility"]),
        selected=len(plan["samples"]), excluded_by_length=sum(not r["eligible"] for r in plan["eligibility"]),
        strata=strata, paired_outcomes=comparisons,
        bm25_input_reduction=1-tokens["bm25"]/tokens["full"],
        first_call=dict(method=rows[0]["method"], whole_request_seconds=rows[0]["whole_request_seconds"],
                        load_seconds=rows[0]["usage"]["load_duration"]/1e9),
        telemetry=dict(samples=len(samples), peak_gpu_utilization_percent=max(r[0] for r in samples),
                       peak_device_memory_mib=max(r[1] for r in samples), peak_temperature_c=max(r[2] for r in samples)),
        resident_models_sha256=file_hash(directory/"resident-models.json"),
        conclusion="No promotion. Small controls-only development subset; reader quality and runtime benefit unqualified.")


if __name__ == "__main__":
    result = derive()
    write_new(DIRECTORY / "analysis.json", result)
    print(json.dumps({k: result[k] for k in ("selected", "eligible", "paired_outcomes", "telemetry")}))

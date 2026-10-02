"""Publish bounded numeric evidence, never benchmark source or raw responses.

Copyright (c) 2026 Prashant Jagtap. MIT License.
"""

import argparse
import hashlib
import json
from pathlib import Path


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def summarize_run(folder):
    plan = read(folder / "plan.json")
    summary = read(folder / "summary.json")
    controls = read(folder / "grader-controls.json")
    if summary["plan_sha256"] != sha(folder / "plan.json"):
        raise ValueError("plan/summary binding failed")
    if controls["empty"]["passed"] or not controls["oracle"]["passed"]:
        raise ValueError("native verifier controls failed")
    rows = []
    for row in summary["rows"]:
        allowed = ("arm", "status", "passed", "steps", "input_tokens", "output_tokens",
                   "model_seconds", "request_seconds", "agent_elapsed_seconds", "valid")
        values = {key: row[key] for key in allowed if key in row}
        if "artifacts" in row:
            values["artifact_present"] = {name: value is not None for name, value in row["artifacts"].items()}
        rows.append(values)
    return {"task": plan["task"], "task_revision": plan["task_revision"],
            "model": {"name": plan["model"]["name"], "digest": plan["model"]["digest"]},
            "order": plan["order"], "verifier_image": plan["verifier_image"],
            "empty_control_passed": controls["empty"]["passed"],
            "oracle_control_passed": controls["oracle"]["passed"],
            "model_unchanged": summary["model_unchanged"], "rows": rows,
            "plan_sha256": sha(folder / "plan.json"),
            "summary_sha256": sha(folder / "summary.json"),
            "controls_sha256": sha(folder / "grader-controls.json")}


def main(args):
    fix = read(args.fix_packets / "packets.json")
    modern = read(args.modern_packets / "packets.json")
    if fix["agreement"]["route_status"] != "abstain":
        raise ValueError("ambiguous source route did not abstain")
    if modern["task_revision"] != fix["task_revision"]:
        raise ValueError("task source revisions differ")
    record = {"schema": 1,
              "purpose": "selected development coding tasks; no full benchmark or deployment claim",
              "fix_preflight": {"task": fix["task"], "revision": fix["task_revision"],
                   "functions": fix["functions"], "index_bytes": fix["index_bytes"],
                   "agreement": fix["agreement"],
                   "packets_sha256": sha(args.fix_packets / "packets.json")},
              "modern_context": {"task": modern["task"], "files": len(modern["manifest"]),
                   "stamp_payload_bytes": 32 * len(modern["manifest"]),
                   "baseline_packet_bytes": len(modern["baseline"]["text"].encode()),
                   "closure_packet_bytes": len(modern["stamp"]["text"].encode()),
                   "packets_sha256": sha(args.modern_packets / "packets.json")},
              "agent_attempt_1": summarize_run(args.agent_v1),
              "agent_attempt_2": summarize_run(args.agent_v2),
              "agent_attempt_3": summarize_run(args.agent_v3),
              "agent_final_source": summarize_run(args.agent_final),
              "code_generation": summarize_run(args.codegen),
              "code_generation_replay": summarize_run(args.codegen_v2),
              "code_generation_final_source": summarize_run(args.codegen_final),
              "frontier_calls": 0, "benchmark_training": False,
              "raw_materials_published": False}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"veto": record["fix_preflight"]["agreement"],
                      "agent_final": record["agent_final_source"]["rows"],
                      "codegen_final": record["code_generation_final_source"]["rows"]}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for name in ("fix_packets", "modern_packets", "agent_v1", "agent_v2", "agent_v3",
                 "agent_final", "codegen", "codegen_v2", "codegen_final", "out"):
        parser.add_argument("--" + name.replace("_", "-"), type=Path, required=True)
    args = parser.parse_args()
    for name in ("fix_packets", "modern_packets", "agent_v1", "agent_v2", "agent_v3",
                 "agent_final", "codegen", "codegen_v2", "codegen_final", "out"):
        setattr(args, name, getattr(args, name).resolve())
    main(args)

"""Publish numeric summaries without benchmark source, code, tests or logs.

Copyright (c) 2026 Prashant Jagtap. MIT License.
"""

import argparse
import hashlib
import json
from pathlib import Path


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def summarize(name, folder):
    plan = read(folder / "plan.json")
    summary = read(folder / "summary.json")
    controls = read(folder / "controls.json")
    if plan["task"] != "multi-source-data-merger" or not summary["model_unchanged"]:
        raise ValueError("task or model identity mismatch")
    if controls["empty"]["passed"] or not controls["oracle"]["passed"]:
        raise ValueError("native controls failed")
    if set(summary["rows"]) != set(plan["order"]):
        raise ValueError("missing or extra arm")
    arms = {}
    for key, row in summary["rows"].items():
        if row["arm"] != key:
            raise ValueError("arm mismatch")
        attempts = row["attempts"]
        if row["input_tokens"] != sum(item["input_tokens"] for item in attempts):
            raise ValueError("input token sum mismatch")
        if row["output_tokens"] != sum(item["output_tokens"] or 0 for item in attempts):
            raise ValueError("output token sum mismatch")
        arms[key] = {"native_passed": row["native_passed"], "submitted": row["submitted"],
                     "input_tokens": row["input_tokens"], "output_tokens": row["output_tokens"],
                     "model_request_seconds": row["model_request_seconds"],
                     "local_execution_seconds": row.get("local_execution_seconds"),
                     "generation_attempts": len(attempts),
                     "local_return_codes": [a.get("local_return_code") for a in attempts
                                            if "local_return_code" in a],
                     "final_execution_return_code": row["execution_return_code"],
                     "path_violations": [a.get("path_violations") for a in attempts],
                     "code_sha256": row["code_sha256"],
                     "prompt_sha256": [a["prompt_sha256"] for a in attempts]}
    return {"name": name, "external_plan_sha256": digest(folder / "plan.json"),
            "external_summary_sha256": digest(folder / "summary.json"),
            "model": plan["model"]["name"], "model_digest": plan["model"]["digest"],
            "num_ctx": plan.get("num_ctx", 16384), "num_predict": plan.get("num_predict", 2048),
            "pattern": plan.get("pattern", "none"), "runtime_repair": plan.get("runtime_repair", False),
            "order": plan["order"], "view_bytes": plan["view_bytes"],
            "stamp_payload_bytes": plan["stamp_payload_bytes"],
            "stamp_build_seconds": plan["stamp_build_seconds"], "arms": arms}


def main(args):
    rows = []
    for name, directory in args.run:
        rows.append(summarize(name, Path(directory).resolve(strict=True)))
    if len({row["name"] for row in rows}) != len(rows):
        raise ValueError("duplicate run names")
    result = {"schema": 1, "task": "multi-source-data-merger",
              "task_revision": "7131e4375048a0e408a8fb404b5f499d726b695b",
              "frozen_protocol": "evidence/terminal-coding-schema-fresh-v1/protocol.md",
              "native_controls": {"empty_passed": False, "reference_passed": True},
              "runs": rows, "frontier_calls": 0, "benchmark_training": False,
              "limits": "one initially unopened task; later runs use inspected development task; no stamp-specific gain"}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({row["name"]: {arm: value["native_passed"] for arm, value in row["arms"].items()}
                      for row in rows}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", nargs=2, action="append", metavar=("NAME", "DIRECTORY"), required=True)
    parser.add_argument("--out", type=Path, required=True)
    main(parser.parse_args())

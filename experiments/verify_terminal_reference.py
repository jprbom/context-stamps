"""Replay the stronger local coding control; no model or code execution."""

import hashlib
import json
from pathlib import Path

from source_evidence import verify_sources
from terminal_pilot import parse_action, render
from verify_terminal_pilot import read, sha, verify_grader

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "evidence/terminal-pilot-coder7b-v1"


def main():
    manifest = read(EVIDENCE / "manifest.json")
    for name, digest in manifest["files"].items():
        assert sha(EVIDENCE / name) == digest, name
    plan, summary = read(EVIDENCE / "plan.json"), read(EVIDENCE / "summary.json")
    baseline = read(ROOT / "evidence/terminal-pilot-v1/attempt-2/plan.json")
    for key in ("config", "max_steps", "max_input_tokens", "tasks", "system_prompt", "task_files", "source_manifest_sha256"):
        assert plan[key] == baseline[key], key
    verify_sources(plan["source_hashes"])
    assert plan["model"]["name"] == "qwen2.5-coder:7b"
    assert plan["model"]["digest"] == "dae161e27b0e90dd1856c8bb3209201fd6736d8eb66298e75ed87571486f4364"
    assert summary["model_unchanged"] and summary["provider_charges_usd"] == 0
    assert read(EVIDENCE / "started.json")["plan_sha256"] == sha(EVIDENCE / "plan.json")
    token_source = read(EVIDENCE / "tokenizer-source.json")
    assert token_source["files"]["tokenizer.json"] == plan["tokenizer"]["tokenizer_sha256"]
    assert token_source["files"]["LICENSE"] == plan["model"]["license_sha256"]
    controls = read(EVIDENCE / "graders-qualified.json")
    assert controls["passed"] and controls["model_calls"] == 0 and len(controls["records"]) == 4
    assert controls["plan_sha256"] == sha(EVIDENCE / "plan.json")
    for control in controls["records"]:
        verify_grader(control["task"], control["result"])
        assert control["result"]["passed"] == control["oracle"]
    assert len(summary["rows"]) == 2
    calls, inputs, outputs, passed = 0, 0, 0, 0
    for row in summary["rows"]:
        trajectory = read(EVIDENCE / (row["task"] + "-trajectory.json"))
        graded = read(EVIDENCE / (row["task"] + "-grade.json"))
        verify_grader(row["task"], graded)
        assert row["passed"] == graded["passed"] == (row["task"] == "cancel-async-tasks")
        assert row["status"] == trajectory["status"] == "step_budget_exhausted"
        assert row["calls"] == len(trajectory["attempts"]) == 12
        assert trajectory["artifact"] is not None
        profile = trajectory["sandbox"]
        assert profile["network_mode"] == "none" and profile["read_only"] and not profile["mounts"]
        messages = trajectory["initial_messages"].copy()
        for attempt in trajectory["attempts"]:
            prompt = render(messages)
            assert hashlib.sha256(prompt.encode()).hexdigest() == attempt["prompt_sha256"]
            response = attempt["response"]
            assert attempt["input_tokens"] == response["prompt_eval_count"]
            assert response["done_reason"] != "length" and response["done"]
            action = parse_action(response["response"])
            assert not action["done"] and attempt["tool_result"]["boundary_failure"] is None
            messages.append({"role": "assistant", "content": response["response"]})
            messages.append({"role": "user", "content": json.dumps({"tool_result": attempt["tool_result"]}, ensure_ascii=False)})
            calls += 1
            inputs += response["prompt_eval_count"]
            outputs += response["eval_count"]
        assert row["input_tokens"] == sum(a["response"]["prompt_eval_count"] for a in trajectory["attempts"])
        assert row["output_tokens"] == sum(a["response"]["eval_count"] for a in trajectory["attempts"])
        passed += row["passed"]
    assert (calls, inputs, outputs, passed) == (24, 24744, 875, 1)
    assert manifest["runtime_treatment"] is False and manifest["training_runs"] == 0
    print("Local coding reference: 1/2 tasks, 24 calls, 24744 input / 875 output tokens.")
    print("Both tasks exhausted 12 steps; controls and complete prompt replay verified. No runtime/training gain.")


if __name__ == "__main__":
    main()

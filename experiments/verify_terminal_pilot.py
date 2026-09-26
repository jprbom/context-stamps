"""Replay terminal-pilot records without Docker, model calls or executing code."""

import hashlib
import json
import re
from pathlib import Path

from source_evidence import verify_sources
from terminal_pilot import parse_action, render

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "evidence/terminal-pilot-v1"
TASK_COUNTS = {"cancel-async-tasks": 6, "log-summary-date-ranges": 2}
REPAIR = ("Action rejected without execution. Return exactly command and done. "
          "Use a nonempty command with done=false to execute it, or an empty command with done=true to finish.")


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_grader(task, grade):
    summary = grade["ctrf"]["results"]["summary"]
    tests = grade["ctrf"]["results"]["tests"]
    assert summary["tests"] == len(tests) == TASK_COUNTS[task]
    assert summary["skipped"] == summary["pending"] == summary["other"] == 0
    assert summary["passed"] == sum(t["status"] == "passed" for t in tests)
    assert summary["failed"] == sum(t["status"] == "failed" for t in tests)
    assert grade["ctrf"]["results"]["tool"]["version"] == "9.0.3"
    assert grade["passed"] == (summary["passed"] == summary["tests"])
    assert grade["passed"] == (grade["test_run"]["return_code"] == 0)
    assert grade["test_run"]["boundary_failure"] is None


def main():
    manifest = read(EVIDENCE / "manifest.json")
    for name, digest in manifest["files"].items():
        assert sha(EVIDENCE / name) == digest, name
    verify_sources(manifest["source_hashes"])
    source = read(EVIDENCE / "upstream/source-manifest.json")
    assert sha(EVIDENCE / "upstream/LICENSE") == source["files"]["LICENSE"]["sha256"]
    sandbox = read(EVIDENCE / "sandbox-qualification.json")
    assert sandbox["passed"] and all(sandbox["boundary_checks"].values())
    assert sandbox["model_calls"] == 0
    for name, case in sandbox["checks"].items():
        assert not case["containers_remaining"]
        assert case["profile"]["network_mode"] == "none"
        assert not case["profile"]["mounts"] and case["profile"]["read_only"]
        assert case["result"]["boundary_failure"] == {"ordinary": None, "deadline": "timeout", "output_limit": "output_limit"}[name]
    hardened = read(EVIDENCE / "sandbox-qualification-hardened.json")
    assert hardened["passed"] and all(hardened["boundary_checks"].values())
    assert hardened["boundary_checks"]["supervisor_module_shadowing_denied"]
    assert hardened["source_sha256"] == manifest["source_hashes"]["experiments/harbor_sandbox.py"]
    assert hardened["model_calls"] == 0
    assert all(not c["containers_remaining"] for c in hardened["checks"].values())
    engineering = read(EVIDENCE / "engineering.json")
    assert len(engineering["runs"]) == 3
    for run in engineering["runs"]:
        assert run["exit_code"] == run["skips"] == 0 and run["tests"] == 16
        log = (EVIDENCE / ("boundary-tests-" + run["environment"] + ".txt")).read_text(encoding="utf-8")
        assert "Ran 16 tests" in log and log.rstrip().endswith("OK")
    old_audit = read(EVIDENCE / "dependency-audit-original.json")
    original = {v["id"] for dep in old_audit["dependencies"] for v in dep["vulns"]}
    assert original == {"PYSEC-2026-1845"}
    patched = read(EVIDENCE / "dependency-audit-patched.json")
    assert len(patched["dependencies"]) == 6  # Resolver omitted packaging; retained as observed.
    assert not any(d.get("vulns") or d.get("skip_reason") for d in patched["dependencies"])
    complete = read(EVIDENCE / "dependency-audit-complete.json")
    pins = dict(re.findall(r"^([a-z-]+)==([0-9.]+)", (EVIDENCE / "verifier-requirements.txt").read_text(), re.M))
    assert len(pins) == len(complete["dependencies"]) == 7
    assert pins == {d["name"]: d["version"] for d in complete["dependencies"]}
    assert not any(d.get("vulns") or d.get("skip_reason") for d in complete["dependencies"])
    calls, inputs, outputs = 0, 0, 0
    for version in (1, 2):
        directory = EVIDENCE / f"attempt-{version}"
        plan, summary = read(directory / "plan.json"), read(directory / "summary.json")
        verify_sources(plan["source_hashes"])
        assert sandbox["source_sha256"] == plan["source_hashes"]["experiments/harbor_sandbox.py"]
        assert plan["task_files"] == source["files"]
        assert plan["source_manifest_sha256"] == sha(EVIDENCE / "upstream/source-manifest.json")
        assert summary["model_unchanged"] and summary["provider_charges_usd"] == 0
        assert read(directory / "started.json")["plan_sha256"] == sha(directory / "plan.json")
        assert plan["tasks"] == list(TASK_COUNTS)
        controls = read(directory / "graders-qualified.json")
        assert controls["passed"] and controls["model_calls"] == 0 and len(controls["records"]) == 4
        if version == 2:
            assert controls["plan_sha256"] == sha(directory / "plan.json")
        for control in controls["records"]:
            verify_grader(control["task"], control["result"])
            assert control["result"]["passed"] == control["oracle"]
        assert len(summary["rows"]) == version
        for row in summary["rows"]:
            trajectory = read(directory / (row["task"] + "-trajectory.json"))
            graded = read(directory / (row["task"] + "-grade.json"))
            verify_grader(row["task"], graded)
            assert row["passed"] == graded["passed"] is False
            assert row["status"] == trajectory["status"]
            assert row["calls"] == len(trajectory["attempts"])
            messages = trajectory["initial_messages"].copy()
            for attempt in trajectory["attempts"]:
                prompt = render(messages)
                assert hashlib.sha256(prompt.encode()).hexdigest() == attempt["prompt_sha256"]
                response = attempt["response"]
                assert attempt["input_tokens"] == response["prompt_eval_count"]
                calls += 1
                inputs += response["prompt_eval_count"]
                outputs += response["eval_count"]
                if response["done_reason"] == "length":
                    assert "tool_result" not in attempt
                    continue
                messages.append({"role": "assistant", "content": response["response"]})
                try:
                    action = parse_action(response["response"])
                except ValueError:
                    assert "tool_result" not in attempt
                    if version == 2:
                        assert attempt["format_error"] == "invalid_action"
                        messages.append({"role": "user", "content": REPAIR})
                    continue
                if not action["done"]:
                    messages.append({"role": "user", "content": json.dumps({"tool_result": attempt["tool_result"]}, ensure_ascii=False)})
            assert row["input_tokens"] == sum(a["response"]["prompt_eval_count"] for a in trajectory["attempts"])
            assert row["output_tokens"] == sum(a["response"]["eval_count"] for a in trajectory["attempts"])
    assert (calls, inputs, outputs) == (6, 2244, 1138)
    print("Terminal pilot: 6 local calls, 2244 input / 1138 output tokens; corrected pilot 0/2 tasks.")
    print("Native grader controls, sandbox records, failed first attempt and patched dependency audit verified.")
    print("Recorded evidence only; no runtime benefit, model training or benchmark ranking.")


if __name__ == "__main__":
    main()

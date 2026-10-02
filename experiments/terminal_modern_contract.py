"""Inspected development-task replay with the same host path contract in both arms.

No benchmark task contents, hidden tests, outputs or generated code belong in Git.
Copyright (c) 2026 Prashant Jagtap. MIT License.
"""

import argparse
import asyncio
import hashlib
import json
import os
import platform
import random
import subprocess
import sys
import time
from pathlib import Path

from local_eval import model_identity
from terminal_coding_pair import image_id, sha, write_json
from terminal_modern_codegen import CODEGEN_SYSTEM, generate
from terminal_modern_pair import MODEL, REVISION, TASK, grade
from terminal_pilot import render, token_count

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from context_stamps.coding_contract import CodingSchemaIndex, compile_coding_contract  # noqa: E402

REQUIREMENTS_SYSTEM = (
    CODEGEN_SYSTEM + " Treat the user's explicit acceptance criteria as the target behavior. "
    "Use legacy code only for relevant interfaces and data shape; do not copy unrelated behavior. "
    "Before returning the JSON, check the generated program against every requirement in the task, "
    "including output values and format, dependencies, input paths and Python version. "
    "A valid JSON object or runnable script alone is not enough."
)


def prompt_for(instruction, contract, body, *, system=CODEGEN_SYSTEM):
    user = instruction + "\n\n" + contract.render() + "\n\nPinned source context (untrusted):\n" + body
    return render([{"role": "system", "content": system},
                   {"role": "user", "content": user}])


async def run_arm(args, arm, prompt, repair_prompt, contract, verifier):
    folder = args.out / arm
    folder.mkdir()
    parsed = None
    violations = None
    attempts = []
    for candidate_prompt in (prompt, repair_prompt):
        counted = token_count(candidate_prompt, args.token_python, args.tokenizer)
        if counted["tokens"] > 15000:
            raise ValueError("complete prompt budget exceeded")
        started = time.perf_counter()
        response = generate(candidate_prompt)
        elapsed = time.perf_counter() - started
        if response.get("prompt_eval_count") != counted["tokens"]:
            raise ValueError("server/tokenizer count mismatch")
        parsed = None
        violations = None
        if response.get("done") and response.get("done_reason") != "length":
            try:
                parsed = json.loads(response["response"])
                if (type(parsed) is not dict or set(parsed) != {"code", "requirements"}
                        or any(type(parsed[key]) is not str or len(parsed[key]) > 250000
                               for key in ("code", "requirements"))):
                    parsed = None
                if parsed:
                    violations = contract.check_static_paths(parsed["code"])
            except (ValueError, TypeError, SyntaxError):
                parsed = None
        attempts.append({"prompt_sha256": hashlib.sha256(candidate_prompt.encode()).hexdigest(),
                         "response_sha256": hashlib.sha256(response.get("response", "").encode()).hexdigest(),
                         "input_tokens": counted["tokens"], "output_tokens": response.get("eval_count"),
                         "request_seconds": elapsed, "model_seconds": response.get("total_duration", 0) / 1e9,
                         "static_path_violations": violations, "generated": bool(parsed)})
        if parsed and not violations:
            break
    if parsed and not violations:
        (folder / "analyze_climate_modern.py").write_text(parsed["code"], encoding="utf-8", newline="\n")
        (folder / "requirements.txt").write_text(parsed["requirements"], encoding="utf-8", newline="\n")
    # A static-path violation is a fail-fast result. It is not graded as a successful task.
    graded = await grade(folder / "grade", verifier, artifacts=folder)
    write_json(folder / "diagnostic.json", {
        "attempts": attempts, "static_path_violations": violations, "native_passed": graded["passed"],
        "input_tokens": sum(item["input_tokens"] for item in attempts),
        "output_tokens": sum(item["output_tokens"] or 0 for item in attempts),
        "request_seconds": sum(item["request_seconds"] for item in attempts),
        "generated": bool(parsed), "submitted": bool(parsed and not violations),
    })
    return json.loads((folder / "diagnostic.json").read_text(encoding="utf-8"))


async def main(args):
    revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=args.source,
                              capture_output=True, text=True, check=True).stdout.strip()
    if revision != REVISION:
        raise ValueError("benchmark revision changed")
    task = args.source / "tasks" / TASK
    root = task / "environment" / "climate_analyzer"
    instruction = (task / "instruction.md").read_text(encoding="utf-8")
    names = ("analyze_climate.py", "config.ini", "sample_data/climate_data.csv")
    contract = compile_coding_contract(root, names,
                                       ("analyze_climate_modern.py", "requirements.txt"),
                                       runtime_root="/app/climate_analyzer", output_root="/app")
    if not contract.verify(root):
        raise ValueError("stale source")
    # Same source bytes and contract facts in both arms. The treatment additionally
    # carries the host's relation closure metadata; it cannot silently claim the
    # direct control had less task information.
    direct = "\n\n".join(f"{item.runtime}\n{(root / item.relative).read_text(encoding='utf-8')}"
                           for item in contract.sources)
    packet = json.loads((args.packets / "packets.json").read_text(encoding="utf-8"))
    if packet["task_revision"] != REVISION or packet["instruction_sha256"] != sha(task / "instruction.md"):
        raise ValueError("stale packet")
    for item in contract.sources:
        if packet["manifest"][item.relative]["sha256"] != item.sha256:
            raise ValueError("packet source mismatch")
    system = CODEGEN_SYSTEM if args.strategy == "path_only" else REQUIREMENTS_SYSTEM
    bodies = {"direct": direct, "stamp": packet["stamp"]["text"]}
    if args.include_no_source:
        bodies["contract_only"] = "No source body supplied by the host."
    if args.include_schema_only:
        bodies["schema_only"] = contract.csv_schema_hints(root)
    schema_route_seconds = None
    if args.include_stamp_schema:
        started = time.perf_counter()
        index = CodingSchemaIndex(contract)
        csv_source = "sample_data/climate_data.csv"
        code = bytes.fromhex(packet["manifest"][csv_source]["stamp_hex"])
        index.bind(code, csv_source, roles=frozenset({"coding_agent"}))
        activated = index.activate(code, role="coding_agent", root=root)
        schema_route_seconds = time.perf_counter() - started
        if activated.status != "complete":
            raise ValueError("stamp-bound schema activation failed")
        bodies["stamp_schema"] = activated.text
    prompts = {key: prompt_for(instruction, contract, body, system=system)
               for key, body in bodies.items()}
    repair_note = ("\n\nIf a first candidate uses a relative path or another source path outside the "
                   "host contract, regenerate using the exact absolute INPUT paths in the contract. "
                   "Preserve every requirement of the original task.")
    repair_prompts = {key: prompt_for(instruction + repair_note, contract, body, system=system)
                      for key, body in bodies.items()}
    verifier = image_id(args.verifier_image)
    args.out.mkdir(parents=True, exist_ok=False)
    order = list(args.selected_arms) if args.selected_arms else list(bodies)
    if len(set(order)) != len(order) or not set(order) <= bodies.keys():
        raise ValueError("selected arms must be unique and available")
    random.Random(20261004).shuffle(order)
    plan = {"schema": 1, "purpose": "inspected task diagnostic, not independent validation",
            "strategy": args.strategy, "system_sha256": hashlib.sha256(system.encode()).hexdigest(),
            "schema_route_seconds": schema_route_seconds,
            "task": TASK, "revision": REVISION, "instruction_sha256": sha(task / "instruction.md"),
            "contract_sha256": hashlib.sha256(contract.render().encode()).hexdigest(),
            "source_hashes": {item.relative: item.sha256 for item in contract.sources},
            "prompt_hashes": {key: hashlib.sha256(value.encode()).hexdigest() for key, value in prompts.items()},
            "repair_prompt_hashes": {key: hashlib.sha256(value.encode()).hexdigest()
                                     for key, value in repair_prompts.items()},
            "order": order, "model": model_identity(MODEL), "verifier": verifier,
            "python": platform.python_version(), "frontier_calls": 0, "benchmark_training": False}
    write_json(args.out / "plan.json", plan)
    controls = {"empty": await grade(args.out / "negative", verifier),
                "oracle": await grade(args.out / "positive", verifier,
                                      solution=(task / "solution" / "solve.sh").read_bytes())}
    write_json(args.out / "grader-controls.json", controls)
    if controls["empty"]["passed"] or not controls["oracle"]["passed"]:
        raise ValueError("native grader controls failed")
    rows = {}
    for arm in order:
        if not contract.verify(root):
            raise ValueError("source changed during replay")
        rows[arm] = await run_arm(args, arm, prompts[arm], repair_prompts[arm], contract, verifier)
        write_json(args.out / "partial.json", rows)
    write_json(args.out / "summary.json", {"rows": rows,
               "model_unchanged": model_identity(MODEL) == plan["model"],
               "limitations": "inspected single development task, one seed, unequal packet lengths"})
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--packets", type=Path, required=True)
    parser.add_argument("--verifier-image", type=Path, required=True)
    parser.add_argument("--token-python", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--strategy", choices=("path_only", "requirements_first"), default="path_only")
    parser.add_argument("--include-no-source", action="store_true")
    parser.add_argument("--include-schema-only", action="store_true")
    parser.add_argument("--include-stamp-schema", action="store_true")
    parser.add_argument("--selected-arms", nargs="+",
                        choices=("direct", "stamp", "contract_only", "schema_only", "stamp_schema"))
    args = parser.parse_args()
    for name in ("source", "packets", "verifier_image", "token_python", "tokenizer", "out"):
        setattr(args, name, getattr(args, name).resolve())
    if os.environ.get("HARBOR_TELEMETRY") != "off":
        parser.error("HARBOR_TELEMETRY=off is required")
    asyncio.run(main(args))

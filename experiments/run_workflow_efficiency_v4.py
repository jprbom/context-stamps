"""Measured sequential planner/coder/reviewer fixtures, including update failures."""

import ast
import json
import random
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from context_stamps import ContextGraph, ContextNode, Family, Stamp256Codec, activate_constrained
from context_stamps.contracts import render_integer_assignments
from experiments.run_local_tasks import MODEL, SYSTEM, api
from experiments.run_spherical import ENCODER, NAMES, encode
from experiments.run_spherical_public import read, save, sha

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "evidence/workflow-efficiency-v4"
METHODS = ("full_current", "exact_graph", "spherical_guarded")


def elapsed(start):
    return (time.perf_counter_ns() - start) / 1e6


def grade(output, expected, coding):
    try:
        value = json.loads(output)
        if not coding:
            return value == expected
        tree = ast.parse(value["code"])
        assignments = {}
        for node in tree.body:
            if not isinstance(node, ast.Assign) or len(node.targets) != 1 or not isinstance(node.targets[0], ast.Name):
                return False
            if not isinstance(node.value, ast.Constant) or type(node.value.value) is not int:
                return False
            if node.targets[0].id in assignments:
                return False
            assignments[node.targets[0].id] = node.value.value
        return assignments == {"SETTING": expected["setting"], "LIMIT": expected["limit"]}
    except (ValueError, KeyError, TypeError, SyntaxError):
        return False


def fixtures(seed):
    rng = random.Random(seed)
    return [{"workflow": i, "target": rng.randrange(8), "setting": rng.randrange(100, 900),
             "limit": rng.randrange(1000, 4000), "new_limit": rng.randrange(5000, 9000),
             "entity_prefix": f"component_{rng.randrange(10**8)}", "failure": "stale" if i % 2 else "revoked"}
            for i in range(8)]


def build(case, method):
    start = time.perf_counter_ns()
    graph, revisions, metadata, candidates = ContextGraph(), {}, [], []
    for i in range(8):
        facets = {"content": "timeout configuration and contract", "entity": f"{case['entity_prefix']}_{i}",
                  "intent": "implement", "task": "timeout"}
        metadata.append(facets)
        stamp = encode(facets, 17, "views") if method == "spherical_guarded" else None
        candidates.append(stamp)
        setting = case["setting"] if i == case["target"] else 10 + i
        limit = case["limit"] if i == case["target"] else 100 + i
        graph.put(ContextNode(f"root-{i}", json.dumps({**facets, "setting": setting}), "1", frozenset({"worker"}), stamp))
        graph.put(ContextNode(f"contract-{i}", f"Contract for {facets['entity']}. Approved limit: {limit}.", "1", frozenset({"worker"})))
        graph.link(f"root-{i}", f"contract-{i}", "depends_on", provenance="fixture-contract")
        revisions[f"root-{i}"] = revisions[f"contract-{i}"] = "1"
    codec = Stamp256Codec({name: Family(ENCODER.identity, 64, 64, 17 + i)
                           for i, name in enumerate(NAMES)}) if method == "spherical_guarded" else None
    return graph, revisions, metadata, candidates, codec, elapsed(start)


def mutate(case, stage, graph, revisions):
    start = time.perf_counter_ns()
    root, dep = f"root-{case['target']}", f"contract-{case['target']}"
    if stage == 2:
        old = graph._nodes[dep]
        if case["failure"] == "stale":
            node = ContextNode(dep, old.text.replace(str(case["limit"]), str(case["new_limit"])), "2", old.roles)
            revisions[dep] = "2"
        else:
            node = ContextNode(dep, old.text, old.revision, frozenset({"administrator"}))
        graph.put(node)
    if stage == 3:
        old = graph._nodes[dep]
        graph.put(ContextNode(dep, old.text, old.revision, frozenset({"worker"})))
        graph.link(root, dep, "depends_on", provenance="fixture-contract-revalidated")
    return elapsed(start)


def main():
    protocol = read(OUT / "protocol.json")
    cases = fixtures(protocol["seed"])
    rng = random.Random(protocol["seed"] + 1)
    rows, setups = [], []
    warmup = api("generate", {"model": MODEL, "system": SYSTEM, "prompt": "Return setting 1 and limit 2.",
        "stream": False, "format": {"type": "object", "properties": {"setting": {"type": "integer"}, "limit": {"type": "integer"}},
        "required": ["setting", "limit"], "additionalProperties": False}, "keep_alive": "10m",
        "options": {"temperature": 0, "seed": protocol["seed"], "num_predict": 96, "num_ctx": 8192}})
    for repeat in range(2):
        for case in cases:
            states = {method: build(case, method) for method in METHODS}
            setups.extend({"workflow": case["workflow"], "repeat": repeat, "method": method, "setup_ms": state[-1]}
                          for method, state in states.items())
            for stage in range(4):
                methods = list(METHODS)
                rng.shuffle(methods)
                for method in methods:
                    graph, revisions, metadata, candidates, codec, _ = states[method]
                    start = time.perf_counter_ns()
                    update_ms = mutate(case, stage, graph, revisions)
                    target = metadata[case["target"]]
                    required = {k: target[k] for k in ("entity", "intent", "task")}
                    transport_bytes = 0
                    if method == "spherical_guarded":
                        query = encode(target, 17, "views")
                        payload = codec.pack(query)
                        query = codec.unpack(payload, schema_id=codec.schema.identity)
                        transport_bytes = len(payload)
                        hits = activate_constrained(query, candidates, metadata=metadata, required=required, threshold=.5, limit=1)
                        roots = [f"root-{h['index']}" for h in hits]
                    else:
                        roots = [f"root-{i}" for i, fields in enumerate(metadata) if all(fields[k] == v for k, v in required.items())]
                    packet = graph.handoff(roots, role="worker", revisions=revisions, budget_bytes=65536)
                    if method == "full_current" and packet.status == "complete":
                        packet = graph.handoff([f"root-{i}" for i in range(8)], role="worker", revisions=revisions, budget_bytes=65536)
                    expected = {"setting": case["setting"], "limit": case["new_limit"] if stage == 3 and case["failure"] == "stale" else case["limit"]}
                    prompt, output, response = "", "", {}
                    should_abstain = stage == 2
                    if packet.status == "complete":
                        task = ('For the configuration-code stage, extract the two integer constants. ' if stage == 1 else '')
                        task += 'Return JSON {"setting": number, "limit": number}.'
                        prompt = "EVIDENCE\n" + packet.text + "\nTASK\nFor entity " + target["entity"] + ", use its setting and dependent contract limit. " + task
                        response = api("generate", {"model": MODEL, "system": SYSTEM, "prompt": prompt,
                            "stream": False, "format": {"type": "object", "properties": {"setting": {"type": "integer"}, "limit": {"type": "integer"}},
                                 "required": ["setting", "limit"], "additionalProperties": False}, "keep_alive": "10m",
                            "options": {"temperature": 0, "seed": protocol["seed"], "num_predict": 96, "num_ctx": 8192}})
                        output = response["response"]
                    rendered_code = ""
                    graded_output = output
                    if stage == 1 and output:
                        try:
                            values = json.loads(output)
                            if set(values) != {"setting", "limit"}:
                                raise ValueError("unexpected model fields")
                            rendered_code = render_integer_assignments(
                                {"SETTING": values["setting"], "LIMIT": values["limit"]}, allowed_names=("SETTING", "LIMIT"))
                            graded_output = json.dumps({"code": rendered_code})
                        except (ValueError, TypeError, KeyError):
                            graded_output = ""
                    correct = (packet.status != "complete") if should_abstain else (packet.status == "complete" and grade(graded_output, expected, stage == 1))
                    rows.append({"workflow": case["workflow"], "repeat": repeat, "stage": stage, "method": method,
                                 "failure_fixture": case["failure"], "expected": expected, "should_abstain": should_abstain,
                                 "abstained": packet.status != "complete", "correct": correct,
                                 "prompt": prompt, "output": output, "rendered_code": rendered_code, "end_to_end_ms": elapsed(start), "update_ms": update_ms,
                                 "prompt_tokens": response.get("prompt_eval_count", 0), "output_tokens": response.get("eval_count", 0),
                                 "prefill_ms": response.get("prompt_eval_duration", 0) / 1e6,
                                 "decode_ms": response.get("eval_duration", 0) / 1e6,
                                 "context_bytes": packet.units, "transport_bytes": transport_bytes, "load_ms": response.get("load_duration", 0) / 1e6})
            save(OUT / "results.json", rows)
            print("workflow", repeat, case["workflow"], flush=True)
    summary = []
    for method in METHODS:
        selected = [r for r in rows if r["method"] == method]
        durations = [sum(r["end_to_end_ms"] for r in selected if (r["workflow"], r["repeat"]) == (case, repeat))
                     + next(s["setup_ms"] for s in setups if (s["method"], s["workflow"], s["repeat"]) == (method, case, repeat))
                     for case in range(8) for repeat in range(2)]
        summary.append({"method": method, "workflow_attempts": 16, "unique_workflows": 8,
                        "complete_workflow_success": sum(all(r["correct"] for r in selected if (r["workflow"], r["repeat"]) == (case, repeat))
                                                         for case in range(8) for repeat in range(2)),
                        "correct_stages": sum(r["correct"] for r in selected), "stages": len(selected),
                        "model_calls": sum(not r["abstained"] for r in selected),
                        "input_tokens_total": sum(r["prompt_tokens"] for r in selected),
                        "output_tokens_total": sum(r["output_tokens"] for r in selected),
                        "mean_workflow_ms_including_setup": statistics.mean(durations),
                        "median_workflow_ms_including_setup": statistics.median(durations),
                        "p95_workflow_ms_including_setup": sorted(durations)[-1],
                        "p95_definition": "nearest-rank p95 of16 attempts equals maximum",
                        "mean_setup_ms": statistics.mean(s["setup_ms"] for s in setups if s["method"] == method)})
    for name, value in (("fixtures", cases), ("setup", setups), ("summary", summary)):
        save(OUT / (name + ".json"), value)
    metadata = api("show", {"model": MODEL})
    metadata.pop("modelfile", None)
    save(OUT / "manifest.json", {"protocol": protocol, "model_metadata": metadata, "ollama_version": api("version"),
                                 "warmup_input_tokens": warmup.get("prompt_eval_count"),
                                 "schema_bootstrap_bytes": len(json.dumps(states["spherical_guarded"][4].schema.views).encode()),
                                 "source_sha256": {p: sha(ROOT / p) for p in ("experiments/run_workflow_efficiency_v4.py", "context_stamps/stamp256.py", "context_stamps/contracts.py",
                                    "context_stamps/workflow.py", "context_stamps/guarded_activation.py",
                                    "context_stamps/activation.py", "experiments/run_spherical.py")}})
    save(OUT / "checksums.json", {p.name: sha(p) for p in OUT.glob("*.json") if p.name != "checksums.json"})


if __name__ == "__main__":
    main()

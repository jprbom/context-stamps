"""Fresh-entity regression after exact-constraint guard; retains unguarded controls."""

import hashlib
import json
import platform
import random
import statistics
import sys
import time
from pathlib import Path

import tiktoken

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from context_stamps.activation import StampSchema, activate
from context_stamps.facet_model import FacetModel
from context_stamps.guarded_activation import activate_constrained
from context_stamps.workflow import ContextGraph, ContextNode
from experiments.run_local_tasks import MODEL, SYSTEM, api
from experiments.run_spherical import encode, prepare, save
from experiments.run_spherical_v2 import counterbalanced

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "evidence/spherical-guarded-v1"
DATA = ROOT / "evidence/spherical-v2"


def main():
    cases = json.loads((DATA / "fixtures.json").read_text())
    cases["test"] = counterbalanced("test", 80, 98127)
    encoder = tiktoken.get_encoding("cl100k_base")
    results, chosen = [], {}
    for seed in (17, 41, 83):
        model = FacetModel.from_json((DATA / f"facet-model-{seed}.json").read_text())
        val = prepare(cases["validation"], seed, "views")
        trials = []
        for threshold in (.25, .5, .75, 1.0):
            tp = fp = fn = 0
            for case, query, candidates in val:
                hits = {r["index"] for r in activate(query, candidates, threshold=threshold, limit=8, model=model)}
                tp += int(case["positive"] in hits)
                fp += len(hits - {case["positive"]})
                fn += int(case["positive"] not in hits)
            trials.append({"threshold": threshold, "tp": tp, "fp": fp, "fn": fn,
                           "f1": 2 * tp / max(1, 2 * tp + fp + fn)})
        threshold = max(trials, key=lambda r: r["f1"])["threshold"]
        chosen[str(seed)] = {"threshold": threshold, "validation": trials}
        for split in ("test", "ood"):
            for case, query, candidates in prepare(cases[split], seed, "views"):
                hits = activate(query, candidates, threshold=threshold, limit=8, model=model)
                ids = [r["index"] for r in hits]
                exact = [i for i, c in enumerate(case["candidates"]) if all(
                    case["query"][name] == c[name] for name in ("entity", "intent", "task"))]
                schema = StampSchema.for_stamp(query)
                compact = schema.pack(query)
                assert schema.unpack(compact) == query
                bootstrap = json.dumps({"schema": schema.identity, "views": schema.views})
                results.append({"case": case["id"], "seed": seed, "split": split,
                                "positive": case["positive"], "activated": ids,
                                "recall": int(case["positive"] in ids), "false_activations": len(set(ids) - {case["positive"]}),
                                "no_answer_false_activation": bool(set(ids) - {case["positive"]}),
                                "exact_field_correct": exact == [case["positive"]],
                                "compact_bytes": len(compact), "json_bytes": len(query.to_payload()),
                                "compact_tokens": len(encoder.encode(compact)),
                                "json_tokens": len(encoder.encode(query.to_payload())),
                                "bootstrap_tokens": len(encoder.encode(bootstrap))})
    save(OUT / "activation.json", results)
    save(OUT / "calibration.json", chosen)
    summaries = []
    for split in ("test", "ood"):
        rows = [r for r in results if r["split"] == split]
        summaries.append({"split": split, "query_seed_pairs": len(rows), **{
            k: statistics.mean(r[k] for r in rows) for k in (
                "recall", "false_activations", "no_answer_false_activation", "exact_field_correct",
                "compact_bytes", "json_bytes", "compact_tokens", "json_tokens", "bootstrap_tokens")}})
    save(OUT / "activation-summary.json", summaries)

    # Live tasks: synthetic fields/edges supplied by application; no neural training.
    model = FacetModel.from_json((DATA / "facet-model-17.json").read_text())
    threshold = chosen["17"]["threshold"]
    tasks, indexes = [], {}
    rng = random.Random(61193)
    for case in cases["test"][:12]:
        t0 = time.perf_counter_ns()
        graph = ContextGraph()
        revisions = {}
        candidates = []
        for i, c in enumerate(case["candidates"]):
            key, dep = f"root-{i}", f"contract-{i}"
            value, limit = rng.randrange(100, 999), rng.randrange(1000, 9999)
            node_text = json.dumps({**c, "setting": value})
            graph.put(ContextNode(key, node_text, "1", frozenset({"worker"}), encode(c, 17, "views")))
            graph.put(ContextNode(dep, f"Contract for {key}. Approved limit: {limit}.", "1", frozenset({"worker"})))
            graph.link(key, dep, "depends_on", provenance="procedural contract")
            revisions[key] = revisions[dep] = "1"
            candidates.append(graph._nodes[key].stamp)
            if i == case["positive"]:
                answer = {"setting": value, "limit": limit}
        setup_ms = (time.perf_counter_ns() - t0) / 1e6
        task = {"case": case["id"], "query": case["query"], "expected": answer, "index_setup_ms": setup_ms}
        tasks.append(task)
        indexes[case["id"]] = (graph, revisions, candidates)
    save(OUT / "slm-fixtures.json", tasks)
    warmup = api("generate", {"model": MODEL, "prompt": "Return {}", "stream": False,
                              "format": "json", "options": {"temperature": 0, "num_predict": 8}})
    rows = []
    for repetition in range(2):
        for task in tasks:
            modes = ["full", "stamp_only", "activated_graph", "guarded_graph", "exact_graph"]
            rng.shuffle(modes)
            for mode in modes:
                graph, revisions, candidates = indexes[task["case"]]
                t0 = time.perf_counter_ns()
                transport_tokens = 0
                if mode == "full":
                    packet = graph.handoff([f"root-{i}" for i in range(8)], role="worker",
                                           revisions=revisions, budget_bytes=65536)
                else:
                    query = encode(task["query"], 17, "views")
                    schema = StampSchema.for_stamp(query)
                    transport = schema.pack(query)
                    query = schema.unpack(transport)
                    hits = activate(query, candidates, threshold=threshold, limit=1, model=model)
                    if mode in {"guarded_graph", "exact_graph"}:
                        required = {k: task["query"][k] for k in ("entity", "intent", "task")}
                        metadata = [json.loads(graph._nodes[f"root-{i}"].text) for i in range(8)]
                        if mode == "guarded_graph":
                            hits = activate_constrained(query, candidates, metadata=metadata,
                                required=required, threshold=threshold, limit=1, model=model)
                        else:
                            hits = [{"index": i} for i, m in enumerate(metadata)
                                    if all(m.get(k) == v for k, v in required.items())]
                    roots = [f"root-{hits[0]['index']}"] if hits else []
                    transport_tokens = len(encoder.encode(transport))
                    target = graph
                    if mode == "stamp_only":
                        target = ContextGraph()
                        for key in roots:
                            target.put(graph._nodes[key])
                    packet = target.handoff(roots, role="worker", revisions=revisions,
                                            budget_bytes=65536) if roots else None
                text = packet.text if packet else "No matching evidence."
                prompt = ("EVIDENCE\n" + text + "\nTASK\nFind the record matching these exact facets: "
                          + json.dumps({k: task["query"][k] for k in ("entity", "intent", "task")})
                          + '. Return JSON {"setting": number, "limit": number} using its setting and its '
                          + 'dependent contract. Use null for any missing value.')
                assembly_ms = (time.perf_counter_ns() - t0) / 1e6
                request_start = time.perf_counter_ns()
                response = api("generate", {"model": MODEL, "system": SYSTEM, "prompt": prompt,
                               "stream": False, "format": "json", "keep_alive": "10m",
                               "options": {"temperature": 0, "seed": 61193, "num_predict": 96, "num_ctx": 8192}})
                request_ms = (time.perf_counter_ns() - request_start) / 1e6
                try:
                    correct = json.loads(response["response"]) == task["expected"]
                except (ValueError, TypeError):
                    correct = False
                rows.append({"case": task["case"], "mode": mode, "repetition": repetition,
                             "prompt": prompt, "output": response["response"], "expected": task["expected"],
                             "correct": correct, "prompt_tokens": response.get("prompt_eval_count"),
                             "output_tokens": response.get("eval_count"), "transport_tokens_cl100k": transport_tokens,
                             "assembly_ms": assembly_ms, "request_ms": request_ms,
                             "end_to_end_ms": assembly_ms + request_ms,
                             "index_setup_ms": task["index_setup_ms"]})
            print("SLM", repetition, task["case"], flush=True)
    save(OUT / "slm-results.json", rows)
    save(OUT / "slm-summary.json", [{"mode": mode, "calls": 24, "unique_cases": 12, **{
        k: statistics.mean(r[k] for r in rows if r["mode"] == mode)
        for k in ("correct", "prompt_tokens", "output_tokens", "transport_tokens_cl100k",
                  "assembly_ms", "request_ms", "end_to_end_ms", "index_setup_ms")}}
        for mode in ("full", "stamp_only", "activated_graph", "guarded_graph", "exact_graph")])
    sources = ["context_stamps/activation.py", "experiments/run_guarded_spherical.py", "context_stamps/guarded_activation.py", "experiments/run_spherical_v2.py"]
    save(OUT / "manifest.json", {"python": sys.version, "platform": platform.platform(), "model": MODEL,
         "ollama_version": api("version"), "model_metadata": api("show", {"model": MODEL}),
         "warmup_prompt_tokens": warmup.get("prompt_eval_count"),
         "source_sha256": {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in sources},
         "input_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in [DATA / "fixtures.json", *DATA.glob("facet-model-*.json")]}})
    save(OUT / "checksums.json", {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                 for p in OUT.glob("*.json") if p.name != "checksums.json"})


if __name__ == "__main__":
    main()

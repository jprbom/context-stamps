"""Prospective procedural experiment: supplied facets, learned scores, graph ablations."""

import hashlib
import json
import platform
import random
import statistics
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import tiktoken

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from context_stamps.facet_model import FacetModel
from context_stamps.spherical import SphericalStamp
from context_stamps.workflow import ContextGraph, ContextNode
from stamps import Family, HashingEncoder

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "evidence/spherical-v1"
ENCODER = HashingEncoder(64)
NAMES = ("content", "entity", "intent", "task")


def save(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8", newline="\n")


def fixtures(split, count, seed):
    rng = random.Random(seed)
    result = []
    tasks = ["calibration", "segmentation", "quantization"] if split == "ood" else ["timeout", "retry", "schema"]
    for i in range(count):
        entity = f"{split}_component_{rng.randrange(100000000)}"
        task = rng.choice(tasks)
        intent = rng.choice(["implement", "review", "measure"])
        query = dict(content=f"Inspect {task} behavior and make required changes", entity=entity,
                     intent=intent, task=task)
        # Positive paraphrase and exact-content hard negatives, with known explicit facets.
        correct = dict(content=f"Procedure and evidence for {task} modification", entity=entity,
                       intent=intent, task=task)
        candidates = [correct]
        for j in range(7):
            item = dict(query)
            if j % 3 == 0:
                item["entity"] = f"{split}_other_{rng.randrange(100000000)}"
            elif j % 3 == 1:
                item["intent"] = "unrelated_intent_" + str(j)
            else:
                item["task"] = "unrelated_task_" + str(j)
            candidates.append(item)
        indices = list(range(len(candidates)))
        rng.shuffle(indices)
        result.append({"id": f"{split}-{i}", "query": query,
                       "candidates": [candidates[j] for j in indices], "positive": indices.index(0)})
    return result


def encode(facets, seed, mode):
    if mode == "single":
        names, bits = ("content",), 256
    else:
        names, bits = NAMES, 64
    families = {name: Family(ENCODER.identity, 64, bits, seed + index)
                for index, name in enumerate(names)}
    vectors = {name: ENCODER.encode(facets["content" if mode == "directions" else name]) for name in names}
    return SphericalStamp.encode(vectors, families)


def prepare(cases, seed, mode):
    return [(case, encode(case["query"], seed, mode),
             [encode(c, seed, mode) for c in case["candidates"]]) for case in cases]


def evaluate(prepared, model=None):
    rows = []
    for case, query, candidates in prepared:
        start = time.perf_counter_ns()
        scores = [query.score(c) if model is None else model.score(query, c) for c in candidates]
        order = sorted(range(len(scores)), key=lambda i: -scores[i])
        elapsed = (time.perf_counter_ns() - start) / 1e6
        rank = order.index(case["positive"]) + 1
        rows.append({"case": case["id"], "positive": case["positive"], "order": order,
                     "scores": scores, "top1": int(rank == 1), "rr": 1 / rank, "score_ms": elapsed})
    return rows


def aggregate(rows):
    return {key: statistics.mean(row[key] for row in rows) for key in ("top1", "rr", "score_ms")}


def workflow_cases(prepared, model):
    tokenizer = tiktoken.get_encoding("cl100k_base")
    rows = []
    for case, query, candidates in prepared[:60]:
        start = time.perf_counter_ns()
        graph = ContextGraph()
        revisions = {}
        for i, candidate in enumerate(candidates):
            key = f"root-{i}"
            revisions[key] = "1"
            text = json.dumps(case["candidates"][i]) + "\n" + ("Recorded experimental observation. " * 48)
            graph.put(ContextNode(key, text, "1", frozenset({"worker"}), candidate))
            dependency = f"dependency-{i}"
            revisions[dependency] = "1"
            graph.put(ContextNode(dependency, f"Contract for {key}: preserve units and verify boundary conditions.",
                                  "1", frozenset({"worker"})))
            graph.link(key, dependency, "depends_on", provenance="procedural declared contract")
        setup_ms = (time.perf_counter_ns() - start) / 1e6
        target = f"root-{case['positive']}"
        expected = {target, f"dependency-{case['positive']}"}
        selected_index = max(range(len(candidates)), key=lambda i: model.score(query, candidates[i]))
        selected = f"root-{selected_index}"
        all_roots = [f"root-{i}" for i in range(8)]
        for mode in ("full_authorized", "graph_known_root", "stamp_only", "stamp_plus_graph"):
            t0 = time.perf_counter_ns()
            if mode == "stamp_only":
                # Resolve just the root in a graph without dependencies: explicit ablation.
                flat = ContextGraph()
                flat.put(graph._nodes[selected])
                packet = flat.handoff([selected], role="worker", revisions=revisions, budget_bytes=65536)
            else:
                roots = all_roots if mode == "full_authorized" else [target if mode == "graph_known_root" else selected]
                packet = graph.handoff(roots, role="worker", revisions=revisions, budget_bytes=65536)
            elapsed = (time.perf_counter_ns() - t0) / 1e6
            # Include the serialized query stamp in the stamp-method transport total.
            transport = query.to_payload() + "\n" if mode.startswith("stamp") else ""
            rows.append({"case": case["id"], "mode": mode, "sources": packet.sources,
                         "expected": sorted(expected), "complete": int(expected <= set(packet.sources)),
                         "status": packet.status, "tokens": len(tokenizer.encode(transport + packet.text)),
                         "packet_tokens": len(tokenizer.encode(packet.text)), "handoff_ms": elapsed,
                         "graph_setup_ms": setup_ms, "bytes": len((transport + packet.text).encode("utf-8"))})
        for failure in ("stale_revision", "changed_dependency", "unauthorized_dependency", "budget", "conflict"):
            dep = f"dependency-{case['positive']}"
            original = graph._nodes[dep]
            altered_revisions = dict(revisions)
            budget = 65536
            if failure == "stale_revision":
                altered_revisions[dep] = "2"
            elif failure == "changed_dependency":
                graph.put(ContextNode(dep, "changed contract", "1", original.roles))
            elif failure == "unauthorized_dependency":
                graph.put(ContextNode(dep, original.text, "1", frozenset({"private"})))
            elif failure == "budget":
                budget = 1
            else:
                graph.link(target, dep, "contradicts", provenance="fixture conflict")
            packet = graph.handoff([target], role="worker", revisions=altered_revisions, budget_bytes=budget)
            rows.append({"case": case["id"], "mode": failure, "status": packet.status,
                         "empty": not packet.text, "correct_abstention": packet.status == "insufficient" and not packet.text})
            graph.put(original)
    return rows


def main():
    protocol = json.loads((OUT / "protocol.json").read_text())
    cases = {split: fixtures(split, n, protocol["split_seeds"][split])
             for split, n in protocol["splits"].items()}
    save(OUT / "fixtures.json", cases)
    all_rows, training, summaries = [], [], []
    workflow = []
    for seed in protocol["seeds"]:
        prepared = {split: prepare(values, seed, "views") for split, values in cases.items()}
        train = prepared["train"]
        features, labels = [], []
        for case, query, candidates in train:
            for i, candidate in enumerate(candidates):
                features.append(list(query.compare(candidate).values()))
                labels.append(int(i == case["positive"]))
        trials = []
        for penalty in (0.01, 1.0, 100.0):
            t0 = time.perf_counter_ns()
            model = FacetModel.fit(train[0][1], features, labels, penalty=penalty)
            elapsed = (time.perf_counter_ns() - t0) / 1e6
            metrics = aggregate(evaluate(prepared["validation"], model))
            trials.append((metrics["top1"], metrics["rr"], model, penalty))
            training.append({"seed": seed, "penalty": penalty, "validation": metrics,
                             "fit_ms": elapsed, "examples": len(labels)})
        _, _, model, penalty = max(trials, key=lambda x: (x[0], x[1]))
        (OUT / f"facet-model-{seed}.json").write_text(model.to_json() + "\n", encoding="utf-8", newline="\n")
        for split in ("test", "ood"):
            for method, mode in (("single_content_256", "single"), ("directions_4x64", "directions"),
                                 ("views_4x64_uniform", "views"), ("views_4x64_learned", "views")):
                t0 = time.perf_counter_ns()
                data = prepared[split] if mode == "views" else prepare(cases[split], seed, mode)
                preparation_ms = (time.perf_counter_ns() - t0) / 1e6
                rows = evaluate(data, model if method.endswith("learned") else None)
                all_rows.extend(dict(row, seed=seed, split=split, method=method) for row in rows)
                summaries.append({"seed": seed, "split": split, "method": method, "queries": len(rows),
                                  "metrics": aggregate(rows), "penalty": penalty if method.endswith("learned") else None,
                                  "preparation_ms": preparation_ms,
                                  "preparation_cached": mode == "views", "fingerprint_bits": data[0][1].bits,
                                  "query_payload_bytes": len(data[0][1].to_payload().encode())})
        if seed == protocol["seeds"][0]:
            workflow = workflow_cases(prepared["test"], model)
        print("Completed seed", seed, flush=True)
    save(OUT / "training.json", training)
    save(OUT / "retrieval.json", all_rows)
    save(OUT / "summary.json", summaries)
    save(OUT / "workflow.json", workflow)
    groups = {}
    for mode in protocol["workflow_ablations"] + protocol["negative_workflows"]:
        group = [r for r in workflow if r["mode"] == mode]
        groups[mode] = {"cases": len(group)}
        for key in ("complete", "tokens", "packet_tokens", "handoff_ms", "graph_setup_ms", "correct_abstention"):
            if key in group[0]:
                groups[mode][key] = statistics.mean(r[key] for r in group)
    save(OUT / "workflow-summary.json", groups)
    sources = ["context_stamps/spherical.py", "context_stamps/workflow.py", "context_stamps/facet_model.py",
               "experiments/run_spherical.py", "stamps.py"]
    save(OUT / "manifest.json", {"protocol": protocol, "python": sys.version, "platform": platform.platform(),
         "numpy": np.__version__, "tokenizer": tiktoken.__version__, "device": "CPU; no GPU training in this run",
         "git_parent": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
         "source_sha256": {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in sources}})
    save(OUT / "checksums.json", {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                 for p in sorted(OUT.glob("*.json")) if p.name != "checksums.json"})


if __name__ == "__main__":
    main()

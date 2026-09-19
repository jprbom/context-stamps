"""Train small selectors and evaluate changing-evidence fixtures. CPU, no downloads."""

import argparse
import hashlib
import json
import platform
import sys
import time
import tracemalloc
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from context_stamps.baselines import bm25
from context_stamps.memory import ContextMemory
from context_stamps.selection import fit_selector, select_evidence
from experiments.fixtures import generate
from stamps import Family, HashingEncoder, content_digest, similarity, stamp_vector


def save(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")


class Rows:
    def __init__(self, rows):
        self.rows = rows

    def _rank(self, query, revisions):
        return sorted((dict(r) for r in self.rows), key=lambda r: (-r["score"], r["source"]))


def prepare(case, encoder, family):
    q = stamp_vector(encoder.encode(case["query"]), family)
    lexical = bm25(case["query"], [d["text"] for d in case["documents"]])
    peak = max(lexical, default=1) or 1
    rows = []
    for doc, lex in zip(case["documents"], lexical):
        row = {**doc, "dependencies": "{}"}
        version = content_digest("changed/" + doc["text"]) if doc["stale"] else doc["digest"]
        row["freshness"] = ContextMemory._freshness({**row, "stale": False}, {doc["source"]: version})
        row["score"] = similarity(q, stamp_vector(encoder.encode(doc["text"]), family))
        row["bm25"] = lex / peak
        rows.append(row)
    return rows


def run(out):
    out.mkdir(parents=True, exist_ok=True)
    cases = generate()
    fixture = "".join(json.dumps(c, sort_keys=True) + "\n" for c in cases)
    (out / "fixtures.jsonl").write_text(fixture, encoding="utf-8", newline="\n")
    encoder = HashingEncoder()
    family = Family(encoder.identity, encoder.dim)
    prepared = {c["id"]: prepare(c, encoder, family) for c in cases}
    # Independent fixture oracle: measure the effect of restoring each document
    # when one required fact is omitted. This is evidence coverage, not an LLM judge.
    interventions = []
    for case in cases:
        for omitted in case["required_facts"]:
            background = {
                fact
                for row in prepared[case["id"]]
                if not row["stale"] and omitted not in row["facts"]
                for fact in row["facts"]
            }
            for row in prepared[case["id"]]:
                restored = background | (set(row["facts"]) if not row["stale"] else set())
                before = int(set(case["required_facts"]) <= background)
                after = int(set(case["required_facts"]) <= restored)
                interventions.append(
                    {
                        "case": case["id"],
                        "split": case["split"],
                        "omitted_fact": omitted,
                        "source": row["source"],
                        "before": before,
                        "after": after,
                        "restoration_gain": after - before,
                    }
                )
    (out / "interventions.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in interventions), encoding="utf-8"
    )
    training = [
        (c["query"], r["text"], r["score"], int(bool(r["facts"])))
        for c in cases
        if c["split"] == "train"
        for r in prepared[c["id"]]
        if not r["stale"]
    ]
    models, traces = {}, {}
    started = time.perf_counter()
    for seed in [7, 19, 43]:
        model, trace = fit_selector(training, seed=seed)
        models[seed] = model
        traces[seed] = trace
        model.save(out / f"selector-seed-{seed}.json")
    gains = {}
    for row in interventions:
        key = (row["case"], row["source"])
        gains[key] = max(gains.get(key, 0), row["restoration_gain"])
    intervention_examples = [
        (c["query"], r["text"], r["score"], gains[(c["id"], r["source"])])
        for c in cases
        if c["split"] == "train"
        for r in prepared[c["id"]]
        if not r["stale"]
    ]
    intervention_model, traces["restoration"] = fit_selector(intervention_examples, seed=7)
    intervention_model.save(out / "selector-restoration.json")
    training_seconds = time.perf_counter() - started
    # Select regularization-free inference coverage weight on validation only.
    grid = [0.0, 0.2, 0.5]
    validation = []
    for weight in grid:
        metrics = []
        for c in cases:
            if c["split"] != "validation":
                continue
            result = select_evidence(
                Rows(prepared[c["id"]]),
                c["query"],
                budget=c["budget"],
                revisions={},
                reranker=models[7].score,
                ambiguity_gap=1,
                coverage_weight=weight,
            )
            facts = {f for r in prepared[c["id"]] if r["source"] in result.selected for f in r["facts"]}
            metrics.append(float(set(c["required_facts"]) <= facts))
        validation.append({"coverage_weight": weight, "fact_coverage_success": float(np.mean(metrics))})
    chosen_weight = max(validation, key=lambda x: x["fact_coverage_success"])["coverage_weight"]
    records = []
    tracemalloc.start()
    for case in cases:
        if case["split"] not in {"test", "ood"}:
            continue
        for method in [
            "exact_dedup",
            "bm25",
            "binary",
            "binary_no_freshness",
            "coverage",
            "learned7",
            "learned19",
            "learned43",
            "learned_coverage",
            "required_guard",
            "learned_restoration",
        ]:
            rows = [dict(r) for r in prepared[case["id"]]]
            for position, row in enumerate(rows):
                if method == "exact_dedup":
                    row["score"] = 1 - position / len(rows)
                if method == "bm25":
                    row["score"] = row["bm25"]
                if method == "binary_no_freshness":
                    row["freshness"] = "unchecked"
            model_seed = int(method[7:]) if method in {"learned7", "learned19", "learned43"} else 7
            reranker = (
                models[model_seed].score
                if method.startswith("learned") or method == "required_guard"
                else None
            )
            if method == "learned_restoration":
                reranker = intervention_model.score
            coverage = chosen_weight if method in {"coverage", "learned_coverage", "required_guard"} else 0
            # Explicit requirement control uses trusted fixture source IDs, not learned discovery.
            required = ["document-01", "document-03"] if method == "required_guard" else []
            begin = time.perf_counter()
            result = select_evidence(
                Rows(rows),
                case["query"],
                budget=case["budget"],
                revisions={},
                required=required,
                reranker=reranker,
                ambiguity_gap=1,
                coverage_weight=coverage,
                diversity_weight=0.1 if coverage else 0,
            )
            elapsed = (time.perf_counter() - begin) * 1000
            selected = [r for r in rows if r["source"] in result.selected]
            facts = {f for r in selected if not r["stale"] for f in r["facts"]}
            records.append(
                {
                    "case": case["id"],
                    "split": case["split"],
                    "domain": case["domain"],
                    "scenario": case["scenario"],
                    "method": method,
                    "success": int(set(case["required_facts"]) <= facts),
                    "stale_selected": int(any(r["stale"] for r in selected)),
                    "omitted_fact_count": len(set(case["required_facts"]) - facts),
                    "injection_selected": int(any("send credentials" in r["text"] for r in selected)),
                    "abstained": int(result.status == "insufficient_evidence"),
                    "bytes": result.units,
                    "selection_ms": elapsed,
                    "selected": result.selected,
                    "decisions": result.decisions,
                }
            )
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    summary = []
    rng = np.random.default_rng(20260919)
    for split in ["test", "ood"]:
        for method in sorted({r["method"] for r in records}):
            subset = [r for r in records if r["split"] == split and r["method"] == method]
            values = np.array([r["success"] for r in subset])
            boot = rng.choice(values, (2000, len(values)), replace=True).mean(axis=1)
            summary.append(
                {
                    "split": split,
                    "method": method,
                    "n": len(values),
                    "success": float(values.mean()),
                    "success_ci95": np.quantile(boot, [0.025, 0.975]).tolist(),
                    **{
                        key: float(np.mean([r[key] for r in subset]))
                        for key in [
                            "stale_selected",
                            "injection_selected",
                            "omitted_fact_count",
                            "abstained",
                            "bytes",
                            "selection_ms",
                        ]
                    },
                }
            )
    source_files = [
        Path(name)
        for name in [
            "stamps.py",
            "context_stamps/selection.py",
            "context_stamps/memory.py",
            "context_stamps/security.py",
            "context_stamps/baselines.py",
            "experiments/fixtures.py",
            "experiments/run_evidence.py",
        ]
    ]
    manifest = {
        "scope": "synthetic evidence coverage, not generated answer correctness or prompt-injection resistance",
        "python": platform.python_version(),
        "numpy": np.__version__,
        "platform": platform.system(),
        "data_sha256": hashlib.sha256(fixture.encode()).hexdigest(),
        "source_sha256": {p.as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_files},
        "split_counts": {
            s: sum(c["split"] == s for c in cases) for s in ["train", "validation", "test", "ood"]
        },
        "training_examples": len(training),
        "training_seconds": training_seconds,
        "python_peak_traced_bytes_evaluation": peak,
        "seeds": [7, 19, 43],
        "epochs": 400,
        "validation": validation,
        "selected_coverage_weight": chosen_weight,
        "license": "MIT",
        "data_origin": "original fictional procedural fixtures",
    }
    save(out / "manifest.json", manifest)
    save(out / "training.json", traces)
    save(out / "summary.json", summary)
    (out / "per-case.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")
    print(json.dumps({"output": str(out), "training_examples": len(training), "summary": summary}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("evidence/synthetic-v1"))
    run(parser.parse_args().out)

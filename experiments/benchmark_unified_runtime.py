"""Local reader comparison: full scope, dependency packet, verified exact reuse.

Fictional research configurations. No private documents or model weights published.
"""

import hashlib
import json
import random
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from benchmark_computation import call  # noqa: E402
from run_hybrid_retrieval import save, sha  # noqa: E402

from context_stamps import (  # noqa: E402
    ContextExpert,
    ContextNode,
    ContextRuntime,
    RuntimeBudget,
    Verification,
)

OUT = ROOT / "evidence/runtime-v1"
MODELS = {
    "qwen2.5:1.5b": "65ec06548149b04c096a120e4a6da9d4017ea809c91734ea5631e89f96ddc57b",
    "cortex-harness:1.7b-v11": "f9489974d20ea63778fb39591547611ce614e7456c18167f59ad4383351946c6",
}
OPTIONS = dict(temperature=0, seed=20260926, num_ctx=4096, num_predict=80)


def fixture(task, event):
    batch, accumulation = 8 + 2 * task + (2 if event >= 2 else 0), 2 + task % 3
    nodes = [ContextNode("result", "Target experiment depends on config and schedule.", "v1", frozenset({"reader"})),
             ContextNode("config", f"Target experiment training_batch_size = {batch}.", "v2" if event >= 2 else "v1", frozenset({"reader"})),
             ContextNode("schedule", f"Target experiment gradient_accumulation_steps = {accumulation}.", "v1", frozenset({"reader"}))]
    for i in range(10):
        nodes.append(ContextNode(f"unrelated-{i}", f"Unrelated experiment {i}: training_batch_size = {64 + i}; gradient_accumulation_steps = 7. "
             + "This note concerns a separate project and records its optimizer settings and validation procedure. " * 3,
             "v1", frozenset({"reader"})))
    return nodes, dict(batch_size=batch, accumulation_steps=accumulation, effective_batch_size=batch * accumulation)


def main():
    tags = {r["name"]: r for r in call("tags")["models"]}
    for name, digest in MODELS.items():
        if name not in tags or tags[name]["digest"] != digest or tags[name].get("remote_host"):
            raise ValueError("required pinned local reader unavailable")
    save(OUT / "workflow-protocol.json", dict(models=MODELS, options=OPTIONS, tasks=12, events=["initial", "repeat", "source_edit", "repeat"],
        modes=["full_scope", "prepared_context", "verified_reuse"], rounds=1,
        timing="Whole request including packet preparation, model HTTP or verified cache lookup. Serialized warm local models, randomized mode order. Server prefix-cache remains enabled. No hard callback cancellation.",
        quality="Host exact JSON verification of three integers including multiplication; every failure and model call retained. Source edits rebind dependencies before reuse.",
        limits="Fictional narrow scalar/arithmetic tasks; 50% repeated requests by design; one timing run; not open-ended coding, agent planning or production throughput."))
    records, summaries = [], {}
    rng = random.Random(20260926)  # nosec B311 - order only
    prompt = ('For the target experiment only, return a JSON object with integer batch_size, accumulation_steps, '
              'and effective_batch_size (batch_size multiplied by accumulation_steps). Read config and schedule; ignore unrelated projects.')
    for model, digest in MODELS.items():
        call("generate", dict(model=model, prompt='Return {"ready":true}.', stream=False, format="json", think=False, options=OPTIONS, keep_alive="5m"))
        modes = ["full_scope", "prepared_context", "verified_reuse"]
        rng.shuffle(modes)
        for mode in modes:
            for task in range(12):
                runtime = ContextRuntime(tenant="fixture-lab", principal="reader", role="reader", policy="fixture-v1")
                prior = None
                for event in range(4):
                    nodes, expected = fixture(task, event)
                    if prior is None:
                        for node in nodes:
                            runtime.put(node)
                        runtime.link("result", "schedule", "depends_on", provenance="fixture")
                    if event in (0, 2):
                        runtime.put(nodes[1])
                        runtime.link("result", "config", "depends_on", provenance="fixture")
                    prior = nodes
                    elapsed_start = time.perf_counter()
                    expert = ContextExpert("fixture-exact", lambda r: ["result"], 0)
                    if mode == "full_scope":
                        expert = ContextExpert("fixture-exact", lambda r: list(r.eligible), 0)
                    prepared = runtime.prepare_context(prompt, eligible=[n.key for n in nodes], experts=[expert], baseline=expert.name,
                        scope="fixture-v1", limit=13 if mode == "full_scope" else 1,
                        verifier=lambda p: Verification({"config", "schedule"} <= set(p.sources)), budget=RuntimeBudget(bytes=8192, milliseconds=1000))
                    if prepared.status != "complete":
                        raise AssertionError(prepared)
                    stats = dict(prompt_tokens=0, output_tokens=0, model_calls=0)
                    observation = dict(actual=None, exact_schema=False)
                    def compute(text):
                        response = call("generate", dict(model=model, prompt=text + "\n\n" + prompt, stream=False,
                            format="json", think=False, options=OPTIONS, keep_alive="5m"))
                        stats.update(prompt_tokens=response.get("prompt_eval_count", 0), output_tokens=response.get("eval_count", 0), model_calls=1)
                        return response["response"]
                    def verify(value, text):
                        observation["response_sha256"] = hashlib.sha256(value.encode()).hexdigest()
                        try:
                            actual = json.loads(value)
                            valid = isinstance(actual, dict) and set(actual) == set(expected) and all(type(actual[k]) is int for k in expected)
                            observation["exact_schema"] = valid
                            observation["actual"] = actual if valid else None
                            return valid and actual == expected
                        except (TypeError, ValueError):
                            return False
                    if mode == "verified_reuse":
                        request_digest = hashlib.sha256(json.dumps(dict(prompt=prompt, options=OPTIONS, task=task), sort_keys=True).encode()).hexdigest()
                        result = runtime.run_verified(prepared, request_digest=request_digest, model=digest, prompt="three-integers-v1", tool="reader",
                            verifier_revision="exact-integers-v1", compute=compute, verify=verify)
                        correct, reused = result["status"] == "verified", result["reused"]
                    else:
                        value = compute(prepared.text)
                        correct, reused = verify(value, prepared.text), False
                    records.append(dict(model=model, mode=mode, task=task, event=event, correct=correct, reused=reused,
                        context_bytes=len(prepared.text.encode()), sources=len(prepared.sources), **stats, **observation, expected=expected,
                        elapsed_ms=(time.perf_counter() - elapsed_start) * 1000))
            selected = [r for r in records if r["model"] == model and r["mode"] == mode]
            times = sorted(r["elapsed_ms"] for r in selected)
            summaries.setdefault(model, {})[mode] = dict(requests=len(selected), verified=sum(r["correct"] for r in selected),
                calls=sum(r["model_calls"] for r in selected), prompt_tokens=sum(r["prompt_tokens"] for r in selected),
                output_tokens=sum(r["output_tokens"] for r in selected), total_ms=sum(times), p50_ms=statistics.median(times), p95_ms=times[int(.95 * (len(times) - 1))])
            save(OUT / "workflow-observations.json", records)
            save(OUT / "workflow-summary.json", summaries)
            print(model, mode, json.dumps(summaries[model][mode]), flush=True)
        call("generate", dict(model=model, keep_alive=0))
    save(OUT / "workflow-manifest.json", dict(ollama=call("version"),
        models={n: {"digest": tags[n]["digest"], "details": tags[n]["details"]} for n in MODELS},
        source_sha256={p: sha(ROOT / p) for p in ("experiments/benchmark_unified_runtime.py", "context_stamps/runtime.py")}))


if __name__ == "__main__":
    main()

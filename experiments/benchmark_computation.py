"""Local Qwen pilot: exact computation reuse, source edits and policy changes.

Copyright (c) 2026 Prashant Jagtap. MIT License.
Fictional experiment settings; no private data or remote model calls.
"""

import hashlib
import json
import random
import statistics
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from context_stamps.computation import ComputationCache, ComputationIdentity  # noqa: E402

OUT = ROOT / "evidence/computation-v1"
MODEL = "qwen2.5:1.5b"
OPTIONS = dict(temperature=0, seed=20260926, num_ctx=2048, num_predict=40)


def digest(value):
    return hashlib.sha256(value.encode()).hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
                    encoding="utf-8", newline="\n")


def call(path, payload=None):
    request = urllib.request.Request("http://127.0.0.1:11434/api/" + path,
        data=None if payload is None else json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=120) as response:  # nosec B310 - fixed loopback URL
        return json.load(response)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    available = call("tags")["models"]
    identity = next(m for m in available if m["name"] == MODEL)
    if identity["digest"] != "65ec06548149b04c096a120e4a6da9d4017ea809c91734ea5631e89f96ddc57b":
        raise ValueError("local model revision differs from frozen pilot")
    protocol = dict(model=MODEL, model_digest=identity["digest"], options=OPTIONS,
        tasks=12, events_per_task=6, repetitions=3, modes=["always_call", "exact_reuse"],
        sequence=["initial", "repeat", "source_edit", "repeat", "policy_edit", "repeat"],
        timing="Warm local service, serialized requests, shuffled mode order per repetition; normal server prefix-cache behavior. End-to-end HTTP/cache lookup, excludes model cold load.",
        limits="12 fictional scalar-extraction tasks, each with50% identical computations. Repetitions measure timing, not independent quality samples. No base-model fine-tuning, ranking, open-ended coding, cross-agent network or production-scale conclusion.")
    save(OUT / "protocol.json", protocol)
    call("generate", dict(model=MODEL, prompt='Return {"value":1}.', stream=False,
                           format="json", options=OPTIONS, keep_alive="5m"))
    rows = []
    randomizer = random.Random(20260926)  # nosec B311 - experimental ordering only
    for repetition in range(3):
        modes = list(protocol["modes"])
        randomizer.shuffle(modes)
        for mode in modes:
            cache = ComputationCache(capacity=128, ttl_seconds=300)
            for task in range(12):
                for event in range(6):
                    value = 100 + task * 7 + (1 if event >= 2 else 0)
                    policy = "policy-v2" if event >= 4 else "policy-v1"
                    source = f"Experiment {task} configuration: training_batch_size = {value}."
                    prompt = source + '\nReturn only a JSON object with the integer training_batch_size under key "value".'
                    payload = dict(model=MODEL, prompt=prompt, stream=False, format="json", options=OPTIONS, keep_alive="5m")
                    start = time.perf_counter()
                    binding = ComputationIdentity("fictional-lab", "researcher", identity["digest"],
                        "scalar-extraction-v1", "none", policy,
                        digest(json.dumps(payload, sort_keys=True, separators=(",", ":"))),
                        (("experiment-" + str(task), "r1", digest(source)),))
                    result = cache.get(binding) if mode == "exact_reuse" else None
                    hit = result is not None
                    prompt_tokens = output_tokens = 0
                    if result is None:
                        response = call("generate", payload)
                        result = response["response"]
                        prompt_tokens = response.get("prompt_eval_count", 0)
                        output_tokens = response.get("eval_count", 0)
                    try:
                        parsed = json.loads(result)
                        correct = set(parsed) == {"value"} and type(parsed["value"]) is int and parsed["value"] == value
                    except (ValueError, TypeError):
                        correct = False
                    if mode == "exact_reuse" and not hit and correct:
                        cache.put(binding, result)
                    rows.append(dict(repetition=repetition, mode=mode, task=task, event=event,
                        correct=correct, hit=hit, prompt_tokens=prompt_tokens, output_tokens=output_tokens,
                        milliseconds=(time.perf_counter() - start) * 1000,
                        result=result, expected=value, policy=policy))
            print("completed", repetition, mode, flush=True)
    summary = {}
    for mode in protocol["modes"]:
        items = [r for r in rows if r["mode"] == mode]
        times = sorted(r["milliseconds"] for r in items)
        summary[mode] = dict(requests=len(items), correct=sum(r["correct"] for r in items),
            hits=sum(r["hit"] for r in items), model_calls=sum(not r["hit"] for r in items),
            prompt_tokens=sum(r["prompt_tokens"] for r in items), output_tokens=sum(r["output_tokens"] for r in items),
            total_ms=sum(times), p50_ms=statistics.median(times), p95_ms=times[int(.95 * (len(times) - 1))])
    save(OUT / "results.json", rows)
    save(OUT / "summary.json", summary)
    save(OUT / "manifest.json", dict(protocol_sha256=hashlib.sha256((OUT / "protocol.json").read_bytes()).hexdigest(),
        ollama_version=call("version"), source_sha256={p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest()
        for p in ("experiments/benchmark_computation.py", "context_stamps/computation.py")}))
    save(OUT / "checksums.json", {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                 for p in OUT.glob("*.json") if p.name != "checksums.json"})
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()

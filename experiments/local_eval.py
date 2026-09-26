"""Native Inspect ARC plumbing pilot. Local Ollama only; not a runtime benefit test.

Prepare records exact sample/model identities before any generation. Run consumes
that plan without changing the native solver or scorer. Raw Inspect logs stay in
the chosen output directory and may contain licensed dataset text.
"""

import argparse
import hashlib
import importlib.metadata
import inspect
import json
import os
import platform
import re
import time
import urllib.request
from pathlib import Path

BASE_URL = "http://127.0.0.1:11434"
SCHEMA = 1
PINNED = {"inspect-ai": "0.3.269", "inspect-evals": "0.22.0", "openai": "3.19.2"}
DATASET_REVISION = "210d026faf9955653af8916fad021475a3f00453"
CONFIG = {"temperature": 0, "seed": 7, "max_tokens": 64, "max_connections": 1,
          "cache": False, "max_retries": 0, "timeout": 120}


def digest(data):
    return hashlib.sha256(data).hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def local_api(path, body=None):
    # Neither host environment proxies nor redirects may reroute local preflight.
    class NoRedirect(urllib.request.HTTPRedirectHandler):
        def redirect_request(self, req, fp, code, msg, headers, newurl):
            raise ValueError("Local endpoint redirect refused")

    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect())
    request = urllib.request.Request(
        BASE_URL + path, data=None if body is None else canonical(body),
        headers={"Content-Type": "application/json"},
    )
    with opener.open(request, timeout=15) as response:  # fixed loopback origin
        data = response.read(4 * 1024 * 1024 + 1)
    if len(data) > 4 * 1024 * 1024:
        raise ValueError("Oversized local response")
    return json.loads(data)


def validate_model(name, tags, details):
    if not re.fullmatch(r"[a-zA-Z0-9_.:/-]{1,160}", name) or "cloud" in name.lower():
        raise ValueError("A local weight-backed model is required")
    matches = [m for m in tags["models"] if m["name"] == name]
    if len(matches) != 1:
        raise ValueError("Exact installed model name required; no automatic download")
    row = matches[0]
    if any(obj.get(field) for obj in (row, details) for field in ("remote_host", "remote_model")):
        raise ValueError("Remote-backed Ollama model refused")
    if row.get("size", 0) <= 0 or details.get("details", {}).get("format") != "gguf":
        raise ValueError("Local GGUF weights required")
    if not re.fullmatch(r"[0-9a-f]{64}", row.get("digest", "")):
        raise ValueError("Model digest missing")
    return {
        "name": name, "digest": row["digest"], "bytes": row["size"],
        "details": details["details"],
        "template_sha256": digest(details.get("template", "").encode()),
        "parameters_sha256": digest(details.get("parameters", "").encode()),
        "license_sha256": digest(details.get("license", "").encode()),
        "system_sha256": digest(details.get("system", "").encode()),
    }


def model_identity(name):
    return validate_model(name, local_api("/api/tags"), local_api("/api/show", {"model": name}))


def versions():
    values = {d.metadata["Name"]: d.version for d in importlib.metadata.distributions()}
    for package, required in PINNED.items():
        if importlib.metadata.version(package) != required:
            raise ValueError(f"Expected {package}=={required}; register a new protocol for changes")
    return dict(sorted(values.items()))


def native_task():
    from inspect_evals.arc import arc_challenge
    from inspect_evals.arc.arc import ARC_DATASET_REVISION

    if ARC_DATASET_REVISION != DATASET_REVISION:
        raise ValueError("Native dataset revision changed")
    return arc_challenge()


def task_sources():
    from inspect_ai.scorer import choice
    from inspect_ai.solver import multiple_choice
    from inspect_evals.arc import arc_challenge

    return {obj.__module__: digest(Path(inspect.getfile(inspect.unwrap(obj))).read_bytes())
            for obj in (choice, multiple_choice, arc_challenge)}


def sample_record(sample):
    return {"id": sample.id, "sha256": digest(canonical({
        "input": sample.input, "choices": sample.choices, "target": sample.target,
    }))}


def write_new(path, value):
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def prepare(directory, model, limit):
    if not 1 <= limit <= 20:
        raise ValueError("This plumbing pilot is limited to 1..20 samples")
    installed = versions()
    identity = model_identity(model)
    task = native_task()
    plan = {
        "schema": SCHEMA, "purpose": "native ARC-Challenge plumbing pilot; no runtime treatment",
        "model": identity, "ollama_version": local_api("/api/version")["version"],
        "dataset": {"name": "allenai/ai2_arc", "subset": "ARC-Challenge", "split": "test",
                    "revision": DATASET_REVISION, "license": "CC-BY-SA-4.0"},
        "samples": [sample_record(s) for s in list(task.dataset)[:limit]],
        "selection": "First N in pinned upstream order; development smoke, excluded from final claims",
        "config": CONFIG,
        "task_source_hashes": task_sources(), "packages": installed,
        "runner_sha256": digest(Path(__file__).read_bytes()),
        "python": platform.python_version(), "platform": platform.platform(),
        "spending": "Local weights only. No hosted model/judge calls.",
    }
    directory.mkdir(parents=True, exist_ok=True)
    write_new(directory / "plan.json", plan)
    print(json.dumps({"prepared": len(plan["samples"]), "model": model, "generation_calls": 0}))


def validate_plan(plan):
    if plan["schema"] != SCHEMA or plan["runner_sha256"] != digest(Path(__file__).read_bytes()):
        raise ValueError("Prepared runner/schema differs")
    if plan["config"] != CONFIG:
        raise ValueError("Pilot config is fixed; provider/fallback overrides refused")
    if not 1 <= len(plan["samples"]) <= 20:
        raise ValueError("Pilot sample limit changed")
    if len({s["id"] for s in plan["samples"]}) != len(plan["samples"]):
        raise ValueError("Duplicate sample IDs")


def run(directory):
    from inspect_ai import eval as inspect_eval
    from inspect_ai.log import read_eval_log
    from inspect_ai.model import get_model

    raw_plan = (directory / "plan.json").read_bytes()
    plan = json.loads(raw_plan)
    validate_plan(plan)
    if versions() != plan["packages"] or task_sources() != plan["task_source_hashes"]:
        raise ValueError("Prepared evaluation environment differs")
    model = plan["model"]["name"]
    if model_identity(model) != plan["model"]:
        raise ValueError("Prepared local model changed")
    if local_api("/api/version")["version"] != plan["ollama_version"]:
        raise ValueError("Ollama version changed")
    task = native_task()
    samples = list(task.dataset)[:len(plan["samples"])]
    if [sample_record(s) for s in samples] != plan["samples"]:
        raise ValueError("Prepared task identities/content changed")
    # A directory is consumed once, even if execution fails; preserve that failure.
    write_new(directory / "started.json", {"plan_sha256": digest(raw_plan), "utc": time.time()})
    for key in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"):
        os.environ.pop(key, None)
    os.environ["NO_PROXY"] = "127.0.0.1,localhost"
    local_model = get_model("ollama/" + model, base_url=BASE_URL + "/v1", api_key="ollama")
    start = time.perf_counter()
    result = inspect_eval(
        task, model=local_model, sample_id=[s.id for s in samples],
        log_dir=str(directory / "raw-logs"), display="plain", max_samples=1,
        fail_on_error=False, retry_on_error=0, time_limit=180, **plan["config"],
    )[0]
    elapsed = time.perf_counter() - start
    log = read_eval_log(result.location)
    unchanged = model_identity(model) == plan["model"]
    rows = []
    for sample in log.samples or []:
        rows.append({
            "id": sample.id, "epoch": sample.epoch,
            "scores": {k: v.value for k, v in (sample.scores or {}).items()},
            "answer": {k: v.answer for k, v in (sample.scores or {}).items()},
            "output_sha256": digest(sample.output.completion.encode()),
            "stop_reasons": [c.stop_reason for c in sample.output.choices],
            "usage": {k: v.model_dump(mode="json") for k, v in sample.model_usage.items()},
            "total_seconds": sample.total_time, "working_seconds": sample.working_time,
            "error": sample.error.message if sample.error else None,
            "limit": sample.limit.model_dump(mode="json") if sample.limit else None,
        })
    report = {
        "schema": SCHEMA, "plan_sha256": digest(raw_plan), "status": log.status,
        "model_unchanged": unchanged, "sample_count": len(rows), "rows": rows,
        "native_results": log.results.model_dump(mode="json") if log.results else None,
        "stats": log.stats.model_dump(mode="json"), "eval_call_seconds": elapsed,
        "raw_log_sha256": digest(Path(result.location).read_bytes()),
        "ttft_seconds": None, "peak_vram_bytes": None, "peak_ram_bytes": None,
        "provider_charge_usd": 0, "electricity_cost_usd": None,
        "limitations": [
            "Small development smoke; no benchmark ranking or runtime gain claim.",
            "Eval wall time includes first-use model load; dataset preparation excluded.",
            "Inspect cache disabled; Ollama internal prefix cache not independently disabled.",
            "No streaming TTFT, resource telemetry, concurrency sweep or confidence claim.",
            "No answer-based changes made to the native solver/scorer.",
        ],
    }
    write_new(directory / "summary.json", report)
    print(json.dumps({"status": log.status, "samples": len(rows), "model_unchanged": unchanged,
                      "scores": report["native_results"], "seconds": elapsed}, indent=2))
    if log.status != "success" or not unchanged or len(rows) != len(samples):
        raise RuntimeError("Pilot incomplete; preserve logs and investigate")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "run"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="qwen2.5:1.5b")
    parser.add_argument("--limit", type=int, default=10)
    args = parser.parse_args()
    if args.action == "prepare":
        prepare(args.output, args.model, args.limit)
    else:
        run(args.output)

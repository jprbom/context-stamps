"""Separate schema/serialization behavior from agent task competence.

Only generates JSON on fixed loopback. No generated command is executed.
"""

import argparse
import hashlib
import json
import time
import urllib.request
from pathlib import Path

import terminal_tools
from local_eval import BASE_URL, local_api, model_identity
from terminal_pilot import render, sha, token_count, write_new

ROOT = Path(__file__).resolve().parents[1]
CONFIG = {"temperature": 0, "seed": 7, "num_ctx": 4096, "num_predict": 128}
TARGETS = (
    {"tool": "shell", "command": "echo 42"},
    {"tool": "read_file", "path": "notes.txt"},
    {"tool": "write_file", "path": "notes.txt", "content": "sample\n"},
)
MODES = ("sorted_union", "ordered_union", "json_only", "sorted_single_variant")


def request_body(model, target, mode):
    prompt = render([{"role": "system", "content": terminal_tools.SYSTEM},
                     {"role": "user", "content": "This is a formatting check, not task execution. Return exactly this "
                      "action as a JSON object; do not call finish: " + json.dumps(target, ensure_ascii=False)}])
    if mode == "json_only":
        schema = "json"
    elif mode == "sorted_single_variant":
        schema = next(s for s in terminal_tools.SCHEMA["oneOf"] if s["properties"]["tool"]["enum"] == [target["tool"]])
    else:
        schema = terminal_tools.SCHEMA
    body = {"model": model, "prompt": prompt, "raw": True, "stream": False, "keep_alive": "5m",
            "options": CONFIG, "format": schema}
    raw = json.dumps(body, sort_keys=mode.startswith("sorted_"), ensure_ascii=False, separators=(",", ":")).encode()
    return prompt, raw


def call(raw):
    class NoRedirect(urllib.request.HTTPRedirectHandler):
        def redirect_request(self, *args, **kwargs):
            raise ValueError("Provider redirect refused")
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect())
    request = urllib.request.Request(BASE_URL + "/api/generate", data=raw, headers={"Content-Type": "application/json"})
    with opener.open(request, timeout=120) as response:
        data = response.read(1024 * 1024 + 1)
    if len(data) > 1024 * 1024:
        raise ValueError("Oversized provider response")
    return json.loads(data)


def run(args):
    args.out.mkdir(parents=True, exist_ok=False)
    model = model_identity(args.model)
    files = ("experiments/terminal_schema_probe.py", "experiments/terminal_tools.py",
             "experiments/terminal_pilot.py", "experiments/qwen_token_count.py", "experiments/local_eval.py")
    count = token_count("", args.token_python, args.tokenizer)
    count.pop("tokens")
    plan = {"schema": 1, "model": model, "ollama": local_api("/api/version"), "config": CONFIG,
            "modes": MODES, "targets": TARGETS, "tokenizer": count,
            "source_hashes": {n: sha(ROOT / n) for n in files},
            "generated_commands_executed": 0, "purpose": "Schema serialization canary, not agent accuracy or training"}
    write_new(args.out / "plan.json", plan)
    rows = []
    for index, target in enumerate(TARGETS):
        for mode in MODES:
            prompt, raw = request_body(args.model, target, mode)
            counted = token_count(prompt, args.token_python, args.tokenizer)
            row = {"target": index, "mode": mode, "request_sha256": hashlib.sha256(raw).hexdigest(),
                   "input_tokens": counted["tokens"], "started_utc": time.time()}
            started = time.perf_counter()
            try:
                response = call(raw)
                row["response"] = response
                if response.get("prompt_eval_count") != counted["tokens"] or not response.get("done"):
                    raise ValueError("Token parity or response completion failed")
                row["matches_target"] = json.loads(response["response"]) == target
            except Exception as error:
                row["error"] = type(error).__name__ + ": " + str(error)[-1000:]
                row["matches_target"] = False
            row["elapsed_seconds"] = time.perf_counter() - started
            write_new(args.out / (str(index) + "-" + mode + ".json"), row)
            rows.append(row)
    unchanged = model == model_identity(args.model)
    summary = {"matches": {m: sum(r["matches_target"] for r in rows if r["mode"] == m) for m in MODES},
               "calls": len(rows), "model_unchanged": unchanged, "commands_executed": 0,
               "provider_charges_usd": 0}
    write_new(args.out / "summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=('qwen2.5:1.5b', 'qwen2.5-coder:7b'), required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--token-python', type=Path, required=True)
    parser.add_argument('--tokenizer', type=Path, required=True)
    run(parser.parse_args())

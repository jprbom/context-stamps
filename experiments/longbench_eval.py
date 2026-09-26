"""Local LongBench v2 development pilot; native prompts/parser, explicit deviations.

No Inspect dependency, remote provider, automatic model pull, tools or judge.
The upstream MIT prompt/parser attribution is in evidence/longbench-v2-pilot.
Full-context, BM25 and no-context controls are not Context Stamps treatments.
"""

import argparse
import collections
import importlib.metadata
import json
import math
import platform
import random
import re
import statistics
import subprocess
import threading
import time
import urllib.request
from pathlib import Path

import local_eval
from local_eval import canonical, digest, local_api, model_identity, write_new

ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = ROOT / "evidence/longbench-v2-pilot/upstream"
DATA_REV = "2b48e494f2c7a2f0af81aae178e05c7e1dde0fe9"
DATA_SHA = "15d61c22d92c96900b3c4948b6aeea218d3214b676a65df48e7b8555604c7fe2"
TOKEN_REV = "989aa7980e4cf806f80c7fef2b1adb7bc71aa306"
CODE_REV = "ef5ccc4bdcb1d505455517e4b419a50bb959862a"
MODEL = "qwen2.5:1.5b"
MODEL_DIGEST = "65ec06548149b04c096a120e4a6da9d4017ea809c91734ea5631e89f96ddc57b"
METHODS = ("full", "bm25", "no_context")
CONFIG = dict(temperature=.1, seed=7, num_ctx=32768, num_predict=128)
MAX_INPUT = 28000
LENGTH_AUDIT_SHA = "82219d735da8bb567a06fd0dc37354be597d9072cbad1bd4be0d9ae5ad1f79b3"
RETRIEVAL_INPUT = 4096
PROMPTS = {
    "full": ("0shot.txt", "68a162252bc9ff71d5d7abca3d69bb31aac3c35f832d657a2866f2018b8a6950"),
    "bm25": ("0shot_rag.txt", "5c50ce4fa24d297b8c72eb048599681ee01ced5d24a6636464386d6d3c72c4c9"),
    "no_context": ("0shot_no_context.txt", "125fb44183af01ffc623df19392b80a287db62af3e2a2805e5b2ff9c7b2bee30"),
}


def file_hash(path):
    import hashlib

    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def sources():
    return {str(p.relative_to(ROOT)).replace("\\", "/"): file_hash(p)
            for p in (Path(__file__).resolve(), Path(local_eval.__file__).resolve())}


def templates():
    result = {}
    for method, (name, expected) in PROMPTS.items():
        raw = (UPSTREAM / name).read_bytes()
        if digest(raw) != expected:
            raise ValueError("Upstream prompt changed")
        result[method] = raw.decode("utf-8")
    return result


def extract_answer(response):
    # Exact upstream pred.py parser, including case sensitivity and first match.
    response = response.replace('*', '')
    match = re.search(r'The correct answer is \(([A-D])\)', response)
    if match:
        return match.group(1)
    match = re.search(r'The correct answer is ([A-D])', response)
    return match.group(1) if match else None


def public_input(item):
    """Whitelist input fields; target and grader metadata cannot enter prompts."""
    return {key: item[key] for key in ("context", "question", "choice_A", "choice_B", "choice_C", "choice_D")}


def render(item, template, context):
    values = {"$DOC$": context.strip(), "$Q$": item["question"].strip()}
    values.update({f"$C_{letter}$": item[f"choice_{letter}"].strip() for letter in "ABCD"})
    # One substitution pass prevents placeholders inside evidence being expanded.
    prompt = re.sub(r"\$DOC\$|\$Q\$|\$C_[ABCD]\$", lambda m: values[m.group()], template)
    return "<|im_start|>user\n" + prompt + "<|im_end|>\n<|im_start|>assistant\n"


def token_count(tokenizer, text):
    return len(tokenizer.encode(text, add_special_tokens=False).ids)


def chunks(tokenizer, text):
    offsets = tokenizer.encode(text, add_special_tokens=False).offsets
    result = []
    for begin in range(0, len(offsets), 448):
        end = min(begin + 512, len(offsets))
        left, right = offsets[begin][0], offsets[end - 1][1]
        result.append((left, right, text[left:right]))
        if end == len(offsets):
            break
    return result


def bm25_order(query, pieces):
    def terms(text):
        return re.findall(r"\w+", text.casefold())
    documents = [collections.Counter(terms(piece[2])) for piece in pieces]
    query_terms = set(terms(query))
    if not documents:
        return []
    lengths = [sum(doc.values()) for doc in documents]
    average = sum(lengths) / len(lengths) or 1
    df = {term: sum(term in doc for doc in documents) for term in query_terms}
    scores = []
    for doc, length in zip(documents, lengths):
        score = 0.
        for term in sorted(query_terms):
            frequency = doc[term]
            idf = math.log(1 + (len(documents) - df[term] + .5) / (df[term] + .5))
            score += idf * frequency * 2.5 / (frequency + 1.5 * (.25 + .75 * length / average))
        scores.append(score)
    return sorted(range(len(pieces)), key=lambda i: (-scores[i], i))


def build_prompt(item, method, tokenizer, prompt_templates):
    item = public_input(item)
    selected, spans = [], []
    context = item["context"] if method == "full" else ""
    if method == "bm25":
        pieces = chunks(tokenizer, item["context"])
        query = "\n".join(item[k] for k in ("question", "choice_A", "choice_B", "choice_C", "choice_D"))
        def packet(indices):
            return "\n\n".join(f"Retrieved chunk {j+1}: {pieces[i][2]}" for j, i in enumerate(sorted(indices)))
        for index in bm25_order(query, pieces):
            candidate = packet(selected + [index])
            if token_count(tokenizer, render(item, prompt_templates[method], candidate)) <= RETRIEVAL_INPUT:
                selected.append(index)
        context = packet(selected)
        spans = [[pieces[i][0], pieces[i][1]] for i in sorted(selected)]
    prompt = render(item, prompt_templates[method], context)
    count = token_count(tokenizer, prompt)
    limit = RETRIEVAL_INPUT if method == "bm25" else MAX_INPUT
    if count > limit:
        raise ValueError("Complete input exceeds the frozen budget; no truncation allowed")
    return prompt, dict(prompt_sha256=digest(prompt.encode()), input_tokens=count,
                        selected_spans=spans, selected_chunks=len(selected))


def load_inputs(data, tokenizer_dir):
    from tokenizers import Tokenizer

    items = json.loads((data / "data.json").read_bytes())
    if not isinstance(items, list) or len(items) != 503 or len({r["_id"] for r in items}) != 503:
        raise ValueError("Expected pinned 503 unique tasks")
    for item in items:
        if item["answer"] not in "ABCD" or len(item["answer"]) != 1:
            raise ValueError("Invalid answer key")
        if not all(isinstance(v, str) for v in public_input(item).values()):
            raise ValueError("Invalid model input")
    tokenizer = Tokenizer.from_file(str(tokenizer_dir / "tokenizer.json"))
    tokenizer.no_truncation()
    tokenizer.no_padding()
    return items, tokenizer


def input_hashes(data, tokenizer_dir):
    return {"data.json": file_hash(data / "data.json"), "data_README.md": file_hash(data / "README.md"),
            **{name: file_hash(tokenizer_dir / name) for name in ("tokenizer.json", "tokenizer_config.json",
                                                                 "config.json", "LICENSE", "README.md")}}


def identity():
    value = model_identity(MODEL)
    if value["digest"] != MODEL_DIGEST:
        raise ValueError("This pilot is pinned to the existing local Qwen control")
    details = local_api("/api/show", {"model": MODEL})
    if details["model_info"].get("qwen2.context_length", 0) < CONFIG["num_ctx"]:
        raise ValueError("Model context capacity is insufficient")
    return value


def cached_lengths(path, hashes):
    if path.stat().st_size > 1024*1024 or file_hash(path) != LENGTH_AUDIT_SHA:
        raise ValueError("Length audit is not the immutable original registration")
    prior = json.loads(path.read_bytes())
    if (prior["input_hashes"] != hashes
            or prior["tokenizers_version"] != importlib.metadata.version("tokenizers")):
        raise ValueError("Length audit dataset/tokenizer changed")
    return [dict(row, eligible=row["full_input_tokens"] <= MAX_INPUT) for row in prior["eligibility"]]


def prepare(data, tokenizer_dir, output, length_audit=None):
    started = time.perf_counter()
    model = identity()
    hashes = input_hashes(data, tokenizer_dir)
    if hashes["data.json"] != DATA_SHA:
        raise ValueError("Dataset differs from pinned Hub LFS SHA256")
    items, tokenizer = load_inputs(data, tokenizer_dir)
    prompt_templates = templates()
    eligibility = cached_lengths(length_audit, hashes) if length_audit else []
    if not length_audit:
        for item in items:
            prompt = render(public_input(item), prompt_templates["full"], item["context"])
            count = token_count(tokenizer, prompt)
            eligibility.append(dict(id=item["_id"], domain=item["domain"], length=item["length"],
                                    difficulty=item["difficulty"], full_input_tokens=count, eligible=count <= MAX_INPUT))
    selected = []
    for domain in sorted({r["domain"] for r in eligibility}):
        candidates = [r for r in eligibility if r["domain"] == domain and r["eligible"]]
        candidates.sort(key=lambda r: digest(("longbench-pilot-7:" + r["id"]).encode()))
        selected.extend(candidates[:2])
    if not selected:
        raise ValueError("No hardware-supported tasks; cannot silently truncate")
    lookup = {r["_id"]: r for r in items}
    samples = []
    for selection in selected:
        item = lookup[selection["id"]]
        methods = list(METHODS)
        random.Random("order-7:" + item["_id"]).shuffle(methods)
        samples.append(dict(**selection, item_sha256=digest(canonical(item)), order=methods,
                            prompts={m: build_prompt(item, m, tokenizer, prompt_templates)[1] for m in methods}))
    plan = dict(schema=1, purpose="development reader/context controls; no runtime-benefit or ranking claim",
        model=model, ollama=local_api("/api/version"), source_hashes=sources(), input_hashes=hashes,
        dataset=dict(repo="zai-org/LongBench-v2", revision=DATA_REV, license="Apache-2.0", split="train"),
        tokenizer=dict(repo="Qwen/Qwen2.5-1.5B-Instruct", revision=TOKEN_REV), upstream_revision=CODE_REV,
        config=CONFIG, max_input=MAX_INPUT, retrieval_input=RETRIEVAL_INPUT,
        samples=samples, eligibility=eligibility, python=platform.python_version(), platform=platform.platform(),
        tokenizers_version=importlib.metadata.version("tokenizers"), preparation_seconds=time.perf_counter()-started,
        length_audit_sha256=file_hash(length_audit) if length_audit else None,
        selection="At most two per domain; SHA256(seed string + ID) order among full-input <=28000 tokens; no labels",
        promotion="None: development plumbing/controls only; retain every failure and malformed response",
        deviations=["Qwen raw ChatML, user only, no stored system message; exact server token parity required",
                    "32768 context capacity and <=28000 full input; no head/tail truncation",
                    "One seed/trial, shuffled treatment order within task; no CoT or retries",
                    "BM25 k1=1.5 b=.75, Unicode word terms, 512-token chunks/64 overlap, 4096 complete-input budget",
                    "Native prompts and exact parser; one-pass substitution preserves placeholders inside evidence",
                    "Corrected length-stratum aggregation; upstream result.py repeats short in its medium branch",
                    "No-context and retrieval are controls, not implemented runtime treatments"],
        spending="Only fixed loopback local GGUF; no paid inference or judges")
    output.mkdir(parents=True, exist_ok=True)
    write_new(output / "plan.json", plan)
    print(json.dumps(dict(selected=len(samples), eligible=sum(r["eligible"] for r in eligibility),
                          input_hashes=hashes, preparation_seconds=plan["preparation_seconds"])))


def generate(prompt):
    class NoRedirect(urllib.request.HTTPRedirectHandler):
        def redirect_request(self, req, fp, code, msg, headers, newurl):
            raise ValueError("Local inference redirect refused")

    body = dict(model=MODEL, prompt=prompt, raw=True, stream=False, options=CONFIG, keep_alive="5m")
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect())
    request = urllib.request.Request(local_eval.BASE_URL + "/api/generate", data=canonical(body),
                                     headers={"Content-Type": "application/json"})
    with opener.open(request, timeout=300) as response:
        raw = response.read(2 * 1024 * 1024 + 1)
    if len(raw) > 2 * 1024 * 1024:
        raise ValueError("Oversized local generation response")
    return json.loads(raw)


def monitor(stop, samples):
    while not stop.is_set():
        row = dict(utc=time.time())
        try:
            result = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,temperature.gpu",
                "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=5)
            row.update(sample=result.stdout.strip(), exit_code=result.returncode)
        except (OSError, subprocess.TimeoutExpired) as error:
            row["error"] = type(error).__name__
        samples.append(row)
        stop.wait(1.)


def summarize(plan, rows):
    summaries = []
    for method in METHODS:
        group = [r for r in rows if r["method"] == method]
        success = [r for r in group if r["status"] == "ok"]
        latencies = sorted(r["whole_request_seconds"] for r in success)
        summaries.append(dict(method=method, expected=len(plan["samples"]), recorded=len(group),
            correct=sum(r.get("correct", False) for r in group), denominator=len(plan["samples"]),
            accuracy=sum(r.get("correct", False) for r in group)/len(plan["samples"]),
            malformed=sum(r.get("prediction") is None for r in success), errors=len(group)-len(success),
            missing=len(plan["samples"])-len(group),
            input_tokens=sum(r["usage"]["prompt_eval_count"] for r in success),
            output_tokens=sum(r["usage"]["eval_count"] for r in success),
            p50_seconds=statistics.median(latencies) if latencies else None,
            p95_seconds=latencies[math.ceil(.95*len(latencies))-1] if latencies else None,
            total_request_seconds=sum(latencies)))
    return summaries


def run(data, tokenizer_dir, output):
    raw_plan = (output / "plan.json").read_bytes()
    plan = json.loads(raw_plan)
    if plan["config"] != CONFIG or plan["source_hashes"] != sources() or plan["model"] != identity():
        raise ValueError("Frozen code, configuration or model changed")
    if plan["input_hashes"] != input_hashes(data, tokenizer_dir) or plan["ollama"] != local_api("/api/version"):
        raise ValueError("Frozen inputs/server changed")
    if plan["tokenizers_version"] != importlib.metadata.version("tokenizers"):
        raise ValueError("Frozen tokenizer implementation changed")
    if (not 1 <= len(plan["samples"]) <= 12
            or len({s["id"] for s in plan["samples"]}) != len(plan["samples"])):
        raise ValueError("Pilot limit changed")
    items, tokenizer = load_inputs(data, tokenizer_dir)
    lookup = {r["_id"]: r for r in items}
    prompt_templates = templates()
    for sample in plan["samples"]:
        if sample["item_sha256"] != digest(canonical(lookup[sample["id"]])) or sorted(sample["order"]) != sorted(METHODS):
            raise ValueError("Task content/order changed")
    write_new(output / "started.json", dict(plan_sha256=digest(raw_plan), utc=time.time()))
    rows, telemetry = [], []
    stop = threading.Event()
    observer = threading.Thread(target=monitor, args=(stop, telemetry), daemon=True)
    observer.start()
    fatal = None
    began = time.perf_counter()
    try:
        for sample in plan["samples"]:
            item = lookup[sample["id"]]
            for method in sample["order"]:
                row = dict(id=sample["id"], method=method, domain=sample["domain"],
                           length=sample["length"], difficulty=sample["difficulty"], status="error")
                started = time.perf_counter()
                try:
                    if identity() != plan["model"]:
                        raise ValueError("Local model identity changed")
                    row["identity_check_seconds"] = time.perf_counter()-started
                    prep_start = time.perf_counter()
                    prompt, meta = build_prompt(item, method, tokenizer, prompt_templates)
                    if meta != sample["prompts"][method]:
                        raise ValueError("Frozen prompt changed")
                    row.update(meta)
                    row["context_preparation_seconds"] = time.perf_counter()-prep_start
                    call_start = time.perf_counter()
                    reply = generate(prompt)
                    row["model_call_seconds"] = time.perf_counter()-call_start
                    row["response"] = reply.get("response", "")
                    row["response_sha256"] = digest(row["response"].encode())
                    row["usage"] = {k: reply.get(k) for k in ("prompt_eval_count", "eval_count", "total_duration",
                        "load_duration", "prompt_eval_duration", "eval_duration", "done_reason")}
                    if not reply.get("done") or reply.get("model") != MODEL:
                        raise ValueError("Generation incomplete or model changed")
                    if reply.get("prompt_eval_count") != meta["input_tokens"]:
                        raise ValueError("Server/tokenizer count mismatch; truncation or template drift possible")
                    if not isinstance(reply.get("eval_count"), int) or reply["eval_count"] < 1:
                        raise ValueError("Missing output token accounting")
                    row["prediction"] = extract_answer(row["response"])
                    row["target"] = item["answer"]
                    row["correct"] = row["prediction"] == item["answer"]
                    row["status"] = "ok"
                except Exception as error:
                    row["error"] = f"{type(error).__name__}: {error}"
                    fatal = row["error"]
                row["whole_request_seconds"] = time.perf_counter()-started
                rows.append(row)
                write_new(output / f"attempt-{len(rows):03d}.json", row)
                print(json.dumps({k: row.get(k) for k in ("id", "method", "status", "correct", "error",
                                                         "whole_request_seconds")}), flush=True)
                if fatal:
                    break
            if fatal:
                break
        if identity() != plan["model"]:
            fatal = "Model changed at final identity check"
        write_new(output / "resident-models.json", local_api("/api/ps"))
    except BaseException as error:
        fatal = f"{type(error).__name__}: {error}"
        raise
    finally:
        stop.set()
        observer.join(timeout=7)
        write_new(output / "telemetry.json", telemetry)
        if len(rows) != len(plan["samples"])*len(METHODS) and fatal is None:
            fatal = "Incomplete attempt sequence"
        write_new(output / "summary.json", dict(plan_sha256=digest(raw_plan), status="failed" if fatal else "completed",
            seconds=time.perf_counter()-began, methods=summarize(plan, rows), fatal=fatal,
            attempt_hashes={f"attempt-{i:03d}.json": file_hash(output/f"attempt-{i:03d}.json")
                            for i in range(1, len(rows)+1)}, telemetry_sha256=file_hash(output/"telemetry.json"),
            ttft_seconds=None, electricity_cost_usd=None, provider_charge_usd=0,
            limitations=["Small development control experiment, no runtime treatment or promotion",
                         "First call includes model load; prefix cache not independently disabled",
                         "No streaming TTFT, repeated trials, RAM measurement or concurrency sweep",
                         "Success-latency summaries exclude failed requests; failed attempts retained separately",
                         "Preparation includes full-dataset eligibility outside request timing",
                         "Labels checked only after generation; correctness does not prove evidence sufficiency"]))
    if fatal:
        raise RuntimeError(fatal)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "run"))
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--length-audit", type=Path, help="Optional original pinned length-only audit; no outcomes")
    args = parser.parse_args()
    if args.action == "prepare":
        prepare(args.data, args.tokenizer, args.output, args.length_audit)
    else:
        run(args.data, args.tokenizer, args.output)

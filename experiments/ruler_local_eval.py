"""Paired local RULER development experiment; no hosted calls or weight updates.

Copyright (c) 2026 Prashant Jagtap. MIT License.
Raw benchmark text stays in an outside-repository output directory.
"""

import argparse
import ast
import hashlib
import json
import random
import statistics
import time
import urllib.request
from pathlib import Path

from local_eval import canonical, local_api, model_identity
from longbench_eval import bm25_order, chunks
from ruler_context import frame, native_score, runtime_context, select
from ruler_native import TASKS, outside_repo, sha, write_new

MODEL = "qwen2.5:1.5b"
MODEL_DIGEST = "65ec06548149b04c096a120e4a6da9d4017ea809c91734ea5631e89f96ddc57b"
OPTIONS = dict(temperature=0, seed=71, num_ctx=32768, num_predict=256)
ARMS = ("full", "bm25", "tool_context", "runtime")
BASE = Path(__file__).resolve().parent


def sources():
    files = [BASE / name for name in ("ruler_local_eval.py", "ruler_context.py", "ruler_native.py",
                                       "longbench_eval.py", "local_eval.py")]
    files += list((BASE.parent / "context_stamps").rglob("*.py")) + [BASE.parent / "stamps.py"]
    return {str(path.relative_to(BASE.parent)).replace("\\", "/"): sha(path) for path in sorted(files)}


def chat(text):
    return "<|im_start|>user\n" + text + "<|im_end|>\n<|im_start|>assistant\n"


def tokenizer_at(path):
    from tokenizers import Tokenizer
    tokenizer = Tokenizer.from_file(str(Path(path) / "tokenizer.json"))
    tokenizer.no_truncation()
    tokenizer.no_padding()
    return tokenizer


def public_rows(data):
    """This loader does not open raw generator JSONL or scoring key files."""
    plan = json.loads((data / "plan.json").read_bytes())
    inventory = json.loads((data / "inventory.json").read_bytes())
    expected = {(length, task) for length in plan["lengths"] for task in TASKS}
    if set((r["length"], r["task"]) for r in inventory) != expected or len(inventory) != len(expected):
        raise ValueError("All registered task/length cells required")
    rows = []
    for entry in inventory:
        path = data / str(entry["length"]) / entry["task"] / "inputs.json"
        if sha(path) != entry["inputs_sha256"]:
            raise ValueError("Prepared public inputs changed")
        group = json.loads(path.read_bytes())
        if len(group) != plan["count_per_task_length"]:
            raise ValueError("Incomplete cell")
        for row in group:
            if set(row) != {"id", "task", "length", "question"}:
                raise ValueError("Only whitelisted public fields may enter model preparation")
        rows.extend(group)
    if len({r["id"] for r in rows}) != len(rows):
        raise ValueError("Duplicate public IDs")
    return rows


def score_parity(path):
    """Execute only the two reviewed pure nested metric functions, not NeMo imports."""
    if sha(path) != "60a831e83ffb0d93e1f80c2b2357d8dc23ca9b17f51b5efb07c0abae16c35d88":
        raise ValueError("Reviewed pinned NeMo scorer required")
    source = ast.parse(Path(path).read_text(encoding="utf-8"))
    entry = next(n for n in source.body if isinstance(n, ast.FunctionDef) and n.name == "eval_ruler")
    names = {"string_match_all_single", "string_match_part_single"}
    functions = [n for n in entry.body if isinstance(n, ast.FunctionDef) and n.name in names]
    if {n.name for n in functions} != names:
        raise ValueError("Native scorer structure changed")
    namespace = {}
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(path), "exec"), namespace)
    cases = [("", ["abc"]), ("ABC", ["abc", "def"]), ("foobar", ["foo", "xyz"]),
             ("x\ny", ["x y", "x\ny"]), ("12", ["1", "12", "123"])]
    for prediction, references in cases:
        for mode in ("all", "part"):
            expected = namespace[f"string_match_{mode}_single"](prediction, references)
            if expected != native_score(prediction, references, mode):
                raise ValueError("Native metric parity failed")
    return dict(sha256=sha(path), cases=len(cases)*2, match=True)


def prompts(question, tokenizer):
    def count(text):
        return len(tokenizer.encode(text, add_special_tokens=False).ids)
    view = frame(question)
    results, metadata = {}, {}
    started = time.perf_counter()
    results["full"] = chat(question)
    metadata["full"] = dict(preparation_seconds=time.perf_counter()-started)
    started = time.perf_counter()
    pieces = chunks(tokenizer, view.context)
    selected = []
    for index in bm25_order(view.suffix, pieces):
        candidate = "\n\n".join(pieces[i][2] for i in sorted(selected + [index]))
        if count(chat(view.render(candidate))) <= 4096:
            selected.append(index)
    results["bm25"] = chat(view.render("\n\n".join(pieces[i][2] for i in sorted(selected))))
    metadata["bm25"] = dict(preparation_seconds=time.perf_counter()-started,
                             selected_spans=[[pieces[i][0], pieces[i][1]] for i in sorted(selected)])
    started = time.perf_counter()
    selection = select(view)
    selection_seconds = time.perf_counter()-started
    started = time.perf_counter()
    tool_text = ("" if selection.kind == "full" else "\n").join(selection.texts)
    results["tool_context"] = chat(view.render(tool_text))
    metadata["tool_context"] = dict(preparation_seconds=selection_seconds+time.perf_counter()-started,
                                     selection_kind=selection.kind, contract=selection.contract)
    started = time.perf_counter()
    material, receipt = runtime_context(view, selection, count)
    results["runtime"] = chat(view.render(material))
    metadata["runtime"] = dict(preparation_seconds=selection_seconds+time.perf_counter()-started,
                                selection_kind=selection.kind, receipt=receipt)
    for arm, prompt in results.items():
        tokens = count(prompt)
        if tokens > (4096 if arm == "bm25" else 30000):
            raise ValueError(f"{arm} complete input exceeds budget; truncation refused")
        metadata[arm].update(input_tokens=tokens, prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest())
    return results, metadata, selection.direct_answer


def prepare(data, tokenizer, scorer, output):
    output = outside_repo(output)
    output.mkdir(parents=True, exist_ok=False)
    model = model_identity(MODEL)
    if model["digest"] != MODEL_DIGEST:
        raise ValueError("Existing local model changed")
    parity = score_parity(scorer)
    tok = tokenizer_at(tokenizer)
    rows = public_rows(data)
    prepared = []
    for row in rows:
        prompt, metadata, direct = prompts(row["question"], tok)
        order = list(ARMS)
        random.Random("ruler-local-71:"+row["id"]).shuffle(order)
        prepared.append({k: v for k, v in row.items() if k != "question"} |
                        dict(prompts=prompt, metadata=metadata, direct_answer=direct, order=order))
    write_new(output / "inputs.json", prepared)
    plan = dict(schema=1, purpose="Paired development treatment; benchmark-aware contracts, no promotion",
                model=model, options=OPTIONS, arms=ARMS, sources=sources(), scorer=parity,
                inputs_sha256=sha(output/"inputs.json"), data_plan_sha256=sha(data/"plan.json"),
                inventory_sha256=sha(data/"inventory.json"), tokenizer_sha256=sha(tokenizer/"tokenizer.json"),
                samples=len(prepared), ollama=local_api("/api/version"),
                limitations=["No semantic sufficiency certificate for document QA: keep full context",
                             "Exact lookup, ordered assignment and frequency contracts use benchmark grammar",
                             "Direct deterministic control receives the same public input; QA abstains",
                             "32-byte receipt references externally held evidence; it is not the reader input",
                             "No weight updates, repeated-task cache, edge-device or retention qualification",
                             "Single trial; default native substring metric can award partial/false-positive credit",
                             "Qwen raw ChatML, 256 output cap, custom loopback Ollama backend; not official leaderboard protocol",
                             "Preparation measures CPU selection/runtime overhead; total request latency adds it to local call time"])
    write_new(output / "plan.json", plan)
    print(json.dumps(dict(samples=len(prepared), scheduled_model_calls=len(prepared)*len(ARMS), scorer=parity)))


def generate(prompt):
    class NoRedirect(urllib.request.HTTPRedirectHandler):
        def redirect_request(self, req, fp, code, msg, headers, newurl):
            raise ValueError("Local model redirect refused")
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect())
    request = urllib.request.Request("http://127.0.0.1:11434/api/generate", data=canonical(
        dict(model=MODEL, prompt=prompt, raw=True, stream=False, options=OPTIONS, keep_alive="10m")),
        headers={"Content-Type": "application/json"})
    with opener.open(request, timeout=240) as response:
        data = response.read(4*1024*1024+1)
    if len(data) > 4*1024*1024:
        raise ValueError("Oversized local generation response")
    return json.loads(data)


def run(output):
    output = outside_repo(output)
    plan = json.loads((output/"plan.json").read_bytes())
    if (plan["sources"] != sources() or plan["options"] != OPTIONS or
            plan["inputs_sha256"] != sha(output/"inputs.json") or
            model_identity(MODEL) != plan["model"]):
        raise ValueError("Frozen source/input/model/config changed")
    rows = json.loads((output/"inputs.json").read_bytes())
    # Fresh run only: uncertain/partial model calls are never silently retried.
    with (output/"generations.jsonl").open("x", encoding="utf-8", newline="\n") as stream:
        started = time.perf_counter()
        warm = generate(chat("Reply with OK."))
        write_new(output/"warmup.json", dict(response=warm, wall_seconds=time.perf_counter()-started,
                                             resident_models=local_api("/api/ps")))
        for row in rows:
            for arm in row["order"]:
                started = time.perf_counter()
                result, error = None, None
                try:
                    result = generate(row["prompts"][arm])
                    if not result.get("done") or result.get("prompt_eval_count") != row["metadata"][arm]["input_tokens"]:
                        raise ValueError("Incomplete response or server/tokenizer input mismatch")
                except Exception as exc:
                    error = f"{type(exc).__name__}: {exc}"
                record = dict(id=row["id"], task=row["task"], length=row["length"], arm=arm,
                              response=result, error=error, wall_seconds=time.perf_counter()-started)
                stream.write(json.dumps(record, ensure_ascii=False)+"\n")
                stream.flush()
                print(json.dumps({k: record[k] for k in ("id", "arm", "error", "wall_seconds")}), flush=True)
    write_new(output/"completed.json", dict(records=len(rows)*len(ARMS),
                                            generations_sha256=sha(output/"generations.jsonl")))


def score(data, output):
    plan = json.loads((output/"plan.json").read_bytes())
    if sha(data/"inventory.json") != plan["inventory_sha256"] or sha(output/"inputs.json") != plan["inputs_sha256"]:
        raise ValueError("Frozen data changed")
    completed = json.loads((output/"completed.json").read_bytes())
    if completed["generations_sha256"] != sha(output/"generations.jsonl"):
        raise ValueError("Generation log changed")
    keys = {}
    for entry in json.loads((data/"inventory.json").read_bytes()):
        path = data/str(entry["length"])/entry["task"]/"keys.json"
        if sha(path) != entry["keys_sha256"]:
            raise ValueError("Scoring key changed")
        for row in json.loads(path.read_bytes()):
            keys[row["id"]] = row
    inputs = {r["id"]: r for r in json.loads((output/"inputs.json").read_bytes())}
    records = [json.loads(line) for line in (output/"generations.jsonl").read_text(encoding="utf-8").splitlines()]
    expected = {(key, arm) for key in inputs for arm in ARMS}
    if len(records) != len(expected) or {(r["id"], r["arm"]) for r in records} != expected:
        raise ValueError("Every paired arm must be retained exactly once")
    scored = []
    for row in records:
        key = keys[row["id"]]
        response = row["response"] or {}
        value = 0. if row["error"] else native_score(response.get("response", ""), key["expected_answer"], key["match_type"])
        metadata = inputs[row["id"]]["metadata"][row["arm"]]
        scored.append({k: row[k] for k in ("id", "arm", "task", "length", "error", "wall_seconds")} |
                      dict(score=value, input_tokens=metadata["input_tokens"], output_tokens=response.get("eval_count", 0),
                           total_seconds=row["wall_seconds"]+metadata["preparation_seconds"],
                           truncated=response.get("done_reason") == "length"))
    summary = dict(schema=1, samples=len(inputs), calls=len(records), errors=sum(bool(r["error"]) for r in scored),
                   truncations=sum(r["truncated"] for r in scored), cells=[], overall={}, direct_control={})
    for length in sorted({r["length"] for r in scored}):
        for task in TASKS:
            cell = dict(length=length, task=task, arms={})
            for arm in ARMS:
                group = [r for r in scored if (r["length"], r["task"], r["arm"]) == (length, task, arm)]
                cell["arms"][arm] = dict(n=len(group), score=statistics.mean(r["score"] for r in group),
                    input_tokens=sum(r["input_tokens"] for r in group), total_seconds=sum(r["total_seconds"] for r in group))
            summary["cells"].append(cell)
    for arm in ARMS:
        group = [r for r in scored if r["arm"] == arm]
        summary["overall"][arm] = dict(native_mean=statistics.mean(r["score"] for r in group),
            complete_credit=sum(r["score"] == 1 for r in group), input_tokens=sum(r["input_tokens"] for r in group),
            output_tokens=sum(r["output_tokens"] for r in group), total_seconds=sum(r["total_seconds"] for r in group))
    direct = [(r, native_score(r["direct_answer"] or "", keys[r["id"]]["expected_answer"], keys[r["id"]]["match_type"])) for r in inputs.values()]
    summary["direct_control"] = dict(native_mean=statistics.mean(v for _, v in direct),
        complete_credit=sum(v == 1 for _, v in direct), abstentions=sum(r["direct_answer"] is None for r, _ in direct), model_calls=0)
    summary["limitations"] = plan["limitations"]
    write_new(output/"scored.json", scored)
    write_new(output/"summary.json", summary)
    print(json.dumps({k: summary[k] for k in ("samples", "calls", "errors", "truncations", "overall", "direct_control")}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "run", "score"))
    parser.add_argument("--data", type=Path)
    parser.add_argument("--tokenizer", type=Path)
    parser.add_argument("--scorer", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.mode == "prepare":
        if None in (args.data, args.tokenizer, args.scorer):
            parser.error("prepare needs --data --tokenizer --scorer")
        prepare(args.data, args.tokenizer, args.scorer, args.output)
    elif args.mode == "run":
        run(args.output)
    else:
        if args.data is None:
            parser.error("score needs --data")
        score(args.data, args.output)

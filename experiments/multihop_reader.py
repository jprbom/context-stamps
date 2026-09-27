"""Frozen four-arm local SLM comparison for source-filtered MuSiQue pairs.

Copyright (c) 2026 Prashant Jagtap. MIT License.
All inputs/prompts stay local. No answer/decomposition file is read by prepare
or run. This is a development protocol, not a complete benchmark submission.
"""

import argparse
import hashlib
import json
import os
import random
import time
import urllib.request
from pathlib import Path

from local_eval import canonical, local_api, model_identity
from relation_ranker import rank
from ruler_local_eval import tokenizer_at
from ruler_native import outside_repo, sha, write_new

from context_stamps.context_state import AccessScope, CanonicalNode, ContextState, TemporalScope

ROOT = Path(__file__).resolve().parents[1]
MODEL = "qwen2.5:1.5b"
DIGEST = "65ec06548149b04c096a120e4a6da9d4017ea809c91734ea5631e89f96ddc57b"
OPTIONS = dict(temperature=0, seed=71, num_ctx=4096, num_predict=128)
ARMS = ("full", "bm25", "pointwise", "diffusion")


def sources():
    names = ("experiments/multihop_reader.py", "experiments/relation_ranker.py", "experiments/multihop_data.py",
             "experiments/local_eval.py", "experiments/ruler_local_eval.py", "experiments/ruler_native.py",
             "context_stamps/context_state.py", "context_stamps/security.py")
    return {name: sha(ROOT/name) for name in names}


def read(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def render(question, selected):
    body = "\n\n".join(f'[{p["idx"]}] {p["title"]}\n{p["text"]}' for p in selected)
    system = ("Answer the question using only the supplied passages. If the passages do not support an answer, "
              "output UNKNOWN. Otherwise output only the shortest answer. Do not explain.")
    return (f"<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\nPassages:\n{body}\n\n"
            f"Question: {question}<|im_end|>\n<|im_start|>assistant\n")


def compile_context(case, arm, policy, tokenizer):
    tick = time.perf_counter()
    selected = case["paragraphs"] if arm == "full" else rank(case, policy)[0][:6]
    selected = sorted(selected, key=lambda p: (p["sha256"], p["idx"]))
    text = render(case["question"], selected)
    tokens = len(tokenizer.encode(text, add_special_tokens=False).ids)
    if tokens > 3968:
        raise ValueError("complete prompt exceeds frozen capacity; no truncation permitted")
    state = ContextState(tenant="musique-public", policy_revision="public-v1", clock=lambda: 1)
    for p in selected:
        state.put(CanonicalNode(key=f'paragraph-{p["idx"]}', revision=p["sha256"], tenant="musique-public",
                                text=p["title"]+"\n"+p["text"], kind="OBSERVATION",
                                temporal=TemporalScope(1, 1), roles=("reader",), provenance="musique-v1"))
    snapshot = state.snapshot(AccessScope("musique-public", "local-reader", "public-v1", ("reader",)), at=1, known_at=1)
    seal = state.seal(snapshot, text)
    if not state.verify_binding(snapshot, text, seal):
        raise ValueError("source/scope binding failed")
    return text, dict(input_tokens=tokens, selected_indices=[p["idx"] for p in selected],
                      selected_hashes=[p["sha256"] for p in selected], prompt_sha256=hashlib.sha256(text.encode()).hexdigest(),
                      compile_seconds=time.perf_counter()-tick, scope_checked=True, sufficiency_certified=False)


def prepare(data, trained, tokenizer, output):
    data, trained, output = map(outside_repo, (data, trained, output))
    identities = model_identity(MODEL)
    if identities["digest"] != DIGEST:
        raise ValueError("registered local model revision required")
    data_manifest = json.loads((data/"prepared.json").read_bytes())
    if sha(data/"evaluation-inputs.jsonl") != data_manifest["files"]["evaluation-inputs.jsonl"]:
        raise ValueError("evaluation inputs changed")
    training = json.loads((trained/"training.json").read_bytes())
    if training["candidate"] != "diffusion":
        raise ValueError("registered validation-selected candidate changed")
    policies = {n: json.loads((trained/(n+".json")).read_bytes()) for n in ("pointwise", "diffusion")}
    output.mkdir(parents=True, exist_ok=False)
    tok = tokenizer_at(tokenizer)
    plan = dict(model=identities, options=OPTIONS, arms=ARMS, inputs=sha(data/"evaluation-inputs.jsonl"),
                prepared=sha(data/"prepared.json"), trained={n: sha(trained/n) for n in ("pointwise.json", "diffusion.json", "training.json")},
                tokenizer=sha(tokenizer/"tokenizer.json"), source_hashes=sources(), selected_per_compact_arm=6,
                key_files_opened=False, planned_cases=128, planned_calls=512,
                primary="native answer F1 and grouped answer+sufficiency; full/BM25/pointwise controls",
                resource_scope="CPU compilation and actual nonstreamed model request wall/tokens; no energy or complete process-tree peak claim")
    write_new(output/"registration.json", plan)
    prepared = []
    for case in read(data/"evaluation-inputs.jsonl"):
        prompts, metadata = {}, {}
        for arm in ARMS:
            prompts[arm], metadata[arm] = compile_context(case, arm, policies.get(arm), tok)
        order = list(ARMS)
        random.Random("musique-reader-v1:"+case["key"]).shuffle(order)
        prepared.append(dict(key=case["key"], prompts=prompts, metadata=metadata, order=order))
    if len(prepared) != 128 or len({r["key"] for r in prepared}) != 128:
        raise ValueError("all 128 unique reserved inputs required")
    write_new(output/"inputs.json", prepared)
    write_new(output/"prepared.json", dict(registration=sha(output/"registration.json"), inputs=sha(output/"inputs.json")))
    print(json.dumps(dict(cases=len(prepared), calls=512, max_input_tokens=max(m["input_tokens"] for r in prepared for m in r["metadata"].values()))), flush=True)


def generate(text):
    class NoRedirect(urllib.request.HTTPRedirectHandler):
        def redirect_request(self, req, fp, code, msg, headers, newurl):
            raise ValueError("local model redirect refused")
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect())
    req = urllib.request.Request("http://127.0.0.1:11434/api/generate", headers={"Content-Type": "application/json"},
                                 data=canonical(dict(model=MODEL, prompt=text, raw=True, stream=False,
                                                     options=OPTIONS, keep_alive="10m")))
    with opener.open(req, timeout=240) as response:
        raw = response.read(4*1024**2+1)
    if len(raw) > 4*1024**2:
        raise ValueError("oversized local result")
    return json.loads(raw)


def run(output):
    plan = json.loads((output/"registration.json").read_bytes())
    prepared = json.loads((output/"prepared.json").read_bytes())
    if plan["source_hashes"] != sources() or plan["options"] != OPTIONS or plan["model"] != model_identity(MODEL):
        raise ValueError("frozen source, model or options changed")
    if sha(output/"registration.json") != prepared["registration"] or sha(output/"inputs.json") != prepared["inputs"]:
        raise ValueError("prepared inputs changed")
    if local_api("/api/ps").get("models"):
        raise ValueError("wait for the existing local workload; do not evict it")
    rows = json.loads((output/"inputs.json").read_bytes())
    warm = generate("<|im_start|>user\nReply OK.<|im_end|>\n<|im_start|>assistant\n")
    write_new(output/"warmup.json", dict(response=warm, resident=local_api("/api/ps")))
    count = 0
    with (output/"generations.jsonl").open("x", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            for arm in row["order"]:
                record = dict(key=row["key"], arm=arm, model=MODEL, **row["metadata"][arm])
                tick = time.perf_counter()
                try:
                    result = generate(row["prompts"][arm])
                    record.update(prediction=result.get("response", ""), output_tokens=result.get("eval_count", 0),
                                  error=None, truncated=result.get("done_reason") == "length",
                                  timings={k: result.get(k) for k in ("total_duration", "load_duration", "prompt_eval_duration", "eval_duration")})
                    if not result.get("done") or result.get("prompt_eval_count") != record["input_tokens"]:
                        raise ValueError("incomplete generation or tokenizer mismatch")
                except Exception as exc:
                    record.update(error=f"{type(exc).__name__}: {exc}", prediction=record.get("prediction", ""))
                    record["generation_wall_seconds"] = time.perf_counter()-tick
                    stream.write(json.dumps(record, ensure_ascii=False)+"\n")
                    stream.flush()
                    os.fsync(stream.fileno())
                    raise  # uncertain requests are not blindly retried
                record["generation_wall_seconds"] = time.perf_counter()-tick
                stream.write(json.dumps(record, ensure_ascii=False)+"\n")
                stream.flush()
                count += 1
            if count % 32 == 0:
                print(json.dumps(dict(records=count, total=512)), flush=True)
    write_new(output/"complete.json", dict(records=count, sha256=sha(output/"generations.jsonl"), resident=local_api("/api/ps")))
    local_api("/api/generate", dict(model=MODEL, keep_alive=0))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "run"))
    for name in ("data", "trained", "tokenizer", "output"):
        parser.add_argument("--"+name, type=Path, required=name == "output")
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args.data, args.trained, args.tokenizer, args.output)
    else:
        run(outside_repo(args.output))

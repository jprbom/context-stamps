"""Paired structural-memory development control on two installed local readers.

Copyright (c) 2026 Prashant Jagtap. MIT License.
The old LME experiment is frozen. This is a new, deliberately smaller protocol,
not a fresh benchmark holdout or a weight-training/activation experiment.
"""

import argparse
import collections
import hashlib
import json
import random
import sqlite3
import time
import urllib.request
from dataclasses import asdict
from pathlib import Path

from lme_memory import episode, native_scorer, prompt, score, summary, terms
from lme_rank_fast import ranked_fast
from local_eval import canonical, local_api, model_identity
from ruler_local_eval import sources, tokenizer_at
from ruler_native import outside_repo, sha, write_new

from context_stamps.context_state import AccessScope, ContextState
from context_stamps.structured_memory import ObservationSpan, StructureFragment, structured_views
from context_stamps.trajectory import trajectory_views

ROOT = Path(__file__).resolve().parents[1]
MODELS = {"qwen2.5:1.5b": "65ec06548149b04c096a120e4a6da9d4017ea809c91734ea5631e89f96ddc57b",
          "qwen2.5-coder:7b": "dae161e27b0e90dd1856c8bb3209201fd6736d8eb66298e75ed87571486f4364"}
OPTIONS = dict(temperature=0, seed=71, num_ctx=8192, num_predict=256)
ARMS = ("raw", "structure")


def code_sources():
    return sources() | {"experiments/"+name: sha(ROOT/"experiments"/name)
                        for name in ("lme_memory.py", "lme_rank_fast.py", "lme_structure.py")}


def build(data, output, allowed):
    tick = time.perf_counter()
    db = sqlite3.connect(output/"structure.sqlite")
    db.execute("CREATE VIRTUAL TABLE memory USING fts5(body, payload UNINDEXED, domain UNINDEXED, episode UNINDEXED, key UNINDEXED)")
    fragments, occurrences, fallbacks, covered = 0, 0, [], set()
    with db, (data/"trajectories.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            if row["id"] not in allowed:
                continue
            ep = episode(row)
            covered.add(ep.key)
            try:
                views = list(structured_views(ep))
            except ValueError as exc:
                # Whole-episode fallback is explicit and counted. No partial mix.
                fallbacks.append(dict(episode=ep.key, reason=str(exc)))
                views = [v for v in trajectory_views(ep) if v.channel == "state"]
            for view in views:
                structured = type(view) is StructureFragment
                body = view.text + "\nRecorded goal: " + ep.goal
                payload = dict(structured=structured, view=asdict(view))
                db.execute("INSERT INTO memory VALUES (?,?,?,?,?)",
                           (body, json.dumps(payload, ensure_ascii=False), row["domain"], ep.key, view.key))
                fragments += 1
                occurrences += len(view.occurrences) if structured else 1
            if len(covered) % 25 == 0:
                print(json.dumps(dict(indexed_episodes=len(covered), fragments=fragments)), flush=True)
    if covered != allowed:
        raise ValueError("incomplete authorized corpus")
    db.execute("INSERT INTO memory(memory) VALUES ('optimize')")
    db.commit()
    write_new(output/"index.json", dict(episodes=len(covered), fragments=fragments, occurrences=occurrences,
                                        raw_fallbacks=fallbacks, bytes=(output/"structure.sqlite").stat().st_size,
                                        seconds=time.perf_counter()-tick))
    return db


def deserialize(payload):
    from context_stamps.trajectory import TraceFragment
    row = json.loads(payload)
    v = row["view"]
    if not row["structured"]:
        return TraceFragment(**(v | dict(steps=tuple(v["steps"]))))
    spans = tuple(ObservationSpan(**(s | dict(ancestors=tuple(s["ancestors"])))) for s in v["occurrences"])
    return StructureFragment(**(v | dict(occurrences=spans)))


def ranked(db, question, domain, allowed):
    words = terms(question)
    if not words:
        return []
    query = " OR ".join('"'+w+'"' for w in words)
    # Filter permitted episodes before the top-k cutoff, including tie handling.
    cursor = db.execute("SELECT payload,episode,key,rank FROM memory WHERE memory MATCH ? AND domain=? ORDER BY rank", (query, domain))
    rows, cutoff = [], None
    try:
        for row in cursor:
            if cutoff is not None and row[-1] > cutoff:
                break
            if row[1] not in allowed:
                continue
            rows.append(row)
            if len(rows) == 240:
                cutoff = row[-1]
    finally:
        cursor.close()
    rows.sort(key=lambda r: (r[-1], r[2]))
    return [deserialize(r[0]) for r in rows[:240]]


def pack(question, domain, views, tok):
    selected, bodies = [], []
    # Same candidate inspection limit, prompt, token cap and selected-view cap.
    for view in views[:32]:
        candidate = json.dumps(bodies+[view.text], ensure_ascii=False)
        if len(tok.encode(prompt(question, domain, candidate), add_special_tokens=False).ids) <= 6144:
            selected.append(view)
            bodies.append(view.text)
        if len(selected) == 16:
            break
    rendered = prompt(question, domain, json.dumps(bodies, ensure_ascii=False))
    count = len(tok.encode(rendered, add_special_tokens=False).ids)
    if count > 6144:
        raise ValueError("question alone exceeds prompt budget")
    state = ContextState(tenant="lme-public", policy_revision="public-v1", clock=lambda: 1)
    for view in selected:
        state.put(view.canonical_node(tenant="lme-public", roles=("reader",), observed_at=1))
    snapshot = state.snapshot(AccessScope("lme-public", "evaluator", "public-v1", ("reader",)), at=1, known_at=1)
    seal = state.seal(snapshot, rendered)
    if not state.verify_binding(snapshot, rendered, seal):
        raise ValueError("source/scope binding failed")
    return rendered, dict(input_tokens=count, prompt_sha256=hashlib.sha256(rendered.encode()).hexdigest(),
                          sources=[dict(key=v.key, episode=v.episode, revision=v.source_revision,
                                        view_sha256=hashlib.sha256(v.text.encode()).hexdigest()) for v in selected],
                          scope_checked=True, sufficient_context_certified=False)


def prepare(data, previous, tokenizer, output):
    output = outside_repo(output)
    output.mkdir(parents=True, exist_ok=False)
    old = json.loads((previous/"registration.json").read_bytes())
    old_prepared = json.loads((previous/"prepared.json").read_bytes())
    if sha(previous/"memory.sqlite") != old_prepared["index_sha256"]:
        raise ValueError("frozen raw baseline index changed")
    if any(sha(data/p) != h for p, h in old["dataset_hashes"].items()):
        raise ValueError("pinned public data changed")
    questions = [json.loads(line) for line in (data/"questions.jsonl").read_text(encoding="utf-8").splitlines()]
    eligible = [q for q in questions if q["id"] in old["train_ids"]+old["holdout_ids"]]
    cells = collections.defaultdict(list)
    for q in eligible:
        cells[(q["domain"], q["question_type"])].append(q)
    chosen = []
    for values in cells.values():
        chosen += sorted(values, key=lambda q: hashlib.sha256(("lme-structure-v1:"+q["id"]).encode()).hexdigest())[:12]
    if len(chosen) != 72:
        raise ValueError("six cells of twelve questions required")
    models = {name: model_identity(name) for name in MODELS}
    if any(models[m]["digest"] != MODELS[m] for m in MODELS):
        raise ValueError("pinned local models required")
    plan = dict(schema=1, sources=code_sources(), models=models, options=OPTIONS, arms=ARMS,
                tokenizer_sha256=sha(tokenizer/"tokenizer.json"), dataset_hashes=old["dataset_hashes"],
                source_revision=old["source_revision"], data_revision=old["data_revision"],
                baseline_index_sha256=old_prepared["index_sha256"], ids=sorted(q["id"] for q in chosen),
                samples=72, scheduled_calls=288, chunk_chars=2400, prompt_token_cap=6144,
                selection="12 per domain/ability cell by fixed SHA-256 order; all already development material",
                model_order=list(MODELS), model_order_randomized=False,
                design="Two readers x two freshly measured arms; paired arm order shuffled per question; no learned routing",
                limits=["No independent deployment holdout, adaptation training, retention test or activation",
                        "Both representations use the same 200 source histories; native metric permits some false positives",
                        "Coder 7B differs in training and size; not an isolated parameter-count ablation",
                        "Structural projection removes instance handles; source remains external",
                        "Warm local workstation measurements; not an edge-device, concurrency or energy result",
                        "No action execution, causal inference, remote judge or paid call"],
                ollama=local_api("/api/version"))
    write_new(output/"registration.json", plan)
    haystacks = json.loads((data/"haystacks/lme_v2_small.json").read_bytes())
    allowed = {i for q in chosen for i in haystacks[q["id"]]}
    structured = build(data, output, allowed)
    raw = sqlite3.connect((previous/"memory.sqlite").resolve().as_uri()+"?mode=ro", uri=True)
    tok, inputs = tokenizer_at(tokenizer), []
    for q in chosen:
        # Answer and scorer keys never enter retrieval or model prompts.
        question, domain, qid = q["question"], q["domain"], q["id"]
        scope = set(haystacks[qid])
        prompts, meta = {}, {}
        for arm in ARMS:
            tick = time.perf_counter()
            views = ranked_fast(raw, question, domain, "state", scope) if arm == "raw" else ranked(structured, question, domain, scope)
            ranked_seconds = time.perf_counter()-tick
            tick = time.perf_counter()
            prompts[arm], meta[arm] = pack(question, domain, views, tok)
            meta[arm].update(retrieval_seconds=ranked_seconds, pack_seconds=time.perf_counter()-tick)
        order = list(ARMS)
        random.Random("lme-structure-arm-v1:"+qid).shuffle(order)
        inputs.append(dict(id=qid, domain=domain, category=q["question_type"], prompts=prompts, metadata=meta, order=order))
    raw.close()
    structured.close()
    write_new(output/"inputs.json", inputs)
    write_new(output/"keys.json", [{k: q[k] for k in ("id", "answer", "eval_function")} for q in chosen])
    write_new(output/"prepared.json", {name: sha(output/name) for name in
                                      ("inputs.json", "keys.json", "registration.json", "structure.sqlite", "index.json")})
    print(json.dumps(dict(prepared=len(inputs), scheduled_calls=288)), flush=True)


def generate(model, rendered):
    if model not in MODELS:
        raise ValueError("unregistered model refused")
    class NoRedirect(urllib.request.HTTPRedirectHandler):
        def redirect_request(self, req, fp, code, msg, headers, newurl):
            raise ValueError("local model redirect refused")
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect())
    request = urllib.request.Request("http://127.0.0.1:11434/api/generate",
        data=canonical(dict(model=model, prompt=rendered, raw=True, stream=False, options=OPTIONS, keep_alive="10m")),
        headers={"Content-Type": "application/json"})
    with opener.open(request, timeout=240) as response:
        value = response.read(4*1024*1024+1)
    if len(value) > 4*1024*1024:
        raise ValueError("oversized response")
    return json.loads(value)


def run(output):
    plan = json.loads((output/"registration.json").read_bytes())
    prepared = json.loads((output/"prepared.json").read_bytes())
    if plan["sources"] != code_sources() or plan["options"] != OPTIONS:
        raise ValueError("registered implementation changed")
    if any(sha(output/p) != h for p, h in prepared.items()):
        raise ValueError("registered data changed")
    if any(model_identity(m) != plan["models"][m] for m in MODELS):
        raise ValueError("registered local model changed")
    rows = json.loads((output/"inputs.json").read_bytes())
    # Do not evict workloads belonging to the user. Start only from an idle server.
    if local_api("/api/ps").get("models"):
        raise ValueError("local model server is occupied; retry once it is idle")
    for model in MODELS:
        label = model.replace(":", "-")
        warm = generate(model, "<|im_start|>user\nReply OK.<|im_end|>\n<|im_start|>assistant\n")
        write_new(output/f"{label}-warmup.json", dict(result=warm, resident=local_api("/api/ps")))
        count = 0
        with (output/f"{label}-generations.jsonl").open("x", encoding="utf-8", newline="\n") as stream:
            for row in rows:
                for arm in row["order"]:
                    record = dict(id=row["id"], model=model, arm=arm, domain=row["domain"], category=row["category"],
                                  prediction="", output_tokens=0, error=None, truncated=False, **row["metadata"][arm])
                    tick = time.perf_counter()
                    try:
                        result = generate(model, row["prompts"][arm])
                        record.update(prediction=result.get("response", ""), output_tokens=result.get("eval_count", 0),
                                      truncated=result.get("done_reason") == "length",
                                      model_timings={k: result.get(k) for k in ("total_duration", "load_duration", "prompt_eval_duration", "eval_duration")})
                        if not result.get("done") or result.get("prompt_eval_count") != record["input_tokens"]:
                            raise ValueError("incomplete generation or tokenizer mismatch")
                    except Exception as exc:
                        record["error"] = f"{type(exc).__name__}: {exc}"
                    record["generation_wall_seconds"] = time.perf_counter()-tick
                    stream.write(json.dumps(record, ensure_ascii=False)+"\n")
                    stream.flush()
                    count += 1
                if count % 24 == 0:
                    print(json.dumps(dict(model=model, records=count)), flush=True)
        write_new(output/f"{label}-completed.json", dict(records=count, sha256=sha(output/f"{label}-generations.jsonl")))
        # Release only this experiment's model; no process termination or power changes.
        local_api("/api/generate", dict(model=model, keep_alive=0))


def evaluate(output, scorer):
    namespace = native_scorer(scorer)
    keys = {r["id"]: r for r in json.loads((output/"keys.json").read_bytes())}
    records, report = [], {}
    for model in MODELS:
        path = output/f"{model.replace(':', '-')}-generations.jsonl"
        completed = json.loads(path.with_name(path.stem.replace("-generations", "-completed")+".json").read_bytes())
        if sha(path) != completed["sha256"]:
            raise ValueError("generation log changed")
        rows = [json.loads(s) for s in path.read_text(encoding="utf-8").splitlines()]
        if len(rows) != 144 or {(r["id"], r["arm"]) for r in rows} != {(i, a) for i in keys for a in ARMS}:
            raise ValueError("incomplete paired experiment")
        rows = [r | dict(correct=not r["error"] and score(namespace, r["prediction"], keys[r["id"]])) for r in rows]
        arms = {a: {r["id"]: r for r in rows if r["arm"] == a} for a in ARMS}
        report[model] = {a: summary(list(arms[a].values())) for a in ARMS}
        report[model]["gains"] = [i for i in keys if arms["structure"][i]["correct"] and not arms["raw"][i]["correct"]]
        report[model]["regressions"] = [i for i in keys if arms["raw"][i]["correct"] and not arms["structure"][i]["correct"]]
        records.extend(rows)
    write_new(output/"scores.json", records)
    write_new(output/"summary.json", report)
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "run", "evaluate"))
    for arg in ("data", "previous", "tokenizer", "scorer", "output"):
        parser.add_argument("--"+arg, type=Path, required=arg == "output")
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args.data, args.previous, args.tokenizer, args.output)
    elif args.command == "run":
        run(args.output)
    else:
        evaluate(args.output, args.scorer)

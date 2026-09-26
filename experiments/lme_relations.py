"""Local page-selection and relation-packet ablations with live stage timing.

Copyright (c) 2026 Prashant Jagtap. MIT License.
Same 72 inspected development questions, no fresh-holdout or activation claim.
"""

import argparse
import collections
import hashlib
import json
import random
import sqlite3
import time
from dataclasses import asdict
from pathlib import Path

from lme_memory import episode, native_scorer, prompt, score, summary, terms
from lme_structure import MODELS, OPTIONS, deserialize, generate, pack
from lme_structure import code_sources as previous_sources
from local_eval import local_api, model_identity
from ruler_local_eval import tokenizer_at
from ruler_native import outside_repo, sha, write_new

from context_stamps.context_state import AccessScope, ContextState
from context_stamps.observation_packets import (
    ObservationPacket,
    RelationOccurrence,
    observation_packets,
    page_hint,
    plan_pages,
)

ARMS = ("structure", "scoped", "relations")
ROOT = Path(__file__).resolve().parents[1]


def code_sources():
    return previous_sources() | {"experiments/lme_relations.py": sha(__file__)}


def build(data, previous, output, allowed):
    tick = time.perf_counter()
    db = sqlite3.connect(output/"memory.sqlite")
    for table in ("structure", "relations"):
        db.execute(f"CREATE VIRTUAL TABLE {table} USING fts5(body,payload UNINDEXED,domain UNINDEXED,episode UNINDEXED,view_id UNINDEXED,page UNINDEXED)")
    old = sqlite3.connect((previous/"structure.sqlite").resolve().as_uri()+"?mode=ro", uri=True)
    with db:
        for body, payload, domain, key, view_id in old.execute("SELECT body,payload,domain,episode,key FROM memory"):
            if key not in allowed:
                continue
            view = deserialize(payload)
            path = page_hint(view.body) if hasattr(view, "body") else ()
            db.execute("INSERT INTO structure VALUES (?,?,?,?,?,?)",
                       (body, payload, domain, key, view_id, json.dumps(path)))
    old.close()
    counts, fallbacks, seen = collections.Counter(), [], set()
    with db, (data/"trajectories.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            if row["id"] not in allowed:
                continue
            ep = episode(row)
            seen.add(ep.key)
            try:
                views = list(observation_packets(ep))
            except ValueError as exc:
                fallbacks.append(dict(episode=ep.key, reason=str(exc)))
                views = []
            for view in views:
                db.execute("INSERT INTO relations VALUES (?,?,?,?,?,?)",
                    (view.text, json.dumps(asdict(view), ensure_ascii=False), row["domain"], view.episode,
                     view.key, json.dumps(view.page)))
                counts[view.relation] += 1
                counts["source_occurrences"] += len(view.occurrences)
            if len(seen) % 25 == 0:
                print(json.dumps(dict(indexed_episodes=len(seen), relations=sum(counts[r] for r in ("element", "ordered_members")))), flush=True)
    if seen != allowed:
        raise ValueError("incomplete public source scope")
    for table in ("structure", "relations"):
        db.execute(f"INSERT INTO {table}({table}) VALUES ('optimize')")
    db.commit()
    write_new(output/"index.json", dict(counts=dict(counts), skipped_relation_episodes=fallbacks,
        episodes=len(seen), bytes=(output/"memory.sqlite").stat().st_size, seconds=time.perf_counter()-tick))
    return db


def packet(payload):
    row = json.loads(payload)
    row["page"] = tuple(row["page"])
    row["occurrences"] = tuple(RelationOccurrence(s["step"], tuple(s["lines"])) for s in row["occurrences"])
    return ObservationPacket(**row)


class Retrieval:
    def __init__(self, path):
        self.db = sqlite3.connect(path.resolve().as_uri()+"?mode=ro", uri=True)
        self.page_rows = [(d, e, tuple(json.loads(p))) for d, e, p in
                          self.db.execute("SELECT DISTINCT domain,episode,page FROM structure")]

    def plan(self, question, domain, allowed):
        pages = tuple(sorted({p for d, e, p in self.page_rows if d == domain and e in allowed and p}))
        return plan_pages(question, pages) if pages else None

    def search(self, question, domain, allowed, channel, page_plan):
        if channel not in ("structure", "relations"):
            raise ValueError("unregistered index channel")
        words = terms(question)
        if not words:
            return []
        query = " OR ".join('"'+w+'"' for w in words)
        pages = set(page_plan.pages) if page_plan is not None and not page_plan.fallback else None
        cursor = self.db.execute(f"SELECT payload,episode,view_id,page,rank FROM {channel} WHERE {channel} MATCH ? AND domain=? ORDER BY rank", (query, domain))
        rows, cutoff = [], None
        try:
            for row in cursor:
                if cutoff is not None and row[-1] > cutoff:
                    break
                if row[1] not in allowed or pages is not None and tuple(json.loads(row[3])) not in pages:
                    continue
                rows.append(row)
                if len(rows) == 240:
                    cutoff = row[-1]
        finally:
            cursor.close()
        rows.sort(key=lambda r: (r[-1], r[2]))
        return [(packet(r[0]) if channel == "relations" else deserialize(r[0])) for r in rows[:240]]


def pack_relations(question, domain, views, tok):
    def render(chosen):
        groups = {}
        for i, view in enumerate(chosen):
            group = groups.setdefault(view.page, dict(page=list(view.page), records=[]))
            group["records"].append(dict(id=i, episode=view.episode,
                steps=sorted({o.step for o in view.occurrences}), relation=view.relation, observation=view.body))
        return prompt(question, domain, json.dumps(list(groups.values()), ensure_ascii=False))
    selected, contents = [], set()
    for view in views[:160]:
        # Do not spend the prompt repeatedly on the same page/body. The full
        # source identity of the chosen observation remains separately bound.
        signature = (view.page, view.body)
        if signature in contents:
            continue
        candidate = render(selected+[view])
        if len(tok.encode(candidate, add_special_tokens=False).ids) <= 6144:
            selected.append(view)
            contents.add(signature)
        if len(selected) == 96:
            break
    rendered = render(selected)
    n = len(tok.encode(rendered, add_special_tokens=False).ids)
    if n > 6144:
        raise ValueError("question alone exceeds budget")
    state = ContextState(tenant="lme-public", policy_revision="public-v1", clock=lambda: 1)
    for view in selected:
        state.put(view.canonical_node(tenant="lme-public", roles=("reader",), observed_at=1))
    snapshot = state.snapshot(AccessScope("lme-public", "evaluator", "public-v1", ("reader",)), at=1, known_at=1)
    if not state.verify_binding(snapshot, rendered, state.seal(snapshot, rendered)):
        raise ValueError("source/scope binding failed")
    return rendered, dict(input_tokens=n, prompt_sha256=hashlib.sha256(rendered.encode()).hexdigest(),
        sources=[dict(view_id=v.key, episode=v.episode, revision=v.source_revision,
                      view_sha256=hashlib.sha256(v.text.encode()).hexdigest()) for v in selected],
        scope_checked=True, sufficient_context_certified=False)


def compile_prompt(retrieval, question, domain, allowed, arm, tok):
    if arm not in ARMS:
        raise ValueError("unregistered arm")
    tick = time.perf_counter()
    plan = None if arm == "structure" else retrieval.plan(question, domain, allowed)
    views = retrieval.search(question, domain, allowed, "relations" if arm == "relations" else "structure", plan)
    search_seconds = time.perf_counter()-tick
    tick = time.perf_counter()
    rendered, meta = (pack_relations if arm == "relations" else pack)(question, domain, views, tok)
    # Normalize the old source-reference field name for the new evidence format.
    meta["sources"] = [{("view_id" if k == "key" else k): v for k, v in item.items()} for item in meta["sources"]]
    meta.update(retrieval_seconds=search_seconds, pack_seconds=time.perf_counter()-tick,
                page_plan=None if plan is None else asdict(plan))
    return rendered, meta


def prepare(data, previous, tokenizer, output):
    output = outside_repo(output)
    output.mkdir(parents=True, exist_ok=False)
    old = json.loads((previous/"registration.json").read_bytes())
    prepared = json.loads((previous/"prepared.json").read_bytes())
    if sha(previous/"structure.sqlite") != prepared["structure.sqlite"]:
        raise ValueError("frozen structural index changed")
    if any(sha(data/p) != h for p, h in old["dataset_hashes"].items()):
        raise ValueError("pinned data changed")
    models = {m: model_identity(m) for m in MODELS}
    if any(models[m]["digest"] != MODELS[m] for m in MODELS):
        raise ValueError("pinned local readers required")
    plan = dict(schema=1, sources=code_sources(), models=models, options=OPTIONS, arms=ARMS,
        ids=old["ids"], samples=72, calls=432, tokenizer_sha256=sha(tokenizer/"tokenizer.json"),
        dataset_hashes=old["dataset_hashes"], data_revision=old["data_revision"], source_revision=old["source_revision"],
        baseline_index_sha256=prepared["structure.sqlite"], scope_relative_floor=.5,
        title_anchor_max_document_fraction=.1, model_order=list(MODELS),
        limits=["Same 72 previously inspected development questions; not an independent holdout or learning qualification",
                "Scope scores are lexical heuristics, not probabilities or an authorization policy",
                "Relation treatment also changes packing, repeated-content handling and candidate/view limits",
                "Ordered cells are observed source order, not verified visual left/right or inferred column mappings",
                "No action execution, new weight training, frontier call, remote judge or activation"],
        timing="Regenerate and verify each registered prompt inside each timed request, then call the local reader; indexes warm; loading/build excluded",
        ollama=local_api("/api/version"))
    write_new(output/"registration.json", plan)
    all_questions = [json.loads(s) for s in (data/"questions.jsonl").read_text(encoding="utf-8").splitlines()]
    questions = [q for q in all_questions if q["id"] in set(plan["ids"])]
    haystacks = json.loads((data/"haystacks/lme_v2_small.json").read_bytes())
    allowed = {i for q in questions for i in haystacks[q["id"]]}
    build(data, previous, output, allowed).close()
    retrieval, tok = Retrieval(output/"memory.sqlite"), tokenizer_at(tokenizer)
    rows = []
    previous_inputs = {r["id"]: r for r in json.loads((previous/"inputs.json").read_bytes())}
    for q in questions:
        prompts, metadata = {}, {}
        for arm in ARMS:
            prompts[arm], metadata[arm] = compile_prompt(retrieval, q["question"], q["domain"], set(haystacks[q["id"]]), arm, tok)
        if prompts["structure"] != previous_inputs[q["id"]]["prompts"]["structure"]:
            raise ValueError("structural baseline changed; a separate protocol is required")
        order = list(ARMS)
        random.Random("lme-relations-arm-v1:"+q["id"]).shuffle(order)
        rows.append(dict(id=q["id"], domain=q["domain"], category=q["question_type"], question=q["question"],
                         allowed=haystacks[q["id"]], prompts=prompts, metadata=metadata, order=order))
        if len(rows) % 12 == 0:
            print(json.dumps(dict(prepared=len(rows))), flush=True)
    retrieval.db.close()
    write_new(output/"inputs.json", rows)
    write_new(output/"references.json", [{k: q[k] for k in ("id", "answer", "eval_function")} for q in questions])
    write_new(output/"prepared.json", {p: sha(output/p) for p in
                                      ("registration.json", "inputs.json", "references.json", "memory.sqlite", "index.json")})


def run(output, tokenizer):
    plan = json.loads((output/"registration.json").read_bytes())
    prepared = json.loads((output/"prepared.json").read_bytes())
    if plan["sources"] != code_sources() or plan["options"] != OPTIONS or sha(tokenizer/"tokenizer.json") != plan["tokenizer_sha256"]:
        raise ValueError("registered source/config/tokenizer changed")
    if any(sha(output/p) != h for p, h in prepared.items()):
        raise ValueError("registered input changed")
    if any(model_identity(m) != plan["models"][m] for m in MODELS):
        raise ValueError("registered local model changed")
    if local_api("/api/ps").get("models"):
        raise ValueError("local server occupied; wait for the existing workload")
    inputs = json.loads((output/"inputs.json").read_bytes())
    tok, retrieval = tokenizer_at(tokenizer), Retrieval(output/"memory.sqlite")
    for model in MODELS:
        label = model.replace(":", "-")
        warm = generate(model, "<|im_start|>user\nReply OK.<|im_end|>\n<|im_start|>assistant\n")
        write_new(output/f"{label}-warmup.json", dict(result=warm, resident=local_api("/api/ps")))
        count = 0
        with (output/f"{label}-generations.jsonl").open("x", encoding="utf-8", newline="\n") as stream:
            for row in inputs:
                for arm in row["order"]:
                    record = dict(id=row["id"], model=model, arm=arm, domain=row["domain"], category=row["category"],
                                  prediction="", output_tokens=0, error=None, truncated=False,
                                  **row["metadata"][arm])
                    record.update(generation_wall_seconds=0., retrieval_seconds=0., pack_seconds=0.)
                    request_tick = time.perf_counter()
                    try:
                        rendered, meta = compile_prompt(retrieval, row["question"], row["domain"], set(row["allowed"]), arm, tok)
                        if rendered != row["prompts"][arm] or meta["prompt_sha256"] != row["metadata"][arm]["prompt_sha256"]:
                            raise ValueError("live compilation differs from registered prompt")
                        record.update(meta)
                        tick = time.perf_counter()
                        try:
                            result = generate(model, rendered)
                        finally:
                            record["generation_wall_seconds"] = time.perf_counter()-tick
                        record.update(prediction=result.get("response", ""), output_tokens=result.get("eval_count", 0),
                            truncated=result.get("done_reason") == "length", model_timings={k: result.get(k) for k in
                                ("total_duration", "load_duration", "prompt_eval_duration", "eval_duration")})
                        if not result.get("done") or result.get("prompt_eval_count") != record["input_tokens"]:
                            raise ValueError("incomplete generation or token mismatch")
                    except Exception as exc:
                        record["error"] = f"{type(exc).__name__}: {exc}"
                    record["request_wall_seconds"] = time.perf_counter()-request_tick
                    stream.write(json.dumps(record, ensure_ascii=False)+"\n")
                    stream.flush()
                    count += 1
                if count % 36 == 0:
                    print(json.dumps(dict(model=model, calls=count)), flush=True)
        write_new(output/f"{label}-completed.json", dict(calls=count, sha256=sha(output/f"{label}-generations.jsonl")))
        local_api("/api/generate", dict(model=model, keep_alive=0))
    retrieval.db.close()


def evaluate(output, scorer):
    namespace = native_scorer(scorer)
    keys = {r["id"]: r for r in json.loads((output/"references.json").read_bytes())}
    all_rows, reports = [], {}
    for model in MODELS:
        label = model.replace(":", "-")
        path = output/f"{label}-generations.jsonl"
        completed = json.loads((output/f"{label}-completed.json").read_bytes())
        if sha(path) != completed["sha256"]:
            raise ValueError("generation log changed")
        rows = [json.loads(s) for s in path.read_text(encoding="utf-8").splitlines()]
        if len(rows) != 216 or {(r["id"], r["arm"]) for r in rows} != {(i, a) for i in keys for a in ARMS}:
            raise ValueError("incomplete paired experiment")
        rows = [r | dict(correct=not r["error"] and score(namespace, r["prediction"], keys[r["id"]])) for r in rows]
        reports[model] = {a: summary([r for r in rows if r["arm"] == a]) for a in ARMS}
        all_rows.extend(rows)
    write_new(output/"scores.json", all_rows)
    write_new(output/"summary.json", reports)
    print(json.dumps(reports))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "run", "evaluate"))
    for arg in ("data", "previous", "tokenizer", "output", "scorer"):
        parser.add_argument("--"+arg, type=Path, required=arg == "output")
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args.data, args.previous, args.tokenizer, args.output)
    elif args.command == "run":
        run(args.output, args.tokenizer)
    else:
        evaluate(args.output, args.scorer)

"""Paired CPU compilation timings with exact context/token/source parity.

No model requests, question answers, fitting, prompt changes or new quality scores.
"""

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import random
import sqlite3
import statistics
import time
from dataclasses import asdict
from pathlib import Path

from lme_batched_packing import pack_fast
from lme_relations import ARMS, Retrieval, code_sources, pack, pack_relations
from ruler_local_eval import tokenizer_at
from ruler_native import outside_repo, sha, write_new

ROOT = Path(__file__).resolve().parents[1]


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()


def summarize(rows):
    groups = {}
    for arm in ARMS:
        groups[arm] = {}
        for method in ("baseline", "batched"):
            measured = [r[method] for r in rows if r["arm"] == arm]
            values = [r["wall_seconds"] for r in measured]
            groups[arm][method] = dict(n=len(values), total_seconds=sum(values), median_seconds=statistics.median(values),
                p95_seconds=statistics.quantiles(values, n=100, method="inclusive")[94],
                total_process_cpu_seconds=sum(r["cpu_seconds"] for r in measured),
                median_search_seconds=statistics.median(r["search_seconds"] for r in measured),
                median_pack_seconds=statistics.median(r["pack_seconds"] for r in measured))
    return dict(groups=groups, pairs=len(rows), all_equal=all(r["equal"] for r in rows), model_calls=0,
                qualification="Warm CPU context compilation only; no model generation, whole-request speedup, new task-quality score or edge deployment qualification")


def compile_context(search, row, arm, tok, fast):
    started, cpu = time.perf_counter(), time.process_time()
    plan = None if arm == "structure" else search.plan(row["question"], row["domain"], set(row["allowed"]))
    views = search.search(row["question"], row["domain"], set(row["allowed"]),
                          "relations" if arm == "relations" else "structure", plan)
    search_seconds = time.perf_counter()-started
    tick = time.perf_counter()
    if fast:
        rendered, meta = pack_fast(row["question"], row["domain"], views, arm, tok)
    else:
        rendered, meta = (pack_relations if arm == "relations" else pack)(row["question"], row["domain"], views, tok)
        meta["sources"] = [{("view_id" if k == "key" else k): v for k, v in item.items()} for item in meta["sources"]]
    pack_seconds = time.perf_counter()-tick
    elapsed, consumed = time.perf_counter()-started, time.process_time()-cpu
    return rendered, meta, views, dict(wall_seconds=elapsed, cpu_seconds=consumed,
                                      search_seconds=search_seconds, pack_seconds=pack_seconds)


def canaries(tok):
    fragments = ("", "plain text", "\r\n\t  ", "cafe\u0301", "\u212b", "\u092d\u093e\u0930\u0924", "\u4e2d\u6587", "\U0001f642",
                 "<|im_start|>user\n", "<|im_end|>", "<tool_call>", "</tool_call>", "a\\\"'[]{}:;", "1234567890",
                 "\u200d\ufe0f", "a"*2048, "<|fim_prefix|>")
    rng = random.Random(8173)
    texts = ["".join(rng.choices(fragments, k=1+i%17)) for i in range(170)]
    values = tok.encode_batch_fast(texts, add_special_tokens=False)
    records = []
    for text, encoded in zip(texts, values):
        normal = tok.encode(text, add_special_tokens=False).ids
        if normal != encoded.ids:
            raise ValueError("native fast tokenizer ID mismatch")
        records.append(dict(text_sha256=hashlib.sha256(text.encode()).hexdigest(), tokens=len(normal),
                            ids_sha256=digest(normal), equal=True))
    return records


def run(study, tokenizer, output):
    if os.environ.get("RAYON_NUM_THREADS") != "8" or os.environ.get("TOKENIZERS_PARALLELISM") != "true":
        raise ValueError("Register the specified eight-worker native tokenizer configuration")
    output = outside_repo(output)
    output.mkdir(parents=True, exist_ok=False)
    original = json.loads((study/"registration.json").read_bytes())
    prepared = json.loads((study/"prepared.json").read_bytes())
    for name in ("inputs.json", "memory.sqlite"):
        if sha(study/name) != prepared[name]:
            raise ValueError("frozen study inputs changed")
    if sha(tokenizer/"tokenizer.json") != original["tokenizer_sha256"]:
        raise ValueError("frozen tokenizer required")
    rows = json.loads((study/"inputs.json").read_bytes())
    if len(rows) != 72 or {r["id"] for r in rows} != set(original["ids"]):
        raise ValueError("complete registered question set required")
    source_map = code_sources() | {"experiments/"+n: sha(ROOT/"experiments"/n)
                                  for n in ("lme_batched_packing.py", "profile_batched_packing.py")}
    write_new(output/"registration.json", dict(sources=source_map, ids=original["ids"], arms=ARMS,
        repeats=2, pairs=432, batch_size=8, tokenizer_sha256=original["tokenizer_sha256"],
        study_registration_sha256=sha(study/"registration.json"), inputs_sha256=prepared["inputs.json"],
        index_sha256=prepared["memory.sqlite"], python=platform.python_version(), platform=platform.platform(),
        tokenizers=importlib.metadata.version("tokenizers"), sqlite=sqlite3.sqlite_version,
        workers=8, design="Two repeats of 72 questions x 3 arms; hash-shuffled case and paired-method order; shared warmed tokenizer/read-only index",
        includes="Page planning, unchanged lookup/deserialization, complete prompt packing, full native final count, source/scope binding",
        excludes="Tokenizer/index initialization, canaries, comparison/evidence hashing, model loading and generation",
        labels_used=False, model_calls=0))
    tick = time.perf_counter()
    tok = tokenizer_at(tokenizer)
    search = Retrieval(study/"memory.sqlite")
    write_new(output/"initialization.json", dict(seconds=time.perf_counter()-tick))
    write_new(output/"tokenizer-canaries.json", canaries(tok))
    cases = [(repeat, row, arm) for repeat in range(2) for row in rows for arm in ARMS]
    random.Random(48103).shuffle(cases)
    records = []
    try:
        with (output/"records.jsonl").open("x", encoding="utf-8", newline="\n") as stream:
            for repeat, row, arm in cases:
                order = ["baseline", "batched"]
                random.Random(f"batch-pair-v1:{repeat}:{row['id']}:{arm}").shuffle(order)
                observed = {method: compile_context(search, row, arm, tok, method == "batched") for method in order}
                baseline, batched = observed["baseline"], observed["batched"]
                equal = (baseline[:3] == batched[:3] and baseline[0] == row["prompts"][arm])
                record = dict(id=row["id"], arm=arm, repeat=repeat, equal=equal, order=order,
                    baseline=baseline[3], batched=batched[3], input_tokens=baseline[1]["input_tokens"],
                    prompt_sha256=baseline[1]["prompt_sha256"], ranked_views_sha256=digest([asdict(v) for v in baseline[2]]),
                    selected_sources_sha256=digest(baseline[1]["sources"]), selected=len(baseline[1]["sources"]))
                stream.write(json.dumps(record)+"\n")
                stream.flush()
                records.append(record)
                if not equal:
                    raise ValueError("Retained mismatch: no substitution permitted")
                if len(records) % 36 == 0:
                    print(json.dumps(dict(pairs=len(records), all_equal=True)), flush=True)
    finally:
        search.db.close()
    write_new(output/"summary.json", summarize(records))
    write_new(output/"completion.json", dict(pairs=len(records), records_sha256=sha(output/"records.jsonl")))
    print(json.dumps(summarize(records)), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("study", "tokenizer", "output"):
        parser.add_argument("--"+name, type=Path, required=True)
    args = parser.parse_args()
    run(args.study, args.tokenizer, args.output)

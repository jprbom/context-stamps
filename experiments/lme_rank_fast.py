"""Exact tie-preserving FTS ranking optimization; no reader or policy changes.

The original study remains frozen in lme_memory.py. SQLite can use its native
rank ordering here. All boundary ties are collected before the original key
tie-break is applied, preserving the original ordered top 240 fragments.
https://sqlite.org/fts5.html#sorting_by_auxiliary_function_results
"""

import argparse
import hashlib
import json
import random
import sqlite3
import statistics
import time
from pathlib import Path

from lme_memory import ranked, terms
from ruler_native import outside_repo, sha, write_new

from context_stamps.trajectory import TraceFragment


def ranked_fast(db, question, domain, channel, allowed):
    words = terms(question)
    if not words:
        return []
    query = " OR ".join('"'+w+'"' for w in words)
    cursor = db.execute("SELECT body,key,episode,source,channel,steps,rank FROM memory WHERE memory MATCH ? AND domain=? AND channel=? ORDER BY rank", (query, domain, channel))
    rows, cutoff = [], None
    try:
        for row in cursor:
            if cutoff is not None and row[-1] > cutoff:
                break
            rows.append(row)
            if len(rows) == 240:
                cutoff = row[-1]
    finally:
        cursor.close()
    rows.sort(key=lambda r: (r[-1], r[1]))
    return [TraceFragment(r[1], r[2], r[3], r[4], tuple(json.loads(r[5])), r[0])
            for r in rows[:240] if r[2] in allowed]


def profile(data, run, output):
    output = outside_repo(output)
    output.mkdir(parents=True, exist_ok=False)
    plan = json.loads((run/"registration.json").read_bytes())
    prepared = json.loads((run/"prepared.json").read_bytes())
    if sha(run/"memory.sqlite") != prepared["index_sha256"]:
        raise ValueError("Frozen index changed")
    if sha(data/"questions.jsonl") != plan["dataset_hashes"]["questions.jsonl"]:
        raise ValueError("Questions changed")
    if sha(data/"haystacks/lme_v2_small.json") != plan["dataset_hashes"]["haystacks/lme_v2_small.json"]:
        raise ValueError("Haystack scope changed")
    ids = set(plan["train_ids"] + plan["holdout_ids"])
    questions = [{k: q[k] for k in ("id", "domain", "question")} for q in
                 (json.loads(s) for s in (data/"questions.jsonl").read_text(encoding="utf-8").splitlines()) if q["id"] in ids]
    haystacks = json.loads((data/"haystacks/lme_v2_small.json").read_bytes())
    write_new(output/"registration.json", dict(index_sha256=prepared["index_sha256"], sqlite=sqlite3.sqlite_version,
                                              original_source_sha256=sha(Path(__file__).with_name("lme_memory.py")),
                                              optimized_source_sha256=sha(__file__), questions=len(questions),
                                              lookups=len(questions)*3, design="One paired lookup per question/channel; hash-randomized order; shared warm connection"))
    db = sqlite3.connect(run/"memory.sqlite")
    results = []
    with (output/"records.jsonl").open("x", encoding="utf-8", newline="\n") as stream:
        for q in questions:
            for channel in ("state", "change", "path"):
                order = ["original", "optimized"]
                random.Random("lme-speed-v1:"+q["id"]+channel).shuffle(order)
                timings, views = {}, {}
                for arm in order:
                    tick = time.perf_counter()
                    views[arm] = (ranked if arm == "original" else ranked_fast)(db, q["question"], q["domain"], channel, set(haystacks[q["id"]]))
                    timings[arm] = time.perf_counter()-tick
                equal = views["original"] == views["optimized"]
                payload = json.dumps([v.__dict__ for v in views["original"]], sort_keys=True).encode()
                row = dict(id=q["id"], domain=q["domain"], channel=channel, seconds=timings, equal=equal,
                           count=len(views["original"]), returned_views_sha256=hashlib.sha256(payload).hexdigest())
                stream.write(json.dumps(row)+"\n")
                stream.flush()
                results.append(row)
                if not equal:
                    raise ValueError("Optimized ranking differs; retained failure, no substitution allowed")
            if len(results) % 75 == 0:
                print(json.dumps(dict(lookups=len(results), all_equal=all(r["equal"] for r in results))), flush=True)
    db.close()
    write_new(output/"summary.json", dict(lookups=len(results), all_equal=all(r["equal"] for r in results),
        arms={a: dict(total_seconds=sum(r["seconds"][a] for r in results),
                       median_seconds=statistics.median(r["seconds"][a] for r in results),
                       p95_seconds=sorted(r["seconds"][a] for r in results)[int(.95*(len(results)-1))]) for a in order},
        records_sha256=sha(output/"records.jsonl"), limitations="Warm local search only; no model/whole-request speedup claim"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("data", "run", "output"):
        parser.add_argument("--"+name, required=True, type=Path)
    args = parser.parse_args()
    profile(args.data, args.run, args.output)

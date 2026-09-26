"""Replay retained lookup aggregates and verify pinned optimization identities."""

import hashlib
import json
import math
import statistics
from pathlib import Path

from source_evidence import verify_sources

ROOT = Path(__file__).resolve().parents[1]


def verify(path):
    registration = json.loads((path/"registration.json").read_bytes())
    summary = json.loads((path/"summary.json").read_bytes())
    raw = (path/"records.jsonl").read_bytes()
    rows = [json.loads(s) for s in raw.splitlines()]
    if (hashlib.sha256(raw).hexdigest() != summary["records_sha256"] or len(rows) != 882
            or len({(r["id"], r["channel"]) for r in rows}) != 882 or not all(r["equal"] is True for r in rows)):
        raise ValueError("Lookup comparison changed or incomplete")
    ids = {r["id"] for r in rows}
    if len(ids) != 294 or {(r["id"], r["channel"]) for r in rows} != {(i, c) for i in ids for c in ("state", "change", "path")}:
        raise ValueError("Incomplete query/channel population")
    for r in rows:
        for value in r["seconds"].values():
            if not math.isfinite(value) or value < 0:
                raise ValueError("Invalid measured time")
    expected = {a: dict(total_seconds=sum(r["seconds"][a] for r in rows),
                        median_seconds=statistics.median(r["seconds"][a] for r in rows),
                        p95_seconds=sorted(r["seconds"][a] for r in rows)[int(.95*(len(rows)-1))])
                for a in ("original", "optimized")}
    if summary["arms"] != expected or summary["lookups"] != 882 or not summary["all_equal"]:
        raise ValueError("Search aggregate mismatch")
    pins = {"experiments/lme_memory.py": registration["original_source_sha256"],
            "experiments/lme_rank_fast.py": registration["optimized_source_sha256"]}
    if (path/"selection.json").exists():
        selection = json.loads((path/"selection.json").read_bytes())
        pins["experiments/lme_rank_selective.py"] = selection["source_sha256"]
        if selection["parent_source_sha256"] != registration["optimized_source_sha256"]:
            raise ValueError("Selection parent mismatch")
    verify_sources(pins)
    return dict(lookups=882, all_equal=True, measurements_replayed=True)


if __name__ == "__main__":
    base = ROOT/"evidence/lme-memory-v1"
    print(json.dumps({name: verify(base/name) for name in ("search", "search-selective")}))

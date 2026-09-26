"""Check every projected occurrence against original public source lines.

This is an independent reconstruction of the saved spans, not another call to
structured_views. It proves projection/coverage, never semantic sufficiency.
"""

import argparse
import collections
import json
import re
import sqlite3
from pathlib import Path

from lme_memory import episode
from lme_structure import deserialize
from ruler_native import sha, write_new


def verify(data, run):
    prepared = json.loads((run/"prepared.json").read_bytes())
    plan = json.loads((run/"registration.json").read_bytes())
    if sha(run/"structure.sqlite") != prepared["structure.sqlite"] or sha(data/"trajectories.jsonl") != plan["dataset_hashes"]["trajectories.jsonl"]:
        raise ValueError("registered index/source changed")
    by_episode = collections.defaultdict(list)
    db = sqlite3.connect((run/"structure.sqlite").resolve().as_uri()+"?mode=ro", uri=True)
    for payload, key in db.execute("SELECT payload,episode FROM memory"):
        by_episode[key].append(payload)
    db.close()
    counts = collections.Counter()
    with (data/"trajectories.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            if row["id"] not in by_episode:
                continue
            ep = episode(row)
            revision = ep.revision
            parsed, coverage = {}, {}
            for step in ep.steps:
                nodes, stack = {}, []
                for number, raw in enumerate(step.observation.splitlines(), 1):
                    if not raw.strip():
                        continue
                    stripped = raw.lstrip(" \t")
                    indentation = len(raw[:len(raw)-len(stripped)].expandtabs(4))
                    while stack and stack[-1][0] >= indentation:
                        stack.pop()
                    text = re.sub(r"^\[[A-Za-z0-9_-]{1,128}\] ", "", stripped, count=1)
                    nodes[number] = (tuple(s[1] for s in stack), text)
                    stack.append((indentation, number))
                parsed[step.index] = (step.location, nodes)
                coverage[step.index] = set()
                counts["observations"] += 1
                counts["nonblank_source_lines"] += len(nodes)
            for payload in by_episode.pop(ep.key):
                view = deserialize(payload)
                if not hasattr(view, "occurrences"):
                    raise ValueError("projection verification requires no raw fallbacks")
                if view.source_revision != revision:
                    raise ValueError("episode binding mismatch")
                for span in view.occurrences:
                    location, nodes = parsed[span.step]
                    if location != view.location or nodes[span.first][0] != span.ancestors or span.last not in nodes:
                        raise ValueError("source location/ancestry mismatch")
                    indices = list(span.ancestors)+[i for i in nodes if span.first <= i <= span.last]
                    expected = "\n".join("  "*len(nodes[i][0])+nodes[i][1] for i in indices)
                    if expected != view.body:
                        raise ValueError("projection differs from recorded source")
                    coverage[span.step].update(indices)
                    counts["occurrences"] += 1
                counts["views"] += 1
            for step, (_, nodes) in parsed.items():
                if coverage[step] != set(nodes):
                    raise ValueError("source line omitted or invented")
            counts["episodes"] += 1
    if by_episode:
        raise ValueError("missing source episode")
    return dict(checks=dict(counts), index_sha256=prepared["structure.sqlite"],
                source_sha256=plan["dataset_hashes"]["trajectories.jsonl"], all_occurrences_reconstructed=True,
                all_nonblank_lines_covered=True, semantic_sufficiency=False, verifier_sha256=sha(__file__))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ("data", "run", "output"):
        parser.add_argument("--"+arg, type=Path, required=True)
    args = parser.parse_args()
    result = verify(args.data, args.run)
    write_new(args.output, result)
    print(json.dumps(result))

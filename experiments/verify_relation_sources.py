"""Reconstruct every retained relation from source lines without rerunning extraction."""

import argparse
import collections
import json
import re
import sqlite3
from pathlib import Path

from lme_memory import episode
from ruler_native import sha, write_new

QUOTED = r'''('(?:[^'\\]|\\.)*'|"(?:[^"\\]|\\.)*")'''
MEMBERS = {"row": {"columnheader", "rowheader", "gridcell", "cell"},
           "combobox": {"option"}, "listbox": {"option"}, "tablist": {"tab"},
           "menu": {"menuitem", "menuitemcheckbox", "menuitemradio"}}


def parse(observation):
    nodes, stack = {}, []
    for number, raw in enumerate(observation.splitlines(), 1):
        if not raw.strip():
            continue
        stripped = raw.lstrip(" \t")
        indent = len(raw[:len(raw)-len(stripped)].expandtabs(4))
        while stack and stack[-1][0] >= indent:
            stack.pop()
        text = re.sub(r"^\[[A-Za-z0-9_-]{1,128}\] ", "", stripped, count=1)
        match = re.fullmatch(r"([A-Za-z][A-Za-z0-9_]{0,63})\s+"+QUOTED+r"(.*)", text.strip())
        role, name, attrs = match.groups() if match else ("opaque", "", text.strip())
        parents = tuple(s[1] for s in stack)
        scope = tuple(i for i in parents if nodes[i]["role"] == "RootWebArea")
        if role == "RootWebArea":
            scope += (number,)
        nodes[number] = dict(text=text, role=role, name=name, attrs=attrs, parents=parents, scope=scope)
        stack.append((indent, number))
    return nodes


def check_occurrence(view, occurrence, nodes):
    refs = set(occurrence["lines"])
    if not refs or not refs <= nodes.keys():
        raise ValueError("missing source reference")
    body = view["body"]
    owner = None
    if body.startswith("Recorded ancestor at source line "):
        first, body = body.split("\n", 1)
        match = re.fullmatch(r"Recorded ancestor at source line (\d+): (.*)", first)
        if match is None:
            raise ValueError("invalid container descriptor")
        owner = int(match[1])
        if owner not in refs or nodes[owner]["text"] != match[2]:
            raise ValueError("container text/source mismatch")
    candidates = [i for i in sorted(refs) if i != owner and nodes[i]["role"] != "RootWebArea"]
    if not candidates:
        raise ValueError("no observed element")
    head, *members = candidates
    node = nodes[head]
    if owner is not None and owner not in node["parents"]:
        raise ValueError("declared container is not an ancestor")
    if tuple(view["page"]) != tuple(nodes[i]["name"] for i in node["scope"]):
        raise ValueError("declared page does not own the source element")
    if refs != set(node["scope"]) | {head} | set(members) | (set() if owner is None else {owner}):
        raise ValueError("unexpected source reference")
    if view["relation"] == "element":
        if members or body != node["text"]:
            raise ValueError("element text/source mismatch")
        return 0
    if view["relation"] != "ordered_members" or node["role"] not in MEMBERS or not members:
        raise ValueError("invalid ordered relation")
    for i in members:
        child = nodes[i]
        same_role = [p for p in child["parents"] if nodes[p]["role"] == node["role"]]
        if child["role"] not in MEMBERS[node["role"]] or not same_role or same_role[-1] != head:
            raise ValueError("member belongs to a different structural owner")
    heading = node["role"] + (" " + node["name"] if node["role"] != "row" else "") + node["attrs"]
    expected = heading+"\nObserved members in source order:\n"+"\n".join(nodes[i]["text"] for i in members)
    if body != expected:
        raise ValueError("ordered source text or order changed")
    return sum(nodes[i]["scope"] != node["scope"] for i in members)


def verify(data, run):
    plan = json.loads((run/"registration.json").read_bytes())
    prepared = json.loads((run/"prepared.json").read_bytes())
    if sha(data/"trajectories.jsonl") != plan["dataset_hashes"]["trajectories.jsonl"] or sha(run/"memory.sqlite") != prepared["memory.sqlite"]:
        raise ValueError("registered source/index changed")
    db = sqlite3.connect((run/"memory.sqlite").resolve().as_uri()+"?mode=ro", uri=True)
    cursor = iter(db.execute("SELECT payload,episode FROM relations ORDER BY rowid"))
    current = next(cursor, None)
    counts = collections.Counter()
    with (data/"trajectories.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            if current is None:
                break
            row = json.loads(line)
            if row["id"] != current[1]:
                continue
            ep = episode(row)
            source_revision = ep.revision
            parsed = {s.index: (s.location, parse(s.observation)) for s in ep.steps}
            while current is not None and current[1] == ep.key:
                view = json.loads(current[0])
                if view["source_revision"] != source_revision:
                    raise ValueError("episode identity mismatch")
                for occurrence in view["occurrences"]:
                    location, nodes = parsed[occurrence["step"]]
                    if location != view["location"]:
                        raise ValueError("source location mismatch")
                    counts["members_crossing_nested_page_boundaries"] += check_occurrence(view, occurrence, nodes)
                    counts["occurrences"] += 1
                counts[view["relation"]] += 1
                current = next(cursor, None)
            counts["episodes"] += 1
    db.close()
    if current is not None:
        raise ValueError("unmatched source episode")
    return dict(checks=dict(counts), all_retained_occurrences_verified=True, complete_source_coverage=False,
                interpretation="Checks retained text, source order, structural ancestor and page ownership; not visual layout, semantic sufficiency, causal effects or facts.",
                verifier_sha256=sha(__file__), index_sha256=prepared["memory.sqlite"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ("data", "run", "output"):
        parser.add_argument("--"+arg, type=Path, required=True)
    args = parser.parse_args()
    result = verify(args.data, args.run)
    write_new(args.output, result)
    print(json.dumps(result))

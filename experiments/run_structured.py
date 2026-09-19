"""Structured selection regression and prospective procedural validation; no fitting."""

import argparse
import hashlib
import itertools
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from context_stamps import Claim, ContextMemory, Requirement, content_digest, select_structured

ROOT = Path(__file__).resolve().parents[1]


def save(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8", newline="\n")


def fixture_claims(text):
    """Parse ONLY the published fictional numeric grammar; no gold fact/source IDs."""
    first = re.match(r"For (\w+), configure (\w+): (\d+) (\w+)", text)
    if not first:
        first = re.match(r"(\w+) (\w+) is (\d+) (\w+)", text)
    if not first:
        return []
    subject, attribute, value, unit = first.groups()
    result = [(subject, attribute, value + " " + unit)]
    for attribute, value, unit in re.findall(r"; (\w+) is (\d+) (\w+)", text):
        result.append((subject, attribute, value + " " + unit))
    return result


def oracle_feasible(case):
    current = [d for d in case["documents"] if not d["stale"]]
    for n in range(1, len(current) + 1):
        for chosen in itertools.combinations(current, n):
            if not set(case["required_facts"]) <= {f for d in chosen for f in d["facts"]}:
                continue
            text = "\n\n".join(
                json.dumps({"source": d["source"], "sha256": d["digest"]}, ensure_ascii=False)
                + "\n"
                + d["text"]
                for d in chosen
            )
            if len(text.encode("utf-8")) <= case["budget"]:
                return True
    return False


def regression():
    cases = [
        json.loads(line) for line in (ROOT / "evidence/synthetic-v1/fixtures.jsonl").read_text().splitlines()
    ]
    records = []
    for case in cases:
        if case["split"] not in {"test", "ood"}:
            continue
        match = re.fullmatch(r"What are the (\w+) and (\w+) settings for (\w+)\?", case["query"])
        first, second, subject = match.groups()
        with ContextMemory() as memory:
            claims, revisions = [], {}
            for doc in case["documents"]:
                row = memory.add(doc["text"], source=doc["source"])
                revisions[doc["source"]] = (
                    content_digest("changed/" + doc["text"]) if doc["stale"] else row["digest"]
                )
                claims.extend(
                    Claim(row["source"], row["digest"], *claim) for claim in fixture_claims(doc["text"])
                )
            result = select_structured(
                memory,
                case["query"],
                requirements=[Requirement(subject, first), Requirement(subject, second)],
                claims=claims,
                revisions=revisions,
                budget_bytes=case["budget"],
            )
            chosen = [d for d in case["documents"] if d["source"] in result.selected]
            facts = {f for d in chosen if not d["stale"] for f in d["facts"]}
            feasible = oracle_feasible(case)
            success = set(case["required_facts"]) <= facts
            records.append(
                {
                    "case": case["id"],
                    "split": case["split"],
                    "scenario": case["scenario"],
                    "feasible": feasible,
                    "complete": success,
                    "abstained": not result.text,
                    "correct_action": success if feasible else not result.text,
                    "stale_selected": any(d["stale"] for d in chosen),
                    "packet": result.to_dict(),
                }
            )
    return records


def prospective(seed):
    """New identifiers/order/values and explicit JSON inputs; not natural-language OOD."""
    rng = random.Random(seed)
    records, fixtures = [], []
    for scenario in ["complete", "missing", "budget", "conflict", "stale_metadata", "stale_source"]:
        for number in range(40):
            subject = f"node-{rng.getrandbits(40):010x}"
            attributes = rng.sample(
                ["batch_size", "window_ms", "retry_cap", "queue_depth", "sample_hz", "cache_ttl"], 3
            )
            values = {key: str(rng.randrange(1, 4096)) for key in attributes}
            docs = []
            for key in attributes:
                docs.append(
                    {"subject": subject, "values": {key: values[key]}, "padding": "x" * rng.randrange(0, 90)}
                )
            docs.append({"subject": subject + "-other", "values": values, "padding": ""})
            docs.append({"subject": subject, "values": {"irrelevant": "1"}, "padding": ""})
            docs.append({"subject": subject, "values": dict(docs[0]["values"]), "padding": "duplicate"})
            if scenario == "missing":
                docs = [d for d in docs if attributes[1] not in d["values"] or d["subject"] != subject]
            if scenario == "conflict":
                docs.append({"subject": subject, "values": {attributes[1]: "different"}, "padding": ""})
            rng.shuffle(docs)
            named = [(f"record-{rng.getrandbits(40):010x}", d) for d in docs]
            budget = 10 if scenario == "budget" else 1800
            fixtures.append(
                {
                    "case": f"{scenario}-{number}",
                    "scenario": scenario,
                    "subject": subject,
                    "attributes": attributes,
                    "records": named,
                    "budget": budget,
                }
            )
            with ContextMemory() as memory:
                claims, revisions = [], {}
                for source, doc in named:
                    row = memory.add(json.dumps(doc, sort_keys=True), source=source)
                    is_target = doc["subject"] == subject and attributes[1] in doc["values"]
                    revisions[source] = (
                        content_digest("new revision")
                        if scenario == "stale_source" and is_target
                        else row["digest"]
                    )
                    digest = (
                        content_digest("old revision")
                        if scenario == "stale_metadata" and is_target
                        else row["digest"]
                    )
                    claims.extend(
                        Claim(source, digest, doc["subject"], key, value)
                        for key, value in doc["values"].items()
                    )
                result = select_structured(
                    memory,
                    f"settings for {subject}",
                    requirements=[Requirement(subject, key) for key in attributes],
                    claims=claims,
                    revisions=revisions,
                    budget_bytes=budget,
                )
                observed = {
                    (doc["subject"], key, value)
                    for source, doc in named
                    if source in result.selected
                    for key, value in doc["values"].items()
                }
                complete = all((subject, key, values[key]) in observed for key in attributes)
                expected_answer = scenario == "complete"
                records.append(
                    {
                        "case": f"{scenario}-{number}",
                        "scenario": scenario,
                        "expected_answer": expected_answer,
                        "complete": complete,
                        "abstained": not result.text,
                        "correct_action": complete and result.status == "current"
                        if expected_answer
                        else not result.text,
                        "packet": result.to_dict(),
                    }
                )
    return fixtures, records


def run(out):
    out.mkdir(parents=True, exist_ok=True)
    protocol = json.loads((ROOT / "evidence/structured-v1/protocol.json").read_text())
    old = regression()
    fixtures, fresh = prospective(protocol["seed"])
    summary = []
    for dataset, records, key in [
        ("historical_regression", old, "split"),
        ("prospective_structured", fresh, "scenario"),
    ]:
        for group in sorted({r[key] for r in records}):
            chosen = [r for r in records if r[key] == group]
            summary.append(
                {
                    "dataset": dataset,
                    "group": group,
                    "n": len(chosen),
                    **{
                        metric: sum(bool(r[metric]) for r in chosen)
                        for metric in ["complete", "abstained", "correct_action"]
                    },
                    "feasible": sum(r.get("feasible", r.get("expected_answer", False)) for r in chosen),
                }
            )
    for name, data in [
        ("regression.json", old),
        ("prospective.json", fresh),
        ("fixtures.json", fixtures),
        ("summary.json", summary),
    ]:
        save(out / name, data)
    sources = ["context_stamps/requirements.py", "experiments/run_structured.py", "context_stamps/memory.py"]
    save(
        out / "manifest.json",
        {
            "protocol": protocol,
            "source_sha256": {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in sources},
            "historical_fixture_sha256": hashlib.sha256(
                (ROOT / "evidence/synthetic-v1/fixtures.jsonl").read_bytes()
            ).hexdigest(),
            "prospective_scenarios": dict(Counter(r["scenario"] for r in fresh)),
            "limitations": "schema/claim metadata is additional structure; no new model training; historical fixtures are regression only; fresh procedural validation is not independent natural-language or external-user validation",
        },
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=ROOT / "evidence/structured-v1")
    run(parser.parse_args().out)

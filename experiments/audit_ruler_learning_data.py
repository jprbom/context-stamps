"""Post-run public-input overlap audit; distinct rows are not IID sources."""

import argparse
import hashlib
import json
import re
from pathlib import Path

from ruler_context import frame
from ruler_local_eval import public_rows
from ruler_native import outside_repo, sha, write_new


def digest(text):
    return hashlib.sha256(text.encode()).hexdigest()


def audit(train, holdout, output):
    sets, counts = {}, {}
    for name, data, length in (("train", train, 4096), ("retention", holdout, 4096), ("adaptation", holdout, 16384)):
        rows = [r for r in public_rows(data) if r["length"] == length]
        questions, documents, full = set(), set(), set()
        qn = 0
        for row in rows:
            full.add(digest(row["question"]))
            if row["task"].startswith("qa_"):
                qn += 1
                view = frame(row["question"])
                query = view.suffix.split("Question:", 1)[1].rsplit("Answer:", 1)[0].strip()
                questions.add(digest(query))
                documents.update(digest(text.strip()) for text in re.split(r"(?:^|\n\n)Document \d+:\n", view.context)[1:])
        sets[name] = dict(questions=questions, documents=documents, full_inputs=full)
        counts[name] = dict(inputs=len(rows), qa_inputs=qn, unique_qa_questions=len(questions),
                            unique_qa_document_texts=len(documents))
    pairs = []
    for a, b in (("train", "retention"), ("train", "adaptation"), ("retention", "adaptation")):
        pairs.append(dict(left=a, right=b, overlaps={key: len(sets[a][key] & sets[b][key]) for key in sets[a]}))
    report = dict(schema=1, kind="post_run_diagnostic_not_a_preregistered_independence_test", counts=counts, pairs=pairs,
        inputs={"train_inventory": sha(train/"inventory.json"), "holdout_inventory": sha(holdout/"inventory.json")},
        sources={"experiments/audit_ruler_learning_data.py": sha(__file__)},
        caveat="QA source documents and synthetic templates/backgrounds can repeat. Distinct full-input hashes do not prove IID task clusters.")
    write_new(outside_repo(output), report)
    print(json.dumps(report))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("train", "holdout", "output"):
        parser.add_argument("--"+name, type=Path, required=True)
    args = parser.parse_args()
    audit(args.train, args.holdout, args.output)

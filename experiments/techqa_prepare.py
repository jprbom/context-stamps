"""Separate TechQA inputs from labels and audit the native development split.

Copyright (c) 2026 Prashant Jagtap. MIT License.
Data remains local under CDLA-Permissive-1.0. Split selection does not use labels.
"""

import argparse
import collections
import json
import statistics
from pathlib import Path

from ruler_native import outside_repo, sha, write_new
from techqa_context import grouped_split, public_input, terms
from techqa_native import canaries, load_native


def prepare(data, output):
    data, output = outside_repo(data), outside_repo(output)
    output.mkdir(parents=True, exist_ok=False)
    write_new(output/"registration.json", dict(data_complete_sha256=sha(data/"complete.json"),
                                               source_hashes={p.name: sha(p) for p in (Path(__file__), Path(__file__).with_name("techqa_context.py"), Path(__file__).with_name("techqa_native.py"))},
                                               split_seed=83, fit_count=400, calibration_count="remaining eligible groups", development_count=310,
                                               inputs_allowlisted=True, labels_used_for_selection=False))
    complete = json.loads((data/"complete.json").read_bytes())
    for member in complete["selected"].values():
        if sha(data/member["file"]) != member["sha256"]:
            raise ValueError("downloaded dataset changed")
    docs = json.loads((data/"training_dev_technotes.json").read_bytes())
    train = json.loads((data/"training_Q_A.json").read_bytes())
    dev = json.loads((data/"dev_Q_A.json").read_bytes())
    if len(train) != 600 or len(dev) != 310:
        raise ValueError("native question counts changed")
    projected = {row["QUESTION_ID"]: public_input(row) for row in train+dev}
    if len(projected) != 910:
        raise ValueError("duplicate native question ID")
    # Group by question text, remove train/dev duplicates using inputs only.
    splits, duplicate_audit = grouped_split([projected[row["QUESTION_ID"]] for row in train],
                                            [projected[row["QUESTION_ID"]] for row in dev])
    all_rows = {row["QUESTION_ID"]: row for row in train+dev}
    doc_sets = {name: set(doc for qid in ids for doc in projected[qid]["doc_ids"])
                for name, ids in splits.items()}
    question_sets = {name: set(" ".join(terms(projected[qid]["question"])) for qid in ids)
                     for name, ids in splits.items()}
    missing = sorted(set.union(*doc_sets.values())-set(docs))
    if missing:
        raise ValueError(f"missing candidate documents: {len(missing)}")
    stats = {}
    for name, ids in splits.items():
        stats[name] = dict(questions=len(ids), candidate_documents=len(doc_sets[name]),
                          candidate_counts=dict(collections.Counter(len(projected[qid]["doc_ids"]) for qid in ids)),
                          maximum_question_characters=max(len(projected[qid]["question"]) for qid in ids))
    overlap = {}
    for left, right in (("fit", "calibration"), ("fit", "development"), ("calibration", "development")):
        overlap[left+"__"+right] = dict(candidate_documents=len(doc_sets[left] & doc_sets[right]),
                                        normalized_questions=len(question_sets[left] & question_sets[right]))
        if overlap[left+"__"+right]["normalized_questions"]:
            raise ValueError("cross-split duplicate question; register a revised split before running")
    # Label validation is separate from selection. Do not display development
    # answer statistics before predictions. Aggregate integrity failures only.
    keys, label_audit = {}, []
    train_ids = {row["QUESTION_ID"] for row in train}
    for qid, row in all_rows.items():
        if row["ANSWERABLE"] not in ("Y", "N"):
            raise ValueError("invalid answerability annotation")
        key = {field: row[field] for field in ("ANSWERABLE", "DOCUMENT", "START_OFFSET", "END_OFFSET", "ANSWER")}
        if row["ANSWERABLE"] == "Y":
            doc_id = row["DOCUMENT"]
            # The pinned JSON represents numeric character offsets as strings,
            # which the native evaluator converts with int(). Keep raw keys.
            offsets = [row[field] for field in ("START_OFFSET", "END_OFFSET")]
            if any(type(value) not in (str, int) or not str(value).isascii() or not str(value).isdecimal()
                   or len(str(value)) > 8 for value in offsets):
                raise ValueError("bounded decimal native offsets required")
            start, end = map(int, offsets)
            if doc_id not in projected[qid]["doc_ids"] or not 0 <= start < end <= len(docs[doc_id]["text"]):
                raise ValueError("native label span outside provided source")
            if docs[doc_id]["text"][start:end] != row["ANSWER"]:
                label_audit.append(dict(id=qid, issue="native ANSWER differs from native character span",
                                       split="training" if qid in train_ids else "development",
                                       action="preserve original annotation; native offsets remain primary metric"))
                if qid in train_ids:
                    raise ValueError("unverified training answer span")
        keys[qid] = key
    lengths = [len(keys[qid]["ANSWER"]) for qid in splits["fit"] if keys[qid]["ANSWERABLE"] == "Y"]
    fit_stats = dict(answerable=len(lengths), answer_chars_min=min(lengths),
                     answer_chars_median=statistics.median(lengths), answer_chars_max=max(lengths),
                     answer_chars_above4096=sum(n > 4096 for n in lengths))
    native_rows = canaries(load_native(data/"techqa_evaluation.py"))
    write_new(output/"native-canaries.json", native_rows)
    write_new(output/"label-integrity-audit.json", label_audit)
    for name, ids in splits.items():
        write_new(output/(name+".inputs.json"), [projected[qid] for qid in ids])
        write_new(output/(name+".keys.json"), {qid: keys[qid] for qid in ids})
    files = {p.name: sha(p) for p in output.iterdir() if p.is_file()}
    write_new(output/"manifest.json", dict(
        data=str(data), data_complete_sha256=sha(data/"complete.json"),
        documents_sha256=sha(data/"training_dev_technotes.json"), files=files,
        split_rule="group normalized questions; withhold native training questions matching dev; hash(techqa-group-split-83 + NUL + group hash); fill 400 fit with whole groups, remaining calibration; retain native 310 dev",
        statistics=stats, overlap=overlap, fit_label_statistics=fit_stats,
        duplicate_audit=duplicate_audit,
        label_integrity=dict(answer_text_span_mismatches=len(label_audit), development_keys_modified=False,
                             numeric_offset_strings_preserved=True, primary_metric="native character offsets"),
        input_corrections=dict(body_field="QUESTION_TEXT (README says QUESTION_BODY)",
                               repeated_candidate_id_cases=sum(len(set(row["DOC_IDS"])) != len(row["DOC_IDS"]) for row in train+dev),
                               action="deduplicate candidate references only; retain every question"),
        source_hashes={p.name: sha(p) for p in (Path(__file__), Path(__file__).with_name("techqa_context.py"),
                                                 Path(__file__).with_name("techqa_native.py"))},
        claim_scope="native public development split; shared candidate documents; unknown base-model pretraining overlap",
        data_license="CDLA-Permissive-1.0", labels_used_for_selection=False))
    print(json.dumps(dict(output=str(output), statistics=stats, overlap=overlap, fit_label_statistics=fit_stats)), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    prepare(args.data, args.output)

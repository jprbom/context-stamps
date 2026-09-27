"""Pinned native TechQA scoring, with oracle threshold fields kept separate.

Copyright (c) 2026 Prashant Jagtap. MIT License.
The separately downloaded IBM evaluator retains its Apache-2.0 license.
"""

import hashlib
import importlib.util
from pathlib import Path

NATIVE_SHA = "8d627929be653be74b6b989be314308b1407d961024ccde18676e24201a78851"


def load_native(path):
    path = Path(path)
    if hashlib.sha256(path.read_bytes()).hexdigest() != NATIVE_SHA:
        raise ValueError("reviewed native scorer bytes required")
    spec = importlib.util.spec_from_file_location("pinned_techqa", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.OPTS.top_k = 1
    return module


def abstention():
    return dict(doc_id="", score=0.0, start_offset=-1, end_offset=-1)


def metrics(native, keys, predictions):
    """Score explicit final decisions; never choose a threshold on these keys."""
    if set(keys) != set(predictions):
        raise ValueError("exact prediction/key coverage required")
    for prediction in predictions.values():
        if (set(prediction) != {"doc_id", "score", "start_offset", "end_offset"}
                or type(prediction["doc_id"]) is not str
                or type(prediction["start_offset"]) is not int
                or type(prediction["end_offset"]) is not int
                or prediction["score"] not in (0.0, 1.0)):
            raise ValueError("bounded final native span required")
        if prediction["doc_id"]:
            if not 0 <= prediction["start_offset"] < prediction["end_offset"]:
                raise ValueError("positive character span required")
        elif prediction != abstention():
            raise ValueError("canonical explicit abstention required")
    preds = {key: [value] for key, value in predictions.items()}
    scores = native.evaluate(preds, keys, threshold=-1.0)
    # Native Best_* fits on evaluation answers. Never report it as a achieved score.
    scores = {key: value for key, value in scores.items() if not key.startswith("Best_")}
    flags = native.make_qid_to_has_ans(keys)
    raw, _, _ = native.get_raw_scores(keys, preds, flags, 1)
    positive = [key for key in keys if flags[key]]
    negative = [key for key in keys if not flags[key]]
    answered = [key for key in keys if predictions[key]["doc_id"]]
    scores.update(
        positive_count=len(positive), negative_count=len(negative), answered_count=len(answered),
        positive_f1=sum(raw[key][0] for key in positive)/len(positive) if positive else None,
        false_positive_count=sum(bool(predictions[key]["doc_id"]) for key in negative),
        abstained_positive_count=sum(not predictions[key]["doc_id"] for key in positive),
        answered_f1=sum(raw[key][0] for key in answered)/len(answered) if answered else None,
    )
    return scores, {key: raw[key][0] for key in keys}


def canaries(native):
    keys = {
        "yes": dict(ANSWERABLE="Y", DOCUMENT="doc", START_OFFSET=10, END_OFFSET=20),
        "no": dict(ANSWERABLE="N", DOCUMENT="-", START_OFFSET="-", END_OFFSET="-"),
    }
    def pred(doc="doc", start=10, end=20):
        return dict(doc_id=doc, score=1.0, start_offset=start, end_offset=end)
    cases = [
        ("exact", {"yes": pred(), "no": abstention()}, 100.0),
        ("wrong_document", {"yes": pred("other"), "no": abstention()}, 50.0),
        ("partial_span", {"yes": pred(end=15), "no": abstention()}, 100*5/6),
        ("no_overlap", {"yes": pred(start=30, end=40), "no": abstention()}, 50.0),
        ("abstain_all", {"yes": abstention(), "no": abstention()}, 50.0),
        ("answer_negative", {"yes": pred(), "no": pred()}, 50.0),
    ]
    rows = []
    for name, predictions, expected in cases:
        result, _ = metrics(native, keys, predictions)
        if abs(result["QA_F1"]-expected) > 1e-10:
            raise ValueError(f"native canary failed: {name}")
        rows.append(dict(name=name, expected=expected, observed=result["QA_F1"]))
    # The native missing-prediction behavior differs from explicit abstention.
    scores, _, _ = native.get_raw_scores(keys, {}, native.make_qid_to_has_ans(keys), 1)
    if scores != {"yes": [0], "no": [0]}:
        raise ValueError("native missing behavior changed")
    rows.append(dict(name="missing_is_zero", passed=True))
    return rows

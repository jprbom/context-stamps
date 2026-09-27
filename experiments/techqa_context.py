"""Input-only TechQA windows with exact character-offset provenance.

Copyright (c) 2026 Prashant Jagtap. MIT License.
No answerability, gold document, answer text or gold offset is accepted here.
"""

import collections
import hashlib
import math
import re


def digest(text):
    return hashlib.sha256(text.encode()).hexdigest()


def terms(text):
    return re.findall(r"\w+", text.casefold())


def grouped_split(train, development, *, fit_count=400):
    """Keep normalized duplicate questions in one cohort, using inputs only.

    Every native development question is retained. Training questions matching
    development questions are withheld and explicitly inventoried.
    """
    def group(item):
        return digest(" ".join(terms(item["question"])))
    dev_groups = collections.defaultdict(list)
    for item in development:
        dev_groups[group(item)].append(item["id"])
    train_groups, withheld = collections.defaultdict(list), []
    for item in train:
        key = group(item)
        if key in dev_groups:
            withheld.append(dict(id=item["id"], question_hash=key, matching_development_ids=sorted(dev_groups[key])))
        else:
            train_groups[key].append(item["id"])
    fitting, calibration = [], []
    for key in sorted(train_groups, key=lambda value: digest("techqa-group-split-83\0"+value)):
        ids = sorted(train_groups[key])
        (fitting if len(fitting)+len(ids) <= fit_count else calibration).extend(ids)
    if len(fitting) != fit_count or not calibration:
        raise ValueError("insufficient complete groups for the registered split")
    return (dict(fit=fitting, calibration=calibration, development=sorted(item["id"] for item in development)),
            dict(withheld_training_questions=sorted(withheld, key=lambda row: row["id"]),
                 grouped_training_duplicates={key: sorted(ids) for key, ids in train_groups.items() if len(ids) > 1},
                 development_duplicate_groups={key: sorted(ids) for key, ids in dev_groups.items() if len(ids) > 1}))


def public_input(row):
    # Pinned files use QUESTION_TEXT; the archive README calls it QUESTION_BODY.
    if "QUESTION_TEXT" in row and "QUESTION_BODY" in row and row["QUESTION_TEXT"] != row["QUESTION_BODY"]:
        raise ValueError("conflicting question body aliases")
    body_key = "QUESTION_TEXT" if "QUESTION_TEXT" in row else "QUESTION_BODY"
    qid, title, body, ids = (row[key] for key in ("QUESTION_ID", "QUESTION_TITLE", body_key, "DOC_IDS"))
    if (not all(isinstance(value, str) for value in (qid, title, body))
            or not qid or len(title)+len(body) > 65536 or not (title+body).strip()
            or not isinstance(ids, list) or not 1 <= len(ids) <= 1000
            or not all(isinstance(value, str) and value for value in ids)):
        raise ValueError("bounded question and candidate IDs required")
    question = title if title.strip() == body.strip() else title+"\n"+body
    # Two native questions repeat a candidate ID. Preserve both questions while
    # avoiding duplicate retrieval weight for the same source document.
    return dict(id=qid, question=question, doc_ids=sorted(set(ids)))


def windows(item, documents, *, width=200, stride=150):
    if set(item) != {"id", "question", "doc_ids"} or not 1 <= stride <= width <= 1000:
        raise ValueError("only projected inputs and bounded windows allowed")
    result = []
    for doc_id in item["doc_ids"]:
        doc = documents[doc_id]
        text, title = doc["text"], doc["title"]
        if not isinstance(text, str) or not isinstance(title, str) or len(text) > 1024*1024:
            raise ValueError("bounded plaintext document required")
        positions = list(re.finditer(r"\S+", text))
        for first in range(0, len(positions), stride):
            last = min(first+width, len(positions))
            start, end = positions[first].start(), positions[last-1].end()
            result.append(dict(doc_id=doc_id, start=start, end=end, text=text[start:end], title=title))
            if last == len(positions):
                break
    return result


def rank_windows(question, candidates):
    """BM25 on candidate windows, with title terms included in ranking only."""
    counts = [collections.Counter(terms(row["title"]+" "+row["text"])) for row in candidates]
    lengths = [sum(count.values()) for count in counts]
    avg = sum(lengths)/max(1, len(lengths)) or 1
    df = collections.Counter(term for count in counts for term in count)
    query = set(terms(question))
    ranked = []
    for row, count, length in zip(candidates, counts, lengths):
        score = sum(math.log(1+(len(counts)-df[term]+0.5)/(df[term]+0.5))*count[term]*2.5 /
                    (count[term]+1.5*(0.25+0.75*length/avg)) for term in query if count[term])
        coverage = len(query.intersection(count))/max(1, len(query))
        ranked.append(dict(row, score=score, query_coverage=coverage))
    return sorted(ranked, key=lambda row: (-row["score"], row["doc_id"], row["start"]))


def locate_answer(answer, selected, *, citations=None):
    """Map exact text to a native body offset. Ambiguity follows displayed order.

    When citations are supplied, only an exact quote containing the answer may
    supply its location. No normalization or gold-offset lookup is performed.
    """
    if not isinstance(answer, str) or not answer.strip():
        return None
    for index, row in enumerate(selected):
        if citations is None:
            offset = row["text"].find(answer)
        else:
            offset = -1
            for citation in citations:
                if citation.get("source") != f"s{index}" or not isinstance(citation.get("quote"), str):
                    continue
                quote = citation["quote"]
                qstart, astart = row["text"].find(quote), quote.find(answer)
                if qstart >= 0 and astart >= 0:
                    offset = qstart+astart
                    break
        if offset >= 0:
            return dict(doc_id=row["doc_id"], start_offset=row["start"]+offset,
                        end_offset=row["start"]+offset+len(answer), score=1.0)
    return None

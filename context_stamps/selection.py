"""Inspectable evidence selection. Scores are utilities, not truth probabilities."""

from __future__ import annotations

import json
import math
import re
from dataclasses import asdict, dataclass

from .security import bounded_text, read_text


def words(text):
    return set(re.findall(r"\w+", text.lower()))


def rank_candidates(
    query, documents, scores, *, limit=10, coverage_weight=0.3, diversity_weight=0.15, pair_similarity=None
):
    """Reorder an external shortlist using relevance, coverage and redundancy.

    This is the same selector used in the SciFact experiment. Source freshness
    must be checked by the caller before supplying candidates. Returned values
    are indexes into documents, in selection order. No calibrated score implied.
    """
    bounded_text(query, 16384)
    if len(documents) != len(scores) or len(documents) > 1000 or not 0 <= limit <= 100:
        raise ValueError("invalid shortlist size or limit")
    if not all(math.isfinite(s) for s in scores):
        raise ValueError("scores must be finite")
    if any(not math.isfinite(v) or not 0 <= v <= 1 for v in (coverage_weight, diversity_weight)):
        raise ValueError("weights must be finite in [0,1]")
    for document in documents:
        bounded_text(document)
    token_sets = [words(document) for document in documents]
    query_words = words(query)
    selected, covered = [], set()
    remaining = list(range(len(documents)))
    while remaining and len(selected) < limit:

        def utility(i):
            gain = len((token_sets[i] & query_words) - covered) / max(1, len(query_words))
            similarities = []
            for j in selected:
                value = (
                    pair_similarity(i, j)
                    if pair_similarity
                    else len(token_sets[i] & token_sets[j]) / max(1, len(token_sets[i] | token_sets[j]))
                )
                if not math.isfinite(value) or not -1.00001 <= value <= 1.00001:
                    raise ValueError("pair similarity must be finite in [-1,1]")
                similarities.append(value)
            return (
                scores[i] + coverage_weight * gain - diversity_weight * max(0, max(similarities, default=0))
            )

        best = max(remaining, key=utility)
        selected.append(best)
        covered.update(token_sets[best] & query_words)
        remaining.remove(best)
    return selected


def features(query, text, score=0.0):
    """Fixed, versioned features; no source ID, freshness label or answer leakage."""
    q, d = words(query), words(text)
    return [
        len(q & d) / max(1, len(q)),
        len(q & d) / max(1, len(q | d)),
        float(score),
        min(len(text.encode("utf-8")), 65536) / 65536,
        float(bool(q) and q <= d),
    ]


@dataclass
class LinearSelector:
    """Small learned reranker with JSON weights and standard-library inference."""

    weights: list[float]
    bias: float
    feature_version: int = 1

    def __post_init__(self):
        if self.feature_version != 1 or len(self.weights) != 5:
            raise ValueError("unsupported selector features")
        if not all(math.isfinite(x) and abs(x) <= 10000 for x in [*self.weights, self.bias]):
            raise ValueError("invalid selector coefficients")

    def score(self, query, text, base_score=0.0):
        value = self.bias + sum(w * x for w, x in zip(self.weights, features(query, text, base_score)))
        return 1 / (1 + math.exp(-max(-60, min(60, value))))

    def save(self, path):
        from pathlib import Path

        Path(path).write_text(json.dumps(asdict(self), indent=2) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path):
        return cls(**json.loads(read_text(path, 8192)))


def fit_selector(examples, *, seed=7, epochs=400, rate=0.3, regularization=0.01):
    """Fit regularized logistic relevance on (query, text, base_score, binary label)."""
    import numpy as np

    if not 1 <= len(examples) <= 100000 or not 1 <= epochs <= 2000:
        raise ValueError("training size or epochs exceed limits")
    x = np.array([features(q, t, s) for q, t, s, _ in examples], dtype=float)
    y = np.array([label for _, _, _, label in examples], dtype=float)
    if not np.isfinite(x).all() or not np.isin(y, [0, 1]).all():
        raise ValueError("training examples must be finite with binary labels")
    w = np.random.default_rng(seed).normal(0, 0.01, x.shape[1])
    bias = 0.0
    losses = []
    for epoch in range(epochs):
        z = np.clip(x @ w + bias, -60, 60)
        error = 1 / (1 + np.exp(-z)) - y
        w -= rate * (x.T @ error / len(y) + regularization * w)
        bias -= rate * error.mean()
        if epoch % 50 == 0 or epoch == epochs - 1:
            losses.append({"epoch": epoch, "loss": float(np.mean(np.logaddexp(0, z) - y * z))})
    return LinearSelector(w.tolist(), float(bias)), losses


@dataclass
class Selection:
    text: str
    units: int
    budget: int
    status: str
    selected: list[str]
    missing_required: list[str]
    decisions: list[dict]
    reranked: bool
    next_action: str

    def to_dict(self):
        return asdict(self)


def select_evidence(
    memory,
    query,
    *,
    budget=2048,
    revisions=None,
    required=(),
    reranker=None,
    coverage_weight=0.2,
    diversity_weight=0.1,
    ambiguity_gap=0.1,
    counter=None,
):
    """Budgeted greedy utility with hard freshness and must-keep constraints.

    No approximation guarantee is claimed for this budgeted heuristic. Required
    IDs come from the trusted caller. They cannot override a freshness failure.
    """
    bounded_text(query, 16384)
    if not isinstance(budget, int) or not 0 <= budget <= 1048576:
        raise ValueError("invalid budget")
    if any(
        not math.isfinite(v) or not 0 <= v <= 1 for v in (coverage_weight, diversity_weight, ambiguity_gap)
    ):
        raise ValueError("selection weights must be finite in [0,1]")
    required = set(required)
    if len(required) > 128:
        raise ValueError("at most 128 required sources")
    count = counter or (lambda text: len(text.encode("utf-8")))
    rows = memory._rank(query, revisions)
    decisions = []
    eligible = []
    for row in rows:
        if row["freshness"] not in {"current", "unchecked"}:
            decisions.append({"source": row["source"], "reason": row["freshness"]})
        else:
            eligible.append(row)
    reranked = bool(
        reranker and len(eligible) > 1 and eligible[0]["score"] - eligible[1]["score"] <= ambiguity_gap
    )
    if reranked:
        for row in eligible:
            row["score"] = float(reranker(query, row["text"], row["score"]))
            if not math.isfinite(row["score"]) or not 0 <= row["score"] <= 1:
                raise ValueError("reranker must return a finite utility in [0,1]")
    chunks, selected, seen, covered = [], [], set(), set()
    query_words = words(query)
    chosen_words = []
    while eligible:

        def utility(row):
            tokens = words(row["text"])
            gain = len((tokens & query_words) - covered) / max(1, len(query_words))
            redundancy = max(
                (len(tokens & old) / max(1, len(tokens | old)) for old in chosen_words), default=0
            )
            return (
                row["source"] in required,
                row["score"] + coverage_weight * gain - diversity_weight * redundancy,
            )

        # Input order breaks exact ties deterministically.
        row = max(eligible, key=utility)
        eligible.remove(row)
        key = (row["digest"], row["dependencies"])
        if key in seen and row["source"] not in required:
            decisions.append({"source": row["source"], "reason": "exact_duplicate"})
            continue
        header = json.dumps({"source": row["source"], "sha256": row["digest"]}, ensure_ascii=False)
        chunk = header + "\n" + row["text"]
        size = count("\n\n".join([*chunks, chunk]))
        if not isinstance(size, int) or size < 0:
            raise ValueError("counter must return a nonnegative integer")
        if size > budget:
            decisions.append({"source": row["source"], "reason": "budget"})
            continue
        chunks.append(chunk)
        selected.append(row["source"])
        seen.add(key)
        chosen_words.append(words(row["text"]))
        covered.update(chosen_words[-1] & query_words)
        decisions.append(
            {
                "source": row["source"],
                "reason": "selected",
                "freshness": row["freshness"],
                "score": row["score"],
            }
        )
    missing = sorted(required - set(selected))
    status = (
        "insufficient_evidence"
        if missing or not selected
        else ("current" if revisions is not None else "unchecked")
    )
    # No partial packet can appear sufficient when a hard requirement was missed.
    text = "" if missing else "\n\n".join(chunks)
    return Selection(
        text,
        count(text),
        budget,
        status,
        selected if not missing else [],
        missing,
        decisions,
        reranked,
        "refresh_sources_or_increase_budget" if status == "insufficient_evidence" else "review_evidence",
    )

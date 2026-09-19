"""Small disclosed retrieval baselines, not wrappers around competing products."""

import math
from collections import Counter

from .selection import words


def bm25(query, documents, k1=1.5, b=0.75):
    import re

    tokens = [re.findall(r"\w+", text.lower()) for text in documents]
    average = sum(map(len, tokens)) / max(1, len(tokens))
    counts = [Counter(row) for row in tokens]
    scores = [0.0] * len(documents)
    for term in words(query):
        df = sum(term in row for row in counts)
        idf = math.log(1 + (len(documents) - df + 0.5) / (df + 0.5))
        for i, row in enumerate(counts):
            tf = row[term]
            if tf:
                scores[i] += idf * tf * (k1 + 1) / (tf + k1 * (1 - b + b * len(tokens[i]) / max(1, average)))
    return scores


def corpus_mean_similarity(vectors):
    """Exact non-self mean in O(nd) work; avoids an n by n Gram matrix."""
    import numpy as np

    x = np.asarray(vectors, dtype=float)
    if x.ndim != 2 or len(x) < 2 or not np.isfinite(x).all():
        raise ValueError("at least two finite vectors are required")
    return (x @ x.sum(axis=0) - np.einsum("ij,ij->i", x, x)) / (len(x) - 1)


def mmr_order(scores, vectors, *, limit=10, relevance_weight=0.8):
    """Classical MMR, preserving selected order rather than re-sorting relevance."""
    import numpy as np

    x = np.asarray(vectors)
    remaining, chosen = list(range(len(scores))), []
    while remaining and len(chosen) < limit:

        def value(i):
            redundancy = max((float(x[i] @ x[j]) for j in chosen), default=0)
            return relevance_weight * scores[i] - (1 - relevance_weight) * redundancy

        best = max(remaining, key=value)
        chosen.append(best)
        remaining.remove(best)
    return chosen

"""Small query-conditioned lexical relation diffusion; experimental only.

Copyright (c) 2026 Prashant Jagtap. MIT License.
Edges mean literal title mentions, not verified causal/factual dependencies.
Scores allocate attention mass, not calibrated correctness or sufficiency.
"""

import collections
import math

from multihop_data import normalized

STOP = frozenset("a an the of in on at by to for from with and or is was were are be been did do does what which who whom whose when where how why that this it its as into during after before had has have their his her he she they them year years name named".split())
FEATURES = ("bm25", "query_coverage", "title_coverage", "idf_overlap", "title_in_query",
            "query_in_text", "log_length", "out_degree", "in_degree", "max_tf", "query_length", "title_length")


def terms(text):
    return [w for w in normalized(text).split() if w not in STOP]


def features(case):
    """Input-only features; target keys are not accepted in this contract."""
    if set(case) != {"key", "question", "paragraphs"}:
        raise ValueError("only the explicit question/paragraph input contract is accepted")
    docs = case["paragraphs"]
    if not 1 <= len(docs) <= 32:
        raise ValueError("bounded candidate set required")
    for p in docs:
        if set(p) != {"idx", "title", "text", "sha256"}:
            raise ValueError("paragraph contains undeclared fields")
    q = set(terms(case["question"]))
    tokens = [terms(p["title"]+" "+p["text"]) for p in docs]
    counts = [collections.Counter(t) for t in tokens]
    n = len(docs)
    df = collections.Counter(w for count in counts for w in count)
    idf = {w: math.log1p((n-df[w]+.5)/(df[w]+.5)) for w in q}
    total_idf = math.fsum(idf.values()) or 1.
    avg_len = sum(map(len, tokens))/n or 1.
    titles = [normalized(p["title"]) for p in docs]
    bodies = [" "+normalized(p["text"])+" " for p in docs]
    adjacency = [[float(i == j or (len(titles[j]) >= 4 and titles[j] not in STOP
                                   and " "+titles[j]+" " in bodies[i])) for j in range(n)] for i in range(n)]
    transition = [[v/sum(row) for v in row] for row in adjacency]
    x, bm25 = [], []
    for i, (doc, count) in enumerate(zip(docs, counts)):
        score = sum(idf[w]*(count[w]*2.2)/(count[w]+1.2*(.25+.75*len(tokens[i])/avg_len)) for w in q if count[w])
        bm25.append(score)
        title_terms = set(terms(doc["title"]))
        x.append([
            math.log1p(score), len(q & count.keys())/max(1, len(q)),
            len(q & title_terms)/max(1, len(q)), math.fsum(idf[w] for w in q & count.keys())/total_idf,
            float(bool(titles[i]) and " "+titles[i]+" " in " "+normalized(case["question"])+" "),
            float(normalized(case["question"]) in bodies[i]), math.log1p(len(tokens[i]))/10,
            (sum(adjacency[i])-1)/n, (sum(row[i] for row in adjacency)-1)/n,
            math.log1p(max((count[w] for w in q), default=0)), len(q)/32, len(title_terms)/16,
        ])
    return x, transition, bm25


def probabilities(x, transition, weights, alpha, *, steps=4):
    """Truncated personalized random walk with a query-learned restart vector.

    With row-stochastic P and alpha < 1, h <- (1-alpha)q + alpha P^T h
    contracts L1 distances by at most alpha. Four steps are frozen here;
    convergence is not a semantic correctness guarantee.
    """
    if (len(weights) != len(FEATURES) or not 0 <= alpha <= .9
            or type(steps) is not int or not 0 <= steps <= 32):
        raise ValueError("bounded frozen policy required")
    logits = [math.fsum(a*b for a, b in zip(row, weights)) for row in x]
    top = max(logits)
    e = [math.exp(v-top) for v in logits]
    q = [v/sum(e) for v in e]
    h = q[:]
    for _ in range(steps):
        h = [(1-alpha)*q[j]+alpha*math.fsum(transition[i][j]*h[i] for i in range(len(h))) for j in range(len(h))]
    return h


def rank(case, policy=None):
    x, transition, bm25 = features(case)
    if policy is None:
        values = bm25
    else:
        if policy["features"] != list(FEATURES) or policy["steps"] != 4:
            raise ValueError("feature/iteration revision mismatch")
        values = probabilities(x, transition, policy["weights"], policy["alpha"], steps=policy["steps"])
    order = sorted(range(len(values)), key=lambda i: (-values[i], case["paragraphs"][i]["sha256"], case["paragraphs"][i]["idx"]))
    return [case["paragraphs"][i] for i in order], values

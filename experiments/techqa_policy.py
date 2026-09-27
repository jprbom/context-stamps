"""Small local expected-utility policy; statistical learning without LLM updates.

Copyright (c) 2026 Prashant Jagtap. MIT License.
The policy predicts the benefit of answering rather than abstaining. Its score
is not a calibrated probability or a semantic-verification certificate.
"""

import math

import numpy as np
from techqa_context import terms

FEATURES = ("best_bm25", "bm25_margin", "mean_bm25", "query_coverage", "window_count",
            "question_length", "answer_length", "answer_query_overlap", "quote_count",
            "quoted_characters", "answer_to_quote_ratio", "answer_digits_fraction")
RIDGES = (1.0, 10.0, 100.0)
THRESHOLDS = (-0.2, 0.0, 0.2, 0.4, 0.6, 0.8)


def features(compiled, result):
    """Input/output-only features: no dataset label or gold offset accepted."""
    selected = compiled.get("selected", [])
    scores = sorted((r["score"] for r in selected), reverse=True)
    answer = result.get("answer") or ""
    check = result.get("check") or {}
    quotes = check.get("quotes", [])
    quoted = sum(q["end"]-q["start"] for q in quotes)
    qterms, aterms = set(terms(compiled.get("question", ""))), set(terms(answer))
    best = scores[0] if scores else 0.0
    return [math.log1p(best), (best-scores[1])/(1+best) if len(scores) > 1 else 0.0,
            math.log1p(sum(scores)/max(1, len(scores))),
            sum(r["query_coverage"] for r in selected)/max(1, len(selected)), len(selected)/8,
            math.log1p(len(compiled.get("question", ""))), math.log1p(len(answer)),
            len(qterms & aterms)/max(1, len(aterms)), len(quotes)/4, math.log1p(quoted),
            min(1.0, len(answer)/max(1, quoted)), sum(c.isdigit() for c in answer)/max(1, len(answer))]


def fit(x, y, ridge):
    x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    if (x.ndim != 2 or x.shape[1] != len(FEATURES) or len(x) < 20 or y.shape != (len(x),)
            or not np.isfinite(x).all() or not np.isfinite(y).all() or np.abs(y).max() > 1
            or ridge not in RIDGES):
        raise ValueError("at least 20 finite verified fit outcomes required")
    mean, scale = x.mean(axis=0), x.std(axis=0)
    scale = np.maximum(scale, 1e-6)
    z = np.c_[np.ones(len(x)), np.clip((x-mean)/scale, -8, 8)]
    regularizer = np.eye(z.shape[1])*ridge
    regularizer[0, 0] = 0
    weights = np.linalg.solve(z.T@z+regularizer, z.T@y)
    # Compact checkpoint only: serving still reconstructs FP64 arithmetic.
    quant_scale = max(float(np.max(np.abs(weights)))/32767, 1e-12)
    quantized = np.rint(weights/quant_scale).astype(np.int16)
    return dict(schema=1, features=list(FEATURES), mean=mean.tolist(), scale=scale.tolist(),
                weights=weights.tolist(), int16_weights=quantized.tolist(), int16_scale=quant_scale,
                ridge=ridge, fit_observations=len(x), score_kind="expected answer utility, not probability",
                activated=False, quantization_kind="coefficient storage; FP64 inference, no integer-kernel speed claim")


def predict(policy, x, *, quantized=False):
    if policy["features"] != list(FEATURES):
        raise ValueError("policy feature schema changed")
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 2 or x.shape[1] != len(FEATURES) or not np.isfinite(x).all():
        raise ValueError("finite feature matrix required")
    mean, scale = np.asarray(policy["mean"]), np.asarray(policy["scale"])
    weights = np.asarray(policy["int16_weights"])*policy["int16_scale"] if quantized else np.asarray(policy["weights"])
    z = np.c_[np.ones(len(x)), np.clip((x-mean)/scale, -8, 8)]
    return np.clip(z@weights, -1.0, 1.0)

"""Small ChartQA scorers with explicit reference semantics.

Copyright (c) 2026 Prashant Jagtap. MIT License.
Independent implementation of the published metric definition. Reference:
https://github.com/google-research/pix2struct/blob/6fe25c1dc8151823ee3b479519d8d5948812fee4/pix2struct/metrics.py
The zero-target/string behavior is deliberately preserved, not repaired.
"""

PIX2STRUCT_REVISION = "6fe25c1dc8151823ee3b479519d8d5948812fee4"


def numeric_value(value):
    try:
        if value.endswith("%"):
            return float(value.rstrip("%")) / 100
        return float(value)
    except ValueError:
        return None


def relaxed_correctness(target, prediction):
    """Reference-compatible 5% relative tolerance; no whitespace normalization."""
    if type(target) is not str or type(prediction) is not str:
        raise ValueError("string target and prediction required")
    expected, actual = numeric_value(target), numeric_value(prediction)
    if expected and actual is not None:
        return abs(actual - expected) / abs(expected) <= 0.05
    return target.lower() == prediction.lower()


def score_answer(labels, answer):
    """Missing/failed answers score zero; supplied keys never reach a reader."""
    if (type(labels) is not list or not 1 <= len(labels) <= 16
            or any(type(v) is not str or not v.strip() or len(v) > 4096 for v in labels)):
        raise ValueError("bounded nonempty reference strings required")
    if answer is None:
        return dict(relaxed=False, stripped_exact=False, abstained=True)
    if type(answer) is not str or not answer.strip() or len(answer) > 4096:
        raise ValueError("bounded answer string or explicit abstention required")
    return dict(relaxed=any(relaxed_correctness(t, answer) for t in labels),
                stripped_exact=any(t.strip() == answer.strip() for t in labels), abstained=False)

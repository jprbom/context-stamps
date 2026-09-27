"""Versioned typed-scalar visual interface; historical protocol stays unchanged.

Copyright (c) 2026 Prashant Jagtap. MIT License.
This is an interface correction, not training or answer-key-based repair.
"""

import json
import math

import chartqa_protocol as original
from chartqa_quant import strict_json

VERSION = "typed-json-v2"
ANSWER = (
    "Return only a JSON object with exactly one field answer. Use a JSON number for a numeric "
    "answer, a short string for a category, or null if the evidence is insufficient. "
    "Do not include a calculation, explanation or nested object. Keep the chart's numeric "
    "scale; omit a redundant unit name. For a percentage question return the percentage "
    "number, such as 25, not its fractional equivalent. "
    'Examples of the output shape: {"answer":12}, {"answer":"category"}, {"answer":null}.'
)


def request_body(mode, *, question=None, image=None, memory=None):
    body = original.request_body(mode, question=question, image=image, memory=memory)
    if mode in ("direct", "memory"):
        body["messages"][0]["content"] = original.COMMON + ANSWER
        body["format"] = "json"
    return body


def parse_response(raw, mode, *, question=None, table=None):
    if mode not in ("direct", "memory"):
        return original.parse_response(raw, mode, question=question, table=table)
    outer = strict_json(raw.decode("utf-8"), maximum=4*1024**2)
    if type(outer) is not dict or type(outer.get("message")) is not dict:
        raise ValueError("complete response envelope required")
    content = strict_json(outer["message"].get("content"))
    if type(content) is not dict or set(content) != {"answer"}:
        raise ValueError("exact answer field required")
    value = content["answer"]
    if value is None:
        answer, kind = None, "null"
    elif type(value) is str and value.strip() and len(value) <= 4096:
        # Keep strings verbatim. Never extract a number from a malformed answer.
        answer, kind = value, "string"
    elif type(value) is int or (type(value) is float and math.isfinite(value)):
        answer, kind = json.dumps(value, allow_nan=False, separators=(",", ":")), "number"
        if len(answer) > 4096:
            raise ValueError("bounded scalar required")
    else:
        raise ValueError("finite number, bounded nonempty string or null required")
    # Validate the unchanged envelope and usage with the historical checks.
    # Only the already-validated scalar is projected to their string contract.
    # The caller retains the original raw response before this projection.
    outer["message"]["content"] = original.encoded(dict(answer=answer)).decode()
    result = original.parse_response(original.encoded(outer), mode)
    return dict(**result, answer_value=value, answer_type=kind)

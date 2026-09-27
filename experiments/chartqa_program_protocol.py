"""Prospective program-instruction correction after the typed-scalar canary.

Copyright (c) 2026 Prashant Jagtap. MIT License.
No parser repair or extra executable operation is introduced.
"""

from chartqa_scalar_protocol import request_body as scalar_request

VERSION = "typed-json-program-v3"
LOOKUP = (
    ' For a direct numeric cell lookup, the complete response is {"program":{"cell":"c2"}} '
    'when c2 is the selected cell; replace c2 with the actual evidence cell ID. '
    'For its label, use {"program":{"cell":"c2","field":"label"}}. '
    'A cell reference is an expression by itself: there is no operation named cell or lookup. '
    'If the requested category is absent, return {"program":null}.'
)


def request_body(mode, *, question=None, image=None, memory=None):
    body = scalar_request(mode, question=question, image=image, memory=memory)
    if mode == "program":
        body["messages"][0]["content"] += LOOKUP
    return body

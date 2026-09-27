"""Exact source-span checks for untrusted model answers.

Copyright (c) 2026 Prashant Jagtap. MIT License.
Binding a quotation to authorized evidence does not establish that the answer
follows from that evidence. No result from this module is a truth certificate.
"""

import hashlib
import json
from dataclasses import asdict, dataclass, field

from .context_state import ContextSnapshot, ContextState, NodeRef, canonical
from .security import bounded_text


@dataclass(frozen=True)
class CitationPacket:
    question: str
    snapshot: ContextSnapshot
    require_answer_span: bool
    seal: str

    def __post_init__(self):
        bounded_text(self.question, 16384)
        if (not self.question.strip() or type(self.snapshot) is not ContextSnapshot
                or type(self.require_answer_span) is not bool
                or not 1 <= len(self.snapshot.records) <= 32):
            raise ValueError("bounded question, snapshot and extraction policy required")
        bounded_text(self.payload, 1048576)

    @property
    def payload(self):
        return canonical(dict(question=self.question, require_answer_span=self.require_answer_span,
                              sources=[dict(id=f"s{i}", text=r.node.text, kind=r.node.kind)
                                       for i, r in enumerate(self.snapshot.records)]))

    @property
    def digest(self):
        return hashlib.sha256((self.snapshot.state_revision+"\0"+self.payload).encode()).hexdigest()

    def is_current(self, state):
        return state.verify_binding(self.snapshot, self.payload, self.seal)


@dataclass(frozen=True)
class BoundQuote:
    source: NodeRef
    start: int
    end: int
    sha256: str
    kind: str


@dataclass(frozen=True)
class CitationCheck:
    status: str
    reason: str
    answer: str | None
    quotes: tuple[BoundQuote, ...]
    packet_digest: str
    reply_digest: str
    semantics_verified: bool = field(default=False, init=False)

    def to_dict(self):
        return asdict(self)


def prepare_citations(state, snapshot, question, *, refs=None, require_answer_span=True):
    """Bind a bounded selected view; callers obtain the scope from their host.

    Send ``packet.payload`` as untrusted source data with your own model prompt.
    The packet's seal binds the question, exact view and extraction policy.
    """
    if not isinstance(state, ContextState):
        raise ValueError("host-owned context state required")
    if refs is not None:
        snapshot = state.subset(snapshot, refs)
    packet = CitationPacket(question, snapshot, require_answer_span, "")
    return CitationPacket(question, snapshot, require_answer_span, state.seal(snapshot, packet.payload))


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON field")
        result[key] = value
    return result


def _no_constant(_):
    raise ValueError("nonfinite JSON number")


def _bounded_integer(value):
    if len(value) > 8:
        raise ValueError("oversized JSON integer")
    return int(value)


def check_citations(state, packet, reply):
    """Check strict JSON and exact quotes, then recheck current authorization.

    Expected shape: {"answer": str|null, "citations": [{"source": "s0",
    "quote": "exact source text"}]}. Missing or invalid evidence rejects the
    proposed answer. A source-bound answer can still be semantically wrong.
    The host must recheck ``packet.is_current(state)`` before later use; this
    function does not hold an authorization lock over downstream actions.
    """
    if type(packet) is not CitationPacket:
        raise ValueError("host-prepared citation packet required")
    bounded_text(reply, 32768)
    reply_digest = hashlib.sha256(reply.encode()).hexdigest()

    def result(status, reason, answer=None, quotes=()):
        return CitationCheck(status, reason, answer, quotes, packet.digest, reply_digest)

    if not packet.is_current(state):
        return result("rejected", "unavailable_context")
    try:
        proposal = json.loads(reply, object_pairs_hook=_unique_object, parse_constant=_no_constant,
                              parse_int=_bounded_integer)
        if type(proposal) is not dict or set(proposal) != {"answer", "citations"}:
            raise ValueError("exact proposal fields required")
        answer, citations = proposal["answer"], proposal["citations"]
        if type(citations) is not list or len(citations) > 8:
            raise ValueError("bounded citation list required")
        if answer is None:
            if citations:
                raise ValueError("abstention cannot carry answer citations")
            if not packet.is_current(state):
                return result("rejected", "unavailable_context")
            return result("abstained", "model_abstained")
        bounded_text(answer, 4096)
        if not answer.strip() or not citations:
            raise ValueError("nonempty answer and citations required")
        bound, texts, seen = [], [], set()
        sources = {f"s{i}": r.node for i, r in enumerate(packet.snapshot.records)}
        for citation in citations:
            if type(citation) is not dict or set(citation) != {"source", "quote"}:
                raise ValueError("exact citation fields required")
            source_id, quote = citation["source"], citation["quote"]
            if type(source_id) is not str or source_id not in sources:
                raise ValueError("source outside packet")
            bounded_text(quote, 8192)
            if not quote.strip() or (source_id, quote) in seen:
                raise ValueError("empty or duplicate quote")
            seen.add((source_id, quote))
            node = sources[source_id]
            start = node.text.find(quote)
            if start < 0:
                raise ValueError("quote not exact")
            bound.append(BoundQuote(node.ref, start, start+len(quote), hashlib.sha256(quote.encode()).hexdigest(), node.kind))
            texts.append(quote)
        if packet.require_answer_span and not any(answer in text for text in texts):
            return result("rejected", "answer_outside_quotes")
        if not packet.is_current(state):
            return result("rejected", "unavailable_context")
        return result("source_bound", "exact_source_spans_only", answer, tuple(bound))
    except (ValueError, TypeError, KeyError, RecursionError):
        return result("rejected", "invalid_citation_proposal")

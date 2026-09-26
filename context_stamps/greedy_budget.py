"""Batch exact token counts without changing an ordered greedy selection.

Copyright (c) 2026 Prashant Jagtap. MIT License.
Callbacks belong to the trusted host. This module does not authorize evidence,
infer sufficiency, approximate tokens or execute candidate text.
"""

from dataclasses import dataclass

from .security import bounded_text


@dataclass(frozen=True)
class BudgetedSelection:
    indices: tuple[int, ...]
    text: str
    tokens: int
    counted_candidates: int
    batches: int


def pack_in_order(size, render, count_batch, count_one, *, budget, max_selected,
                  batch_size=8, identity=None):
    """Retain sequential greedy semantics using speculative batch counting.

    render(tuple_of_indices) returns the complete prompt, including wrappers.
    count_batch(list_of_texts) returns ordered exact integer counts; count_one
    independently checks the final prompt. Both counters must use the same
    deterministic tokenizer with truncation/padding disabled. Callback outputs
    and identities must be stable during the call; authorization is external.

    Before the first rejection, count one addition at a time. Subsequently,
    count at most batch_size independent additions to the same accepted prefix.
    Consume consecutive rejections and the first fit; discard all later counts
    after a fit because their prefix is now stale. No monotonicity or additive
    token assumption is required. Batches can do extra work; measure their cost.
    """
    limits = ((size, 0, 8192), (budget, 0, 1048576), (max_selected, 0, 8192), (batch_size, 1, 16))
    if any(type(value) is not int or not low <= value <= high for value, low, high in limits):
        raise ValueError("bounded integer selection limits required")
    if not all(callable(fn) for fn in (render, count_batch, count_one)) or identity is not None and not callable(identity):
        raise ValueError("trusted rendering and counting callbacks required")

    def checked_render(indices):
        text = render(indices)
        bounded_text(text, 1024*1024)
        return text

    def checked_count(value):
        if type(value) is not int or not 0 <= value <= 2**31-1:
            raise ValueError("nonnegative exact token count required")
        return value

    selected, seen, position, rejected = [], set(), 0, False
    expected, counted, batches = None, 0, 0
    while position < size and len(selected) < max_selected:
        alternatives = []
        for i in range(position, size):
            key = identity(i) if identity is not None else i
            if key in seen:
                continue
            alternatives.append((i, key, checked_render(tuple(selected+[i]))))
            if len(alternatives) == (batch_size if rejected else 1):
                break
        if not alternatives:
            break
        counts = count_batch([text for _, _, text in alternatives])
        if type(counts) not in (list, tuple) or len(counts) != len(alternatives):
            raise ValueError("complete ordered token-count batch required")
        counts = [checked_count(value) for value in counts]
        batches += 1
        counted += len(counts)
        for (i, key, _), count in zip(alternatives, counts):
            position = i+1
            if count <= budget:
                selected.append(i)
                seen.add(key)
                expected = count
                break
            rejected = True
    text = checked_render(tuple(selected))
    actual = checked_count(count_one(text))
    if actual > budget or expected is not None and actual != expected:
        raise ValueError("full prompt exceeds budget or native tokenizer counts differ")
    return BudgetedSelection(tuple(selected), text, actual, counted, batches)

"""Explicit RULER input contracts, deterministic controls and runtime treatment.

Copyright (c) 2026 Prashant Jagtap. MIT License.
Native prompt markers and metric semantics derive from NVIDIA RULER/NeMo,
copyright 2024/2025 NVIDIA CORPORATION, Apache-2.0; see docs/THIRD_PARTY_NOTICES.md.
Benchmark-specific adapters, not a general semantic sufficiency estimator.
Only the public question string is accepted. No labels or answer-position fields.
"""

import collections
import hashlib
import re
import time
from dataclasses import dataclass

from context_stamps import ContextExpert, ContextNode, ContextRuntime, RuntimeBudget, Verification


@dataclass(frozen=True)
class Frame:
    prefix: str
    context: str
    suffix: str
    kind: str

    def render(self, context):
        return self.prefix + context + self.suffix


def frame(question):
    if type(question) is not str or len(question.encode()) > 1024 * 1024:
        raise ValueError("One bounded public question string required")
    # The last task block excludes CWE's native in-context worked example.
    markers = (
        ("needle", " afterwards.\n", "\nWhat "),
        ("variables", "Memorize and track the chain(s) of variable assignment hidden in the following text.\n\n", "\nQuestion:"),
        ("common", "Below is a numbered list of words. In these words, some appear more often than others. Memorize the ones that appear most often.\n", "\nQuestion:"),
        ("frequency", "Read the following coded text and track the frequency of each coded word. Find the three most frequently appeared coded words. ", "\nQuestion:"),
        ("qa", "The following are given documents.\n\n", "\n\nAnswer the question based on the given documents."),
    )
    for kind, start, end in markers:
        left = question.rfind(start)
        if left < 0:
            continue
        left += len(start)
        right = question.find(end, left)
        if right >= left:
            return Frame(question[:left], question[left:right], question[right:], kind)
    raise ValueError("Unsupported public prompt structure; do not silently slice")


@dataclass(frozen=True)
class Selection:
    kind: str
    texts: tuple[str, ...]
    dependencies: tuple[tuple[int, int], ...]
    direct_answer: str | None
    contract: str
    full_source_sha256: str


def select(view):
    """Read the whole public context. Conservatively fall back if ambiguous."""
    text = view.context
    digest = hashlib.sha256(text.encode()).hexdigest()

    def fallback(reason):
        # Split only to respect per-node byte limits, preserving every character.
        pieces = tuple(text[i:i+8000] for i in range(0, len(text), 8000)) or ("",)
        return Selection("full", pieces, (), None, reason, digest)

    if view.kind == "needle":
        query = re.search(r"for (.*?) mentioned in the provided text\?", view.suffix)
        if query is None:
            return fallback("unparsed_query")
        keys = re.split(r", and |, ", query[1])
        if not keys or len(set(keys)) != len(keys):
            return fallback("ambiguous_keys")
        matches = list(re.finditer(r"One of the special magic (?:numbers|uuids|words) for ([\w-]+) is: ([\w-]+)\.", text))
        if len(matches) != text.count("One of the special magic "):
            return fallback("unparsed_needle_statement")
        chosen = [m for m in matches if m[1] in keys]
        if set(m[1] for m in chosen) != set(keys) or not 1 <= len(chosen) <= 64:
            return fallback("missing_or_unbounded_keys")
        return Selection("exact", tuple(m[0] for m in chosen), (), " ".join(m[2] for m in chosen),
                         "all_literal_key_matches_v1", digest)
    if view.kind == "variables":
        query = re.search(r"assigned the value (\d+) in the text above", view.suffix)
        matches = list(re.finditer(r"VAR ([A-Z]{5}) = (?:VAR ([A-Z]{5})|(\d+))(?=[\s.]|$)", text))
        assignment_count = len(re.findall(r"\bVAR\s+\w+\s*=", text))
        if query is None or not matches or len(matches) > 100 or len(matches) != assignment_count:
            return fallback("unsupported_assignment_grammar")
        if len({m[1] for m in matches}) != len(matches):
            return fallback("reassignment_requires_temporal_semantics")
        values, dependencies, positions = {}, [], {}
        for i, m in enumerate(matches):
            if m[2] and m[2] not in values:
                return fallback("forward_missing_or_cyclic_assignment")
            values[m[1]] = values[m[2]] if m[2] else m[3]
            positions[m[1]] = i
            if m[2]:
                dependencies.append((i, positions[m[2]]))
        chosen = [i for i, m in enumerate(matches) if values[m[1]] == query[1]]
        if not chosen:
            return fallback("target_absent")
        remap = {old: new for new, old in enumerate(chosen)}
        edges = tuple((remap[a], remap[b]) for a, b in dependencies if a in remap and b in remap)
        return Selection("dependency", tuple(matches[i][0] for i in chosen), edges,
                         " ".join(matches[i][1] for i in chosen), "ordered_unique_assignments_v1", digest)
    if view.kind in ("common", "frequency"):
        if view.kind == "common":
            matches = list(re.finditer(r"(\d+)\. ([^\d]+?)(?= \d+\. |$)", text))
            if (not matches or [int(m[1]) for m in matches] != list(range(1, len(matches)+1))
                    or " ".join(m[0] for m in matches) != text):
                return fallback("incomplete_numbered_list_parse")
            terms = [m[2] for m in matches]
            count = 10
        else:
            terms = text.split()
            if not terms or any(t != "..." and re.fullmatch(r"[a-z]{6}", t) is None for t in terms):
                return fallback("unsupported_coded_words")
            terms = [t for t in terms if t != "..."]
            count = 3
        frequencies = collections.Counter(terms)
        ranked = sorted(frequencies.items(), key=lambda row: (-row[1], row[0]))
        if len(ranked) < count or (len(ranked) > count and ranked[count-1][1] == ranked[count][1]):
            return fallback("ambiguous_frequency_boundary")
        text = "Exact CPU counts from the entire source list (word: occurrences):\n" + "\n".join(
            f"{term}: {frequency}" for term, frequency in ranked[:count])
        text += f"\nLargest omitted count: {ranked[count][1] if len(ranked) > count else 0}."
        return Selection("statistic", (text,), (), " ".join(term for term, _ in ranked[:count]),
                         "full_scan_counts_unique_top_k_v1", digest)
    return fallback("no_semantic_sufficiency_certificate")


def runtime_context(view, selection, token_count):
    """Manage selected source revisions and declared variable dependencies.

    Completeness here means satisfying the explicit parser/full-source contract,
    not that arbitrary prose is true or that a reader will answer correctly.
    """
    started = time.perf_counter()
    runtime = ContextRuntime(tenant="ruler-local", principal="evaluator", role="reader", policy="public-v1",
                             token_counter=token_count)
    keys = tuple(f"source-{i:04d}" for i in range(len(selection.texts)))
    for key, text in zip(keys, selection.texts):
        runtime.put(ContextNode(key, text, selection.full_source_sha256, frozenset({"reader"})))
    for a, b in selection.dependencies:
        runtime.link(keys[a], keys[b], "depends_on", provenance=selection.contract)
    # Leaves pull their complete chains through the existing runtime dependency closure.
    depended = {b for _, b in selection.dependencies}
    roots = tuple(k for i, k in enumerate(keys) if i not in depended)
    expert = ContextExpert("contract", lambda request: list(roots), estimated_ms=0)
    prepared = runtime.prepare_context(view.suffix, eligible=keys, experts=[expert], baseline="contract",
        scope="ruler-public", limit=len(roots),
        verifier=lambda packet: Verification(set(packet.sources) == set(keys)),
        budget=RuntimeBudget(bytes=1048576, tokens=30000, milliseconds=10000, iterations=1))
    if prepared.status != "complete":
        raise ValueError(f"Runtime failed explicit contract: {prepared.reason}")
    # Verify the 32-byte receipt resolves before handing any text to the reader.
    resolved = runtime.resolve(prepared.receipt, budget=RuntimeBudget(bytes=1048576, tokens=30000, milliseconds=10000))
    if resolved.status != "complete" or resolved.text != prepared.text:
        raise ValueError("Receipt failed materialization check")
    return prepared.text, dict(receipt_hex=prepared.receipt.hex(), receipt_bytes=len(prepared.receipt),
                               sources=prepared.sources, dependency_edges=len(selection.dependencies),
                               seconds=time.perf_counter()-started, contract=selection.contract,
                               full_source_sha256=selection.full_source_sha256)


def native_score(prediction, references, match_type):
    """NeMo RULER semantics: raw case-insensitive substring recall / any alias.

    Deliberately keep partial credit and permissive substring behavior. This is
    neither exact-match correctness nor a security/production answer verifier.
    """
    if type(prediction) is not str or not references or any(type(r) is not str or not r for r in references):
        raise ValueError("String generation and nonempty references required")
    hits = [float(reference.lower() in prediction.lower()) for reference in references]
    if match_type == "all":
        return sum(hits) / len(hits)
    if match_type == "part":
        return max(hits)
    raise ValueError("Unknown native match mode")

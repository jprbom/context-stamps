"""Exact batched packing with the original retrieval and scope checks unchanged.

Three unsuccessful exploratory lookup/counting variants are retained separately.
The original 432-call experiment remains frozen in lme_relations.py.
"""

import hashlib
import json

from lme_memory import prompt

from context_stamps.context_state import AccessScope, ContextState
from context_stamps.greedy_budget import pack_in_order


def pack_fast(question, domain, views, arm, tokenizer, *, batch_size=8):
    if arm not in ("structure", "scoped", "relations"):
        raise ValueError("unregistered arm")
    relation = arm == "relations"
    candidates, limit = views[:160 if relation else 32], 96 if relation else 16

    def render(indices):
        chosen = [candidates[i] for i in indices]
        if not relation:
            body = [view.text for view in chosen]
        else:
            groups = {}
            for i, view in enumerate(chosen):
                group = groups.setdefault(view.page, dict(page=list(view.page), records=[]))
                group["records"].append(dict(id=i, episode=view.episode,
                    steps=sorted({o.step for o in view.occurrences}), relation=view.relation, observation=view.body))
            body = list(groups.values())
        return prompt(question, domain, json.dumps(body, ensure_ascii=False))

    result = pack_in_order(len(candidates), render,
        lambda texts: [len(e.ids) for e in tokenizer.encode_batch_fast(texts, add_special_tokens=False)],
        lambda text: len(tokenizer.encode(text, add_special_tokens=False).ids),
        budget=6144, max_selected=limit, batch_size=batch_size,
        identity=(lambda i: (candidates[i].page, candidates[i].body)) if relation else None)
    selected, rendered = [candidates[i] for i in result.indices], result.text
    state = ContextState(tenant="lme-public", policy_revision="public-v1", clock=lambda: 1)
    for view in selected:
        state.put(view.canonical_node(tenant="lme-public", roles=("reader",), observed_at=1))
    snapshot = state.snapshot(AccessScope("lme-public", "evaluator", "public-v1", ("reader",)), at=1, known_at=1)
    if not state.verify_binding(snapshot, rendered, state.seal(snapshot, rendered)):
        raise ValueError("source/scope binding failed")
    return rendered, dict(input_tokens=result.tokens, prompt_sha256=hashlib.sha256(rendered.encode()).hexdigest(),
        sources=[dict(view_id=v.key, episode=v.episode, revision=v.source_revision,
                      view_sha256=hashlib.sha256(v.text.encode()).hexdigest()) for v in selected],
        scope_checked=True, sufficient_context_certified=False)

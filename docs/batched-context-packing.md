# Exact batching for context token budgets

By Prashant Jagtap

`context_stamps.greedy_budget.pack_in_order` batches counts for an ordered greedy
packing policy when the host's tokenizer can count several complete prompts in
parallel. It preserves the policy's accept/skip decisions; it does not change
retrieval quality, learn a selector or certify that the selected context is
sufficient. It is an optional compiler primitive with no package dependencies.

## Why complete prompts matter

Token counts are generally not additive across arbitrary strings. A separator,
Unicode normalization or special token can change tokenization at a boundary.
Every candidate here is therefore rendered as the complete prompt, including
its instructions, data serialization and chat template. No character-per-token
estimate or sum of independently encoded chunks controls the budget.

Before the first oversized candidate, the helper counts one possible addition
at a time. After a rejection, it can batch up to eight independent additions to
the same accepted prefix. It consumes consecutive rejections and the first
accepted candidate. Counts after that first acceptance are discarded because
their prefix is now out of date. This reproduces sequential greedy selection
even when counts are nonmonotonic. The final prompt receives a separate native
count; a mismatch raises an error.

Batching may do more total work. It needs enough parallel CPU capacity and a
workload with rejected candidates to benefit. The appropriate batch size is a
deployment choice; the supported range is 1–16. Fewer callback rounds alone do
not establish lower latency, CPU consumption or energy use.

## Use a local tokenizer

This example uses the actual tokenizer supplied with a locally installed model.
It does not download weights, send requests or invoke that model:

```python
import json
from tokenizers import Tokenizer
from context_stamps.greedy_budget import pack_in_order

tokenizer = Tokenizer.from_file("/path/to/local/model/tokenizer.json")
tokenizer.no_truncation()
tokenizer.no_padding()

# Obtain these from the application's current authorization-aware retrieval.
authorized_fragments = ["First source passage", "Second source passage"]

def render(indices):
    context = json.dumps([authorized_fragments[i] for i in indices])
    # Supply the exact chat template used by your chosen model.
    return "<|im_start|>user\nUse this evidence: " + context + "<|im_end|>\n<|im_start|>assistant\n"

result = pack_in_order(
    len(authorized_fragments), render,
    lambda texts: [len(e.ids) for e in tokenizer.encode_batch_fast(
        texts, add_special_tokens=False)],
    lambda text: len(tokenizer.encode(text, add_special_tokens=False).ids),
    budget=6144, max_selected=16, batch_size=8,
)
print(result.indices, result.tokens)
model_input = result.text
```

The measured local environment uses `tokenizers==0.23.2`. The repository core
does not require it; install it separately for this optional integration. The
native fast encoder omits character-offset computation, so its returned offsets
must not be used as source citations. Context Stamps retains its own original
source bindings. [Tokenizer source and API](https://github.com/huggingface/tokenizers)
describe the upstream implementation; the experiment checks full token-ID parity
on Unicode, whitespace, normalization and added-token canaries.

The runnable example is:

```powershell
python -m pip install tokenizers==0.23.2
$env:RAYON_NUM_THREADS='8'
$env:TOKENIZERS_PARALLELISM='true'
python examples/batched_context.py --tokenizer ../qwen25-coder-7b-tokenizer/tokenizer.json
```

Use a deterministic tokenizer without dropout. Both count callbacks must have
the same configuration, vocabulary and special-token behavior. Keep fragment
content and the rendering callback stable during the call. Optional `identity`
deduplicates only identities already accepted; a rejected occurrence does not
suppress a later candidate with the same identity.

## Boundaries and integration

The helper accepts at most 8,192 candidate positions, 16 simultaneous candidate
prompts and 1 MiB per rendered prompt. Its returned `tokens` field has the units
of the trusted token-count callbacks. It rejects missing/invalid batch counts,
invalid limits, an over-budget final prompt or disagreement with the final native
count. It cannot detect a dishonest tokenizer callback or authorize an input.

Authorize and reject stale evidence before invoking it. After selection, retain
source/version bindings and recheck current access before model dispatch. The
LongMemEval integration performs the same canonical-state/scope checks as its
frozen baseline. Packing does not make untrusted source text safe instructions.

This is an ordered greedy policy, not a proof of a globally minimal context set.
The helper does not execute source text, perform background learning, retain
cross-request prompt caches or modify model attention/weights. Host tokenizers
may have their own caches and concurrency behavior. Account for those resources.

## Reproduce the paired CPU experiment

The [page/relation study](../evidence/lme-relations-v2/README.md) supplies the frozen
index and prepared prompts. Run in a quiet CPU environment; adding simultaneous
tests or training would confound the timing comparison.

```powershell
$env:PYTHONPATH='.'
$env:RAYON_NUM_THREADS='8'
$env:TOKENIZERS_PARALLELISM='true'
python experiments/profile_batched_packing.py --study ../lme-relations-v2 --tokenizer ../qwen25-coder-7b-tokenizer --output ../batched-packing-new-run
```

The protocol freezes source/input hashes before profiling, covers two repeats of
all 72 questions and three context arms, and shuffles paired execution order.
It compares complete ranked views, selected source metadata, prompt bytes and
token counts. Whole CPU compilation and process CPU time are recorded separately.
There are no model calls, answer labels, new accuracy scores or fine-tuning.

Three earlier single-question probes are retained: explicit tokenizer-segment
caching, unindexed SQL scope sorting, and a temporary metadata-index query plan.
None established a useful lookup/counting improvement. The full comparison keeps
the original retrieval implementation unchanged.

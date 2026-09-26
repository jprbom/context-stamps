# Structured local memory

The structural adapter keeps recorded parent/child relationships together when preparing memory for a local reader. A settings control travels with its recorded options and ancestor containers when they fit the configured budget. Repeated observations with the same projected content and location share one view, with a separate source occurrence for every observation.

This is a candidate component for locally improving small-model applications. It is not a model-weight update or an automatically qualified memory policy. The two-reader [development experiment](../evidence/lme-structure-v1/README.md) measures what changes when it replaces ordinary text chunks.

## Use it offline

No GPU or dependency is required for the adapter:

```python
from context_stamps.trajectory import ObservedEpisode, ObservedStep
from context_stamps.structured_memory import structured_views

observation = """RootWebArea 'Project settings'
    region 'Build'
        [11] combobox 'Test mode' value='Full'
            [12] option 'Full', selected=True
            [13] option 'Quick', selected=False"""

episode = ObservedEpisode("local-session", "Inspect build settings", (
    ObservedStep(0, observation, location="local/project/settings"),
))
for view in structured_views(episode, chunk_chars=2400):
    print(view.text)
    print(view.occurrences)  # Exact step, source line range and ancestor lines.
    node = view.canonical_node(tenant="local", roles=("owner",), observed_at=1)
    assert node.kind == "DERIVED_RESULT"
```

The complete [example](../examples/structured_local_memory.py) also demonstrates deduplication across two observations. The caller can put these nodes into the existing authorized state, working set and compiler APIs. The host supplies ingestion time and access roles; the adapter does not infer source event times or grants.

## What is preserved

The input is an indentation-based text observation, such as an accessibility tree. Tabs expand to four columns. The adapter groups whole subtrees or contiguous sibling subtrees, repeating their full recorded ancestor path when a large container must be split. Only leading instance handles such as `[11]` and indentation are projected. Names, values, selected/expanded flags and all other nonblank source text remain available across the views.

Each occurrence records one-based inclusive source lines and ancestor line numbers. The complete accepted episode has a SHA-256 revision; every view binds to that revision. The original episode is still necessary for inspection and reconstruction. Neither this digest nor the spherical 32-byte capsule contains the source text.

Equal projected bodies are combined only within an episode and at the same location. Their discrete occurrences do not establish continuous validity between observations. A changed value or location remains distinct. A changed episode produces a different source binding even if one projected body stays the same.

This preserves *recorded structure*. It does not infer a table's column-to-cell semantics, restore missing accessibility data, resolve action targets, prove a successful procedure or establish causation. Long containers can still be split. The full collection covers the source; an individual selected prompt may omit necessary evidence.

## Bounds and trust

The existing episode limits apply: 512 steps, 2 MiB per observation and 16 MiB of observation text per episode. Additional limits are 65,536 lines per observation, 96 ancestor levels, 49,152 UTF-8 bytes per projected body and 4,096 occurrences per view. `chunk_chars` accepts 256–8,192 characters and limits subtree grouping before ancestry is added. A single larger line, excessive ancestry or excessive occurrences raises `ValueError` before any view is yielded. Use an explicitly logged raw fallback; do not silently discard the rest of an episode.

All nodes are `DERIVED_RESULT` with no verified claims. The adapter executes no action, follows no URL and evaluates no source code. Imported text remains untrusted and can contain prompt injection. Removing instance IDs is not a security filter or sufficient defense against malicious memory.

Retain source data under appropriate local access controls. On revocation, deletion or source replacement, invalidate/supersede old nodes and update the index. A new hash does not revoke an old stored node automatically. The public experiment uses an isolated public-data tenant; its SQLite index is not a multi-tenant authorization service.

## Run the local comparison

Use the public data and tokenizer preparation in the [RTX runbook](local-trajectory-memory.md). Install NumPy and `tokenizers` locally. The experiment requires the installed Qwen2.5 1.5B and Qwen2.5-Coder 7B GGUF digests recorded in its registration; no model is downloaded implicitly and remote-backed entries are refused. Both use the same verified Qwen tokenizer bytes.

The original raw index must already exist. It can be created by the earlier runner's **prepare** command; its model-generation run is not needed to prepare this comparison. The local output directories must be new and outside the source repository.

```powershell
$env:PYTHONPATH='.'
python experiments/lme_structure.py prepare --data ../longmemeval-v2-data --previous ../lme-memory-run --tokenizer ../qwen25-coder-7b-tokenizer --output ../lme-structure-run
python experiments/verify_structure_projection.py --data ../longmemeval-v2-data --run ../lme-structure-run --output ../lme-structure-run/projection-check.json
# Begin with an idle local Ollama server. One reader runs at a time.
python experiments/lme_structure.py run --output ../lme-structure-run
python experiments/lme_structure.py evaluate --output ../lme-structure-run --scorer evidence/lme-memory-v1/upstream/qa_eval_metrics.py
```

Preparation freezes 72 questions: twelve from each domain/ability cell by a fixed hash order. All are development material from the already inspected benchmark. Two readers each answer both arms, producing 288 measured calls plus two warmups. Within each reader, arm order is shuffled per question. Readers run sequentially in a fixed order. Every full prompt is at most 6,144 tokens; outputs are capped at 256, with temperature zero and seed 71. Context capacity is 8,192 for both readers. Both baselines are freshly measured under this configuration.

Both arms use lexical ranking, the same prompt, a 32-candidate inspection limit and at most 16 selected views. The structural index deduplicates repeated projections and carries goal text for indexing. This is a combined representation/retrieval treatment, not an ablation isolating only hierarchy. The 7B reader also differs in training specialization, so a difference between readers cannot be attributed to model size alone.

Native deterministic scoring runs only after both logs close. It excludes image and judge-required questions; native phrase matching can award credit despite extra or contradictory prose. Precomputed retrieval/packing times and generation times are separate measurements. Summing them is a stage-cost estimate, not a live end-to-end latency measurement. Loading, index construction, peak RAM, energy and concurrent deployment require separate accounting.

## Local improvement after this experiment

Keep the active policy unchanged while testing this candidate. A deployment needs a future-work adaptation cohort and an independent protected older-task cohort, frozen before evaluating the candidate. Score outcomes with trusted task checks or reviewed labels, not the model's confidence in its own answer. Train a selector or small adapter only on the permitted training partition.

Use the [local learning registry](local-domain-learning.md) to evaluate quality, regressions, deadlines and cost, including repeated-experiment risk accounting. Activation and rollback remain explicit. The lightweight adapter can run on a CPU; optional weight training belongs on hardware with adequate memory. No claim is made that every supported SLM can improve its weights locally, or that this experiment qualifies any autonomous update.

The next useful controls are hierarchy versus deduplication separately, source selection versus reader use, action/transition evidence, and independent retention on a second environment. Related primary work includes [Agent Workflow Memory](https://arxiv.org/abs/2409.07429), which studies reuse of workflows, and [LongMemEval-V2](https://arxiv.org/abs/2605.12493), which evaluates memory of environment experience. These are research context, not matched score comparisons with this local protocol.

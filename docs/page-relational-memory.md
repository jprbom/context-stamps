# Page selection and source-bound relation packets

By Prashant Jagtap

The earlier structural-memory experiment preserved source ancestry but still selected material from several pages and gave the readers verbose UI trees. This follow-up separates page selection from a new relation representation. Both are candidate components for the broader context runtime, not a complete memory system or a qualified autonomous learner.

## Use the adapter

The adapter runs locally on a CPU without third-party dependencies:

```python
from context_stamps.trajectory import ObservedEpisode, ObservedStep
from context_stamps.observation_packets import observation_packets, plan_pages

source = """RootWebArea 'Inventory'
    table 'Available stock'
        row ''
            columnheader 'Item'
            columnheader 'Quantity'
            columnheader 'Location'"""
episode = ObservedEpisode("inventory-session", "Inspect inventory", (
    ObservedStep(0, source, location="local/inventory"),
))
packets = tuple(observation_packets(episode))
packet = packets[0]
print(packet.page)
print(packet.body)
print(packet.occurrences)  # Step and exact source-line references.
node = packet.canonical_node(tenant="local", roles=("owner",), observed_at=1)
assert node.kind == "DERIVED_RESULT" and not node.claims

# Supply only titles the current caller is authorized to see.
plan = plan_pages("Inspect Inventory", tuple(p.page for p in packets))
print(plan.pages, plan.fallback)
```

`page` records the enclosing `RootWebArea` labels, including nested frames. Ordered packets retain the source order of row cells, control options, menu items or tabs. A local structural/named ancestor and its line number distinguish containers within the same page. Other named elements keep their raw label and attributes. Quoted source text is parsed as data, never evaluated as Python or executed as an action.

The representation omits empty wrappers and repeated text copies. It is intentionally lossy. Source order is not a verified visual left/right relation, and a recorded ancestor is not automatically an accessible field label. The adapter does not infer table colspans, semantic column mappings, action effects or successful procedures. Retain the original source and inspect it when a packet does not establish the required relationship.

Each packet binds to the full accepted episode revision. A source mutation changes its identity. Occurrences retain exact source-line identities; they do not establish continuous validity between observations or a wall-clock ordering between episodes. The 256-bit spherical capsule remains a separate routing representation referencing external evidence.

## Page affinity

The page planner uses only the question and authorized title paths. It computes corpus document frequencies, excludes title terms present in more than 10% of the distinct pages, and weights the remaining query/title matches by inverse document frequency. A length adjustment reduces a long title's opportunity to win through incidental matches. Pages within half of the best affinity are retained. No usable anchor produces an explicit fallback.

The threshold is a fixed experimental choice, not a learned confidence boundary. Scores are not probabilities or sufficiency estimates. The current tokenizer uses English letter/digit terms and a limited plural heuristic; multilingual scope selection is unqualified. Page names can be ambiguous, stale or attacker-controlled. A plausible title match does not establish authority or correct scope.

`PagePlan` never grants access. The caller must authorize sources before supplying titles or indexing evidence, then check current access again at use. A multi-tenant service should isolate indexes or implement authorization throughout retrieval; the public benchmark's isolated tenant is not such a service.

## Local three-arm comparison

The [development evidence](../evidence/lme-relations-v2/README.md) compares:

1. The unchanged structural-memory baseline.
2. Page-filtered structural memory, with the same representation and packing limits.
3. Page-filtered relation packets, with grouped page headers and a larger small-packet inspection limit.

The third arm changes representation, duplicate handling and packing limits together. It does not isolate each mechanism. Ordered source relationships are checked independently; they are not promoted to verified factual claims.

The same 72 previously inspected LongMemEval-V2 development questions are used for two installed readers: Qwen2.5 1.5B and Qwen2.5-Coder 7B, both Q4_K_M. Each reader answers all three arms, yielding 432 measured requests. The structural baseline prompts must exactly match the preceding study during preparation. Each model comparison is freshly measured under the same configuration.

Every complete input is capped at 6,144 tokens, with 256 output tokens and an 8,192-token context capacity. Structural arms inspect at most 32 candidate views and select at most 16. The relation arm inspects at most 160 packets and selects at most 96 within the same token ceiling. It omits duplicate page/body content from the prompt while retaining the selected observation's exact source binding.

For each actual request, the runner performs retrieval and context compilation again, checks exact equality with the registered prompt, and then invokes the local reader. It records both component times and the enclosing request duration. Index building, reader loading and two warmups are separate. This is a warm sequential workstation experiment, not a concurrency, energy or target-edge-device benchmark.

Native scoring occurs after both model logs close. Image and judge-required cases remain excluded. All errors, truncations and regressions are retained. The inspected questions and shared histories cannot establish independent generalization, older-task retention or qualification for local activation.

## Reproduce on the local RTX setup

Use the data, tokenizer and installed-model preparation from the [structural-memory runbook](structured-local-memory.md). The runner checks the exact local GGUF model digests and refuses remote-backed models. It requires an idle Ollama server and runs only one reader at a time.

```powershell
$env:PYTHONPATH='.'
python experiments/lme_relations.py prepare --data ../longmemeval-v2-data --previous ../lme-structure-run --tokenizer ../qwen25-coder-7b-tokenizer --output ../lme-relations-run
python experiments/lme_relations.py run --output ../lme-relations-run --tokenizer ../qwen25-coder-7b-tokenizer
python experiments/verify_relation_sources.py --data ../longmemeval-v2-data --run ../lme-relations-run --output ../lme-relations-run/source-audit.json
python experiments/lme_relations.py evaluate --output ../lme-relations-run --scorer evidence/lme-memory-v1/upstream/qa_eval_metrics.py
python experiments/review_lme_relations.py --run ../lme-relations-run --scorer evidence/lme-memory-v1/upstream/qa_eval_metrics.py --output ../lme-relations-run/failure-review.json
```

Run the heavier source audit after inference if measuring request latency. Use fresh output directories outside the repository; partial runs must not be overwritten. The original preparation is retained separately because its ordered relations omitted local container ownership. It made no reader calls and was corrected before this comparison.

## Limits for local improvement

The observation/episode bounds remain in force. Parsing accepts at most 65,536 lines and 96 ancestor levels per observation; single source lines are bounded to 8,192 characters. Each retained packet body is at most 16,384 UTF-8 bytes. Ordered relations are bounded to 1,024 members. An oversized ordered relation falls back to individual elements rather than declaring a partial list complete. Other exceeded bounds raise `ValueError` before any packets are yielded. The experiment records episodes whose relation extraction fails; it never drops their evaluation questions.

This code does not run a background learner or fine-tune model weights. A host can use measured, independently checked outcomes to propose a context policy or adapter, then evaluate it using separate adaptation and retention cohorts in the [local learning registry](local-domain-learning.md). Native phrase-match credit alone can include false positives; it must not be confused with verified production outcomes. Source access, poisoning resistance, deletion, rollback and target-device cost remain application responsibilities and qualification requirements.

Related primary work includes [RAPTOR](https://arxiv.org/abs/2401.18059), which studies tree-organized retrieval, and [SQLite FTS5](https://www.sqlite.org/fts5.html), which provides the lexical ranking used here. This experiment is not a matched comparison against RAPTOR or a claim that hierarchical retrieval is unprecedented.

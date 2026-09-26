# Context Stamps — Spherical Context QR

**Compact context references, exact-first retrieval and version-checked evidence handoffs.**

By **Prashant Jagtap** · Python 3.10+ · MIT-licensed core · **Public research release v0.5.0**

The source repository is public. No PyPI release is available; install from a reviewed clone. This release keeps the spherical stamp at **256 bits / 32 bytes** and defaults to a precise retrieval backend when a compact shortcut has not qualified on validation data.

![Animated Spherical Context QR workflow](docs/assets/spherical-context-flow-animated.svg)

The 32-byte capsule is a routing identity across eight bounded facets. It does not contain the document or the relationship graph. A resolver uses the capsule to narrow activation, then returns current authorized evidence through an exact, certified compact or precise path. The [interactive reference workflow](docs/assets/spherical-context-workflow.html) exposes each stage and relationship.

## Start here

**Active research direction:** an [Enterprise Context Intelligence Runtime](docs/enterprise-context-plan.md) that owns versioned evidence, compiles sufficient authorized context for interchangeable models, and records why each decision used it. The [requirement register](evidence/enterprise-context-v1/requirements.json) separates existing components from unfinished work. This is a development programme, not a 1.0 capability claim.

On the research branch, the [temporal state, compiler and typed-decision APIs](docs/enterprise-state.md) now support historical knowledge cutoffs, verified evidence kinds, finite negative knowledge, conflict handling, dependency-aware context selection and batched Boolean/Choice/Score results. Run `python examples/enterprise_decision.py` for an offline example. These are tested engineering interfaces, not a newly trained model or a public benchmark gain.

The [longer RTX precision study](docs/rtx-precision-capacity.md) records eight training-only capacity windows. BF16 did not consistently improve throughput and changed some ranking orders. These probe weights were discarded; serving precision and existing quality claims are unchanged.

The [durable audit API](docs/durable-audit.md) records typed plans, outcomes and failures with actor-bound receipts. The [managed executor](docs/managed-execution.md) commits exclusive dispatch claims, runs registered adapters/verifiers in cancellable processes, reserves outcome capacity and preserves uncertain effects for provider reconciliation. Run `python examples/managed_decision.py` for that offline path. Process startup adds latency; this is an optional execution mode, not a model-quality or speed improvement.

The new [persistent context store and working set](docs/persistent-context.md) preserve temporal versions/ACLs across restarts and load bounded authorized dependency closures. Run `python examples/persistent_context.py` for persistence through a managed typed decision and audit receipt. Its storage revision recorded 267 passing tests. A 540-check fictional replay reduced payload reads by 47.5%–72.5% with working-set reuse; explicit prefetch was slower than the working set alone and remains opt-in. These are storage measurements, not model-token or public-retrieval gains.

The [incremental computation layer](docs/incremental-computation.md) now recomputes affected active branches and reuses unrelated exact results with current access checks and bound receipts. Run `python examples/incremental_metrics.py`. The current full local suite passes **290 tests with zero skips**; all 720 requests in the fictional arithmetic replay returned expected results. Localized changes needed fewer callbacks; policy-wide changes forced full recomputation and added overhead. This is workflow engineering evidence, not a new model benchmark or token-savings claim.

The [expanded evaluation programme](docs/evaluation-programme.md) specifies paired model/runtime comparisons using established harnesses. A local Inspect pilot runner and seven boundary tests are available; its first preparation was blocked by Windows Application Control before generation. No new native benchmark score or frontier-model comparison is claimed. Frontier calls remain disabled for this phase.

The local **`ContextRuntime`** now combines registered retrieval experts, dependency checks, explicit byte/token budgets, bounded missing-evidence recovery and verified exact-result reuse. Run `python examples/unified_context.py` after installation. [API and complete example](docs/unified-runtime.md) · [runtime results and retained failures](docs/runtime-v1-results.md).

The optional cross-encoder uses length-aware GPU batches and a bounded exact passage-token cache. Its candidates, model weights and evidence limit are preserved. The learned cheaper-expert policy failed calibration and remains disabled. Precision changes require numerical and ranking checks; lower bit width alone does not earn a speed claim.

![Paired runtime latency](docs/assets/runtime-v1-latency.png)

![Complete runtime comparison table](docs/assets/runtime-v1-table.png)

![Two local readers and verified workflow outcomes](docs/assets/runtime-v1-readers.png)

![Spherical same-budget fixture and exact-metadata control](docs/assets/runtime-v1-facets.png)

**26 September follow-up:** nine new RTX training runs test balanced sampling, metric-aware ranking, teacher distillation and contractive recurrence. On a fresh local **FiQA** test, a frozen hybrid/cross-encoder blend reaches **0.4125 nDCG@10 versus 0.3687 dense and 0.3888 hybrid**. This gain comes from precise reranking with external text and embeddings. The selected small student still trails hybrid on four of five collections and remains experimental. [Complete results and remaining gaps](docs/controller-v2-results.md) · [research basis and local RTX commands](docs/controller-methodology-v2.md) · [checkpoints and model card](evidence/controller-v2/MODEL_CARD.md). The earlier [six-run study and failures](docs/local-rtx-controller.md) remain intact.

A separate local Qwen pilot tested exact computation reuse through source edits and policy changes. With 50% repeated computations, model calls and processed input tokens fell by 50%, and total elapsed time fell by 47.2%, with 216/216 correct outputs per mode. **p95 latency did not improve** (114.9 ms → 117.1 ms). These are 12 fictional scalar-extraction tasks repeated for timing, not a broad agent or coding benchmark. [Raw observations](evidence/computation-v1/results.json) · [summary](evidence/computation-v1/summary.json).

```bash
git clone https://github.com/jprbom/context-stamps.git
cd context-stamps
python -m pip install -e .
python examples/unified_context.py
python examples/progressive_context.py
python examples/automatic_facets.py
python examples/zip_spherical_qr.py
```

These examples run offline without a GPU, service or downloaded model. The unified example checks a complete dependency packet, verifies a reusable result and revokes the receipt after a change. The subsequent examples show precise fallback and spherical stamp construction. The core has no mandatory third-party dependencies.

## What the components do

| Component | Purpose | Important boundary |
|---|---|---|
| `ContextRuntime` | Composes experts, budgets, dependency closure, bounded verification and exact reuse | Trusted host adapters/verifier; callbacks need their own timeout; not a distributed service |
| `ManagedExecutor` / `AuditStore` | Owns process dispatch, verification, cancellation and reference-only outcome history | Trusted host adapters; no sandbox or distributed exactly-once guarantee; external effects need provider reconciliation |
| `ContextStore` / `ContextWorkingSet` | Retains exact temporal versions and current ACLs; pages authorized dependency closures into bounded working copies | Local metadata catalogue, host authentication/key custody; eviction does not implement legal retention or physical deletion |
| Optional `EfficientReranker` | Uses length-aware batches and exact passage-token reuse for a pinned BERT scorer | Preserve candidates and pair truncation; qualify ranking parity for the intended profile |
| `Stamp256Codec` | Encodes supplied content/entity/intent/task views into 32 raw bytes | Lossy similarity sketch; schema and evidence are external |
| `FacetCompiler` | Emits observed routing facets with rule and source provenance | Deterministic baseline; not a semantic parser or fact verifier |
| `RelationMap` | Converts typed, directional links into a bounded diffusion view | Lossy feature input; the source graph remains external and versioned |
| `HybridScoreProfile` | Fuses standardized semantic and lexical scores in a precise backend | A frozen global blend can regress; use a scope certificate |
| `ScopeCertificate` | Enables a hybrid profile only when a paired validation lower bound clears the requested gain | Validation must be disjoint from final evaluation |
| `ProgressiveRouter` | Resolves an explicit source or calls a precise backend; optionally uses a validated compact exit | Exact IDs and similarity do not grant access |
| `ContextGraph` | Tracks declared relationships, versions and dependency closures | Does not infer missing relationships |
| `ContextSession` | Reuses valid packets using bounded, expiring 32-byte receipts | Receipts are opaque handles, separate from semantic stamps |
| `ComputationIdentity` / `ComputationCache` | Reuses a result only when request, model, prompt, tool, policy, principal and source bindings match | Host owns authorization and complete input capture; bounded in-process cache |
| Optional `RecurrentEvidenceRanker` | Learns a bounded correction over precise retrieval candidates | Experimental; calibration rejected promotion; serving enforces trained recurrence depth |
| Optional `ContractiveRanker` | Adds metric-aware training and a bounded convergent candidate-attention variant | v2 student is experimental; numerical convergence does not prove ranking quality |
| `ResidualIndex` | Uses a richer index and backing vectors for exact refinement | More than 32 bytes; slower than Faiss in our in-memory tests |

The spherical representation is a product of unit spheres followed by binary projection. A graph stores relationships between items; a stamp describes one item through supplied views. “QR” names the portable reference concept, not a visual barcode standard. Neither stamps nor receipts reconstruct arbitrary context or make a language model understand a hash.

For an offline computation-reuse example, run `python examples/computation_reuse.py`. To train the optional ranker, install the `controller` extra into a CUDA-capable environment and follow the [current RTX guide](docs/controller-methodology-v2.md). The tuning-selected v2 student has 98,817 parameters and a 395,708-byte safetensors checkpoint; the encoder and index are external. The earlier v1 int8 export reduces storage and reconstructs FP32; v2 separately measures actual dynamic-int8 CPU operators. Neither quantization nor extra recurrence is enabled by default.

## Latest measured retrieval comparison

The v2 experiment evaluates 3,677 queries across five collections. Method selection is frozen on tuning data; separate calibration accepts fusion for SciFact and FiQA and retains dense elsewhere. All nine neural runs and regressions are published. These are controlled retrieval comparisons, not a benchmark win over entire agent platforms or a claim that 32 bytes contain the source context.

![Five-dataset retrieval results](docs/assets/controller-v2-results-table.png)

![Quality and measured online latency](docs/assets/controller-v2-quality-latency.png)

![Candidate expansion and its oracle ceiling](docs/assets/controller-v2-candidate-ceiling.png)

The oracle is a diagnostic upper bound that uses relevance labels to sort candidates. These controller-v2 latency figures predate the runtime optimization above. Latency measures the local retrieval/reranking stages and excludes query encoding, reader generation, networking and concurrent load. [Detailed interpretation, confidence intervals and limitations](docs/controller-v2-results.md).

The initial FiQA quality gain cost about **194 ms** per retrieval/reranking call versus **6.8 ms** dense. The [new paired execution benchmark](docs/runtime-v1-results.md) measures lower cold/warm reranking latency while retaining ranking gates. Split-precision student int8 reduces measured score distortion 28–122× but remains slower than FP32 in its small CPU forward test. No broad recursive-intelligence or production-scale claim follows. [Architecture and remaining milestones](docs/unified-context-roadmap.md).

```mermaid
flowchart TD
    Q[Request and host-authorized candidates] --> E{Explicit source ID?}
    E -->|Available| X[Exact resolution]
    E -->|Unavailable| A[Abstain]
    E -->|No| P{Validated compact policy for this scope?}
    P -->|No| D[Application precise retriever]
    P -->|Yes| S[Compare spherical stamps]
    S --> C{Score and margin accepted?}
    C -->|No| D
    C -->|Yes| V[Check evidence and dependencies]
    D --> V
    X --> V
    V --> H[Issue or reuse an evidence packet]
    U[Source or permission change] --> I[Invalidate receipts]
```

## Use your existing retriever

```python
from context_stamps import ProgressiveRouter

def precise(eligible, limit):
    # Replace these demonstration scores with your dense/hybrid search.
    scores = {"implementation": 0.9, "requirement": 0.7}
    return sorted(((key, scores[key]) for key in eligible),
                  key=lambda row: (-row[1], row[0]))[:limit]

router = ProgressiveRouter()
result = router.search(
    eligible=["implementation", "requirement"],
    precise=precise, scope="project:encoder-revision:schema-revision", limit=1,
)
assert result["route"] == "precise"  # No compact policy is enabled by default.
assert result["reason"] == "compact_backend_unavailable"
print(result["ids"])
```

The host supplies current authorized IDs. An explicit `exact_key` bypasses retrieval when available and abstains when unavailable. Compact policies need disjoint calibration and scope binding. The router also requires the certified upper error bound to fit the caller's `maximum_compact_error` budget. No public-data policy qualified in the latest experiment. Do not turn a similarity threshold into a claim of confidence. [Routing API and calibration](docs/progressive-routing.md).

## Compile observed facets and inspect their provenance

```python
from context_stamps import FacetCompiler

compiled = FacetCompiler().compile(
    "Update api/router.py because Router.search() depends on policy.py v2.1.",
    metadata={"authority": "maintainer", "policy": "internal", "modality": "code"},
)
for name, item in compiled.evidence.items():
    print(name, item.value, item.source, item.rule)
```

The compiler extracts bounded identifiers, action terms, declared relations and versions. Authority, policy and modality are host assertions. Missing facets are omitted. The baseline is deterministic and inspectable so extraction errors can be measured before a learned extractor is introduced. The standard research profile assigns 96/32/32/32/16/16/16/16 bits to semantic, task, entity, relation, temporal, authority, policy and modality views. This allocation is a testable profile, not an optimized result. [Adaptive capsule design and controls](docs/adaptive-capsules.md).

![Adaptive 256-bit capsule design](docs/assets/adaptive-capsule.png)

## Add typed relationships without putting the graph in the stamp

```python
from context_stamps import RelationEdge, RelationMap

relations = RelationMap([
    RelationEdge("implementation", "requirement", "implements", 2.0),
    RelationEdge("implementation", "test", "verified_by", 1.0),
    RelationEdge("test", "fixture", "uses", 1.0),
])
relation_view = relations.encode("implementation", dim=64, hops=3)
assert abs(sum(value * value for value in relation_view) - 1.0) < 1e-12
print(relations.revision)  # Changes when an edge changes.
```

The encoder performs bounded personalized diffusion, then hashes typed forward and reverse edges into a normalized feature vector. Each facet can be centered and fitted with its own orthogonal ITQ rotation before its assigned bits are packed into the 256-bit capsule. This combines established diffusion, feature-hashing and ITQ mechanisms in a relation-aware routing architecture; it does not claim a new graph algorithm. [Mathematical definition, limits and use](docs/relation-aware-retrieval.md).

![Static Spherical Context QR reference architecture](docs/assets/spherical-context-cover.png)

## Reuse evidence across workflow steps

```python
from context_stamps import ContextNode, ContextSession

session = ContextSession()
session.put(ContextNode("implementation", "TIMEOUT = 10", "v1", frozenset({"developer"})))
session.put(ContextNode("requirement", "Timeout must be ten seconds.", "v1", frozenset({"developer"})))
session.link("implementation", "requirement", "depends_on", provenance="review-1")
versions = {"implementation": "v1", "requirement": "v1"}

packet, receipt, reused = session.issue(
    ["implementation"], role="developer", revisions=versions,
)
assert packet.status == "complete" and len(receipt) == 32
again, same_receipt, reused = session.issue(
    ["implementation"], role="developer", revisions=versions,
)
assert reused and again == packet and same_receipt == receipt

session.put(ContextNode("requirement", "Timeout must be twenty seconds.", "v2", frozenset({"developer"})))
assert session.resolve(receipt, role="developer", revisions=versions).status == "insufficient"
```

A source update invalidates receipts containing that source; a relationship update invalidates packets containing its source endpoint. Unrelated receipts remain valid. A packet includes the complete declared dependency closure or returns insufficient context. Host authorization, source versions and relationship correctness remain application responsibilities. The cache is process-local and serialized; it is not a distributed memory service. Resolved evidence still consumes model input tokens when sent to a model.

## Partial queries and changes to shared memory

`FacetQuery` compares only observed query views, without fabricating missing entity or task values. Its policy scope includes the facet mask, encoder families and weights. This prevents accidental reuse of a full-facet calibration for a different query shape; scores remain uncalibrated ranking utilities.

```bash
python examples/partial_facets.py
```

Selective invalidation was checked against an uncached graph across 4,000 mutation comparisons. A separate 6,600-handoff replay compared exact graph lookup, the old global cache and the new selective cache. At 100 independent declared pairs with one changed pair per round, the new cache reused **1,881 of 2,000 packets**, versus zero for the old cache. All returned packet hashes matched the uncached control. The larger case repeats actual source text under synthetic IDs; it is not 100 real agent tasks. [Results, API and remaining gaps](docs/selective-context.md).

![Selective invalidation and packet reuse](docs/assets/selective-invalidation.png)

## Create a 32-byte stamp

```python
from context_stamps import Family, HashingEncoder, Stamp256Codec

encoder = HashingEncoder(64)
facets = {"content": "Update timeout", "entity": "worker_alpha",
          "intent": "implement", "task": "timeout"}
codec = Stamp256Codec({name: Family(encoder.identity, 64, 64, 17 + i)
                       for i, name in enumerate(facets)})
stamp = codec.encode({name: encoder.encode(value) for name, value in facets.items()})
raw = codec.pack(stamp)
assert len(raw) == 32
assert codec.unpack(raw, schema_id=codec.schema.identity) == stamp
```

The hashing encoder is a lexical demonstration. Encoder/schema identity, exact source identity, versions, permissions and original evidence are outside the 32 bytes. Base64 and schema envelopes add transport bytes. Arbitrary context cannot be losslessly reduced to this stamp. [Format and bit allocation](docs/zip-spherical-qr.md).

## Latest measured retrieval results

The compact-only result remains below dense retrieval. Across 2,029 previously inspected public test queries, trained 32-byte ITQ reached 0.5453/0.2501/0.4181 nDCG@10 on SciFact/NFCorpus/ArguAna, versus 0.6451/0.3167/0.5041 for dense MiniLM. Shrinking to 16 bytes worsened all three datasets. Expanding to 64 bytes helped but still remained below dense. The 256-bit profile is therefore a routing and cache key, not a replacement for precise evidence recovery. [Compact training and calibration](docs/progressive-routing.md).

![Compact retrieval training across three public datasets](docs/assets/progressive-retrieval.png)

The new precise path standardizes MiniLM cosine and BM25 scores over the eligible corpus and applies a weight selected on 146 SciFact validation queries: `0.75 × semantic + 0.25 × lexical`. The first three test sets had already been inspected during exploration. SciDocs was downloaded, hashed and named in the frozen protocol before its retrieval outcome was computed.

| Precise method, nDCG@10 | SciFact | NFCorpus | ArguAna | SciDocs |
|---|---:|---:|---:|---:|
| Dense MiniLM | 0.6451 | 0.3167 | 0.5041 | **0.2164** |
| BM25 | 0.6646 | 0.3103 | 0.4656 | 0.1504 |
| Frozen hybrid, 0.75/0.25 | **0.7225** | **0.3503** | **0.5385** | 0.2043 |
| Hybrid − dense | +0.0774 | +0.0337 | +0.0344 | **−0.0121** |
| Paired 95% interval | +0.0510 to +0.1072 | +0.0169 to +0.0512 | +0.0218 to +0.0467 | **−0.0194 to −0.0046** |

![Dense, BM25 and frozen hybrid retrieval](docs/assets/hybrid-retrieval-comparison.png)

SciDocs is a clear prospective failure of the global blend. It rules out a universal-superiority claim. `certify_hybrid_scope` now issues a scope-bound profile only when a paired bootstrap lower bound on disjoint validation queries exceeds the requested minimum gain. A failed or missing certificate selects dense retrieval. This makes the supported behavior hybrid where validated and dense where the profile abstains; it does not prove that a new unseen scope will improve.

```python
from context_stamps import HybridScoreProfile, certify_hybrid_scope

profile = HybridScoreProfile(semantic_weight=0.75)
certificate = certify_hybrid_scope(
    "corpus-and-encoder-revision",
    profile,
    dense_validation_ndcg,
    hybrid_validation_ndcg,
)
if certificate.enabled:
    scores = profile.fuse(dense_scores, bm25_scores)
else:
    scores = dense_scores
```

Median measured query time for the hybrid path was 1.125/0.626/6.662/5.837 ms, versus 0.386/0.277/0.763/2.121 ms for dense score lookup on SciFact/NFCorpus/ArguAna/SciDocs. Encoder inference, BM25 index construction, network time and capsule candidate-generation time were excluded. The quality gains on three datasets therefore carry real CPU and memory overhead. [Protocol, results, per-query records and checksums](evidence/hybrid-retrieval-v1/).

## Workflow and transport evidence

- **Repeated packet replay:** 20 actual source files, ten declared pairs, 200 deliveries. Reuse accounted for 83,072 transfer bytes versus 1,533,440 for full packets, a 94.6% reduction with a populated shared resolver. Network/authentication costs were excluded. Both methods resolved the same evidence, so this is not a model-token saving. It is not a completed coding-agent study.
- **Earlier local SLM pilot:** eight fictional workflows, two repeats. Selected context succeeded in 16/16 versus 10/16 for full context, with 82.79% fewer input tokens and 25.71% lower mean workflow time. Exact graph lookup also succeeded in 16/16 and was faster on average. [Controls and failed iterations](docs/retrieval-repair.md).
- **Image, speech and video pilots:** 36 generations, 18 direct/routed pairs, identical outputs. Routing added overhead and did not reduce generator tokens or improve quality. [Historical spherical results](docs/spherical-results.md).

Real unseen coding/research tasks, learned facet extraction, causal benefit from the structured 256-bit profile, cross-scope certificate transfer, internal attention changes, distributed scalability and mobile energy savings remain unproven. The bounded reference graph supports 1,000 nodes and 4,096 relationships. Earlier read-only 1k/10k/100k-vector tests favor Faiss over residual refinement for latency and throughput. SciDocs demonstrates that a fixed lexical blend can make a strong dense reference worse.

## Integration and training

| Entry point | Current support |
|---|---|
| Python library | Progressive routing, spherical stamps, graph and reusable sessions |
| `scqr` CLI | Payload encode/inspect; not a complete routing service |
| `cstamps`, MCP, portable skill/Markdown | Existing SQLite evidence workflow; not automatic wrappers around every new API |
| Standalone `stamps.py` | Projection and similarity primitives |

[Integration guide](docs/integrations.md) · [agent instructions](docs/agent-instructions.md) · [older SQLite usage](docs/historical-v02-guide.md).

The wheel's three optional pairwise facet scorers are historical procedural models, not the newly evaluated ITQ quantizers. They require their exact lexical family schema and do not supply calibrated probabilities. The ITQ artifacts and their dataset-derived licenses are under `evidence/quantizer-seeds-v1`. Neither model family is enabled automatically.

```bash
python -m pip install -e ".[dev,mcp,tokens]"
python -m unittest discover -s tests -v
python experiments/verify_progressive.py
python examples/progressive_context.py
```

Local validation covers mutation, extraction, round trips, numerical recurrence, optional quantization, runtime budgets, permissions, expert abstention and token-cache/truncation boundaries. CI runs Windows/Linux, optional Torch and security checks on each candidate commit. Offline verifiers replay historical and current per-query evidence, including preserved failures. Full numerical benchmark replay still needs pinned external corpora and embedding caches. [Training protocol](program.md) · [validation](docs/validation.md) · [scenario matrix](docs/scenario-matrix.md) · [remaining gaps](docs/unified-context-roadmap.md).

## Security, data and credit

The host supplies authentication and permissions. Stamps can expose similarity; they are not encryption. Retrieved content remains untrusted and may contain prompt injection. Receipt revocation cannot recall previously delivered plaintext. [Security policy](SECURITY.md).

Secret scanning retains all default detection rules. Seven exact public-ID pairs in one evidence file are narrowly excluded; synthetic positive controls confirm credentials remain detectable on the same line. Historical results and unsuccessful experiments remain available.

Copyright © 2026 **Prashant Jagtap**. Preserve the copyright and MIT permission notice when redistributing substantial portions of the code. Citation is appreciated, not an additional MIT restriction. Public-data-derived records and quantizers retain their stated CC-BY-SA-4.0 terms; external models retain their own licenses. Private corpora and base-model weights are not distributed. [License](LICENSE) · [notice](NOTICE.md) · [rights and data](docs/rights-and-data.md) · [citation](CITATION.cff).

# Comparison and intended use

## Latest measured comparisons

The runtime follow-up retains every measured full-fusion top-10 list on 3,677 previously inspected queries while reducing FiQA median reranking-stage time from191.34ms to115.95ms cold/89.17ms warm passage-token cache. It also compares two local readers with full scope, prepared context and verified reuse, and includes exact-metadata and exact-calculator controls. The trained cheap-expert route failed calibration and stays disabled. [Paired measurements and limits](runtime-v1-results.md) · [unified API](unified-runtime.md). These remain local comparisons rather than wins against complete commercial platforms.

The preceding controller study trains nine small models and compares dense, BM25, hybrid, a frozen cross-encoder and fixed score fusion over five datasets. FiQA was a fresh local transfer test in that round: fusion reaches 0.4125 nDCG@10 versus 0.3687 dense and 0.3888 hybrid. It is regression data for subsequent work. The selected student remains below hybrid on four of five datasets. Fusion regresses on SciDocs; the frozen gate uses dense there. [Results and confidence intervals](controller-v2-results.md) · [RTX reproduction](controller-methodology-v2.md). These are lightweight references, not the newest frontier retrieval leaderboard. No industry-wide superiority is claimed.

The earlier [six-run controller study](local-rtx-controller.md) and exact computation-reuse pilot retain their original outcomes. Computation reuse shows a separate benefit on a repetitive local Qwen workload; no head-to-head comparison with a full commercial agent platform was performed.

The public v0.5.0 research release uses a relation-aware 256-bit product-sphere capsule with exact-first routing and precise fallback. Compact-only three-seed ITQ means remain below dense retrieval. A separate precise MiniLM/BM25 blend improves nDCG@10 on SciFact, NFCorpus and ArguAna, then regresses on the prospective SciDocs check. The implementation now requires a positive scope certificate or uses dense retrieval. Faiss remains the stronger measured in-memory latency baseline for conventional vector indexing.

The latest changing-context replay compares an uncached exact graph, ordinary global cache invalidation and dependency-selective invalidation. All packets match; selective invalidation preserves unrelated cache hits. This is a local packet-handling comparison, not a win over complete third-party agent platforms. [Full comparison](selective-context.md). The feature review and older experiment descriptions below retain their original scope.

Reviewed 2026-09-20. Feature descriptions are drawn from the linked projects. Measured comparisons are in [experimental results](experiments.md); no performance comparison with an entire memory platform is implied.

| Need | Context Stamps | Relevant alternative | Decision guidance |
|---|---|---|---|
| Start locally without a service or downloaded model | Standard-library core; `python stamps.py --demo` | [Faiss](https://github.com/facebookresearch/faiss) provides optimized vector indexes | Use Context Stamps for a small inspectable evidence workflow; use Faiss when index scale and retrieval throughput dominate. |
| Keep changed code and instructions visible | Exact digest separate from approximate similarity; explicit source/dependency checks | [Graphiti](https://github.com/getzep/graphiti) tracks temporal facts and provenance in graphs | Use Context Stamps for explicitly versioned files and derived plans; use Graphiti for richer temporal relationships and graph queries. |
| Reduce a prompt to fit a budget | Keeps whole original chunks with source headers and omission reasons | [LLMLingua](https://github.com/microsoft/LLMLingua) provides learned prompt compression | Use Context Stamps when intact source text and explicit requirements matter; evaluate compression when finer-grained shortening is acceptable. They can be composed. |
| Persistent personalized agent memory | Explicit source storage; no automatic personal-fact extraction | [Mem0](https://github.com/mem0ai/mem0) provides broader memory infrastructure | Use Context Stamps when the application owns ingestion and version state; consider Mem0 for a broader memory lifecycle. |
| Compact embedding search | Portable projection families and Hamming ranking | [Sentence Transformers quantization](https://huggingface.co/blog/embedding-quantization) supports binary/scalar quantization and reranking | Compact codes are established technology. Context Stamps adds evidence handling around them; it does not claim to invent quantization. |
| Combine semantic and exact-term retrieval | Standardized MiniLM/BM25 fusion with validation-bound admission and dense fallback | [Elasticsearch hybrid search](https://www.elastic.co/guide/en/elasticsearch/reference/current/semantic-text-hybrid-search.html) and other search stacks combine lexical and vector ranking | Use an optimized search platform for production indexing. The reference code is useful when the admission decision and failed-scope evidence must remain small and inspectable. |
| Route across several context facets | Fixed 256-bit product of semantic, task, entity, relation, temporal, authority, policy and modality views | Multi-vector retrieval systems retain several token or field vectors, usually at a larger storage budget | Use the capsule for a bounded first-stage key; retain multi-vector or dense evidence for precise recovery. No same-budget superiority is established yet. |

## Why it can be easier

One API carries the original text, exact identity, freshness decision and packing explanation. The offline path needs no service account, database server or neural model. Python, CLI, a standalone file, portable agent instructions and MCP support different integration styles. `observe` reads an explicit file list; `select --required` returns no packet if a required source is missing, stale or cannot fit. A small application can adopt these independently without migrating to a full memory platform.

These are scope and usability advantages, not measured installation-time or developer-productivity results. The initial store is limited to 1,000 items and performs a scan. Advanced temporal reasoning, automatic knowledge extraction, large-scale indexing, multi-tenant authorization, encryption and prompt-injection prevention are outside this package.

## What was actually compared

The public-data experiment executes **Faiss IndexFlatIP and IndexBinaryFlat**, a disclosed BM25 reference, classical MMR, a raw diallel cancellation control, coverage/diversity selection and trained linear rerankers. All use the same SciFact corpus and official test queries. Neural methods share a pinned MiniLM encoder and candidate-pool settings recorded in the manifest. BM25 is our transparent reference implementation, not a timing claim about Elasticsearch or another optimized engine.

The synthetic experiment compares exact deduplication, lexical and binary ranking, version checks, coverage/diversity, learned relevance and hard source requirements. It isolates fixture evidence coverage rather than generated-answer quality. The required-source control is given explicit relevant source IDs; it is an upper-bound integration control, not a fair learned-retrieval competitor.

We have not run LLMLingua, Mem0 or Graphiti end to end in this release. Their table entries are sourced feature comparisons. No universal superiority, published-leaderboard equivalence or production cost saving is claimed.

The [frozen follow-up](replication.md) adds all NFCorpus and ArguAna test queries. Coverage/diversity lost to dense retrieval on both, including the low-overlap strata. The SciFact-trained linear selector scored only 0.2168 on ArguAna versus dense retrieval's 0.5014. These regressions narrow the recommendation: use explicit versions, required-source enforcement and inspectable packets where they fit your application; treat reranking and learned selectors as optional experiments.

The [local SLM comparison](local-tasks.md) uses current full context as a strong correctness baseline and an intentionally stale cache as a failure control. It found fewer prompt tokens with selected context, equivalent correctness on the narrow fixtures, and mixed latency. It does not compare whole third-party products. A [developer study kit](usability-study.md) is available; ease of adoption has not yet been independently measured.

## Earlier spherical experiments

The new representation exposes multiple angular views per context item and query-driven activation. An optional graph handles explicit dependencies. This differs in representation from [Graphiti](https://github.com/getzep/graphiti), whose primary abstraction is a temporal knowledge graph. It is not a measured performance advantage over Graphiti. [Charikar’s random-hyperplane similarity method](https://courses.compute.dtu.dk/02289/2022/approxds/charikar.pdf) underlies the angular fingerprints; no invention of angular hashing is claimed. Our supplied-field experiments include an exact-field graph control, which matches the guarded method. See [spherical results](spherical-results.md).

# Comparison and intended use

Reviewed 2026-09-19. Feature descriptions are drawn from the linked projects. Measured comparisons are in [experimental results](experiments.md); no performance comparison with an entire memory platform is implied.

| Need | Context Stamps | Relevant alternative | Decision guidance |
|---|---|---|---|
| Start locally without a service or downloaded model | Standard-library core; `python stamps.py --demo` | [Faiss](https://github.com/facebookresearch/faiss) provides optimized vector indexes | Use Context Stamps for a small inspectable evidence workflow; use Faiss when index scale and retrieval throughput dominate. |
| Keep changed code and instructions visible | Exact digest separate from approximate similarity; explicit source/dependency checks | [Graphiti](https://github.com/getzep/graphiti) tracks temporal facts and provenance in graphs | Use Context Stamps for explicitly versioned files and derived plans; use Graphiti for richer temporal relationships and graph queries. |
| Reduce a prompt to fit a budget | Keeps whole original chunks with source headers and omission reasons | [LLMLingua](https://github.com/microsoft/LLMLingua) provides learned prompt compression | Use Context Stamps when intact source text and explicit requirements matter; evaluate compression when finer-grained shortening is acceptable. They can be composed. |
| Persistent personalized agent memory | Explicit source storage; no automatic personal-fact extraction | [Mem0](https://github.com/mem0ai/mem0) provides broader memory infrastructure | Use Context Stamps when the application owns ingestion and version state; consider Mem0 for a broader memory lifecycle. |
| Compact embedding search | Portable projection families and Hamming ranking | [Sentence Transformers quantization](https://huggingface.co/blog/embedding-quantization) supports binary/scalar quantization and reranking | Compact codes are established technology. Context Stamps adds evidence handling around them; it does not claim to invent quantization. |

## Why it can be easier

One API carries the original text, exact identity, freshness decision and packing explanation. The offline path needs no service account, database server or neural model. Python, CLI, a standalone file, portable agent instructions and MCP support different integration styles. `observe` reads an explicit file list; `select --required` returns no packet if a required source is missing, stale or cannot fit. A small application can adopt these independently without migrating to a full memory platform.

These are scope and usability advantages, not measured installation-time or developer-productivity results. The initial store is limited to 1,000 items and performs a scan. Advanced temporal reasoning, automatic knowledge extraction, large-scale indexing, multi-tenant authorization, encryption and prompt-injection prevention are outside this package.

## What was actually compared

The public-data experiment executes **Faiss IndexFlatIP and IndexBinaryFlat**, a disclosed BM25 reference, classical MMR, a raw diallel cancellation control, coverage/diversity selection and trained linear rerankers. All use the same SciFact corpus and official test queries. Neural methods share a pinned MiniLM encoder and candidate-pool settings recorded in the manifest. BM25 is our transparent reference implementation, not a timing claim about Elasticsearch or another optimized engine.

The synthetic experiment compares exact deduplication, lexical and binary ranking, version checks, coverage/diversity, learned relevance and hard source requirements. It isolates fixture evidence coverage rather than generated-answer quality. The required-source control is given explicit relevant source IDs; it is an upper-bound integration control, not a fair learned-retrieval competitor.

We have not run LLMLingua, Mem0 or Graphiti end to end in this release. Their table entries are sourced feature comparisons. No universal superiority, published-leaderboard equivalence or production cost saving is claimed.

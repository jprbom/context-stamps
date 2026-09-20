# Usage guidelines

This page covers the existing SQLite evidence API. For the recommended v0.3.3 routing/session API and partial-facet queries, start with [selective context](selective-context.md) and the [README](../README.md).

## Choose an encoder and a source boundary

Use the built-in `HashingEncoder` for an offline lexical baseline. It preserves case, punctuation and adjacent token features, but does not understand paraphrases. Use `SentenceTransformerEncoder` or implement `identity`, `dim` and `encode(text)` for semantic retrieval.

An encoder identity must distinguish weights, revision, pooling, normalization and preprocessing. The neural adapter pins a model commit and records its sequence limit. Custom adapters must distinguish these details themselves. Quantized exports may produce different codes and should use a new identity unless compatibility is established.

Ingest sections or functions as individual sources, such as `manual.md#filter` or `ledger.py#transfer`. A source has one active revision; replacement is not a history archive. Keep your version history in Git or your own source system. The library does not scan the filesystem or split documents automatically.

## Source freshness

`digest` is the SHA-256 of exact UTF-8 text. There is no whitespace normalization. This matters for indentation-sensitive code and structured data.

`dependencies` maps stable identifiers to application-defined version strings. For another stored source, use its content digest. For an external environment, a lockfile digest or explicit configuration version is appropriate.

Updating source bytes or declared dependencies invalidates that source's declared transitive dependents, then installs the new source revision. `invalidate(source)` propagates through the same explicit links. Re-adding a source is an explicit re-observation and clears its stale flag. The caller must only do this after actually rechecking its validity.

When `revisions` is supplied to `recall` or `pack`, it must contain the current digest of each candidate source and every dependency of that candidate. A missing entry produces `unknown_version`; a mismatch produces `stale_version`. Both are excluded. An explicitly invalidated record remains excluded even if its bytes match.

When `revisions=None`, uninvalidated records are eligible but labeled `unchecked`. This is convenient for static reference collections; it is not proof that a file on disk is current.

## Packing policy

The first release scans all records and sorts by binary bit agreement, with source name as a deterministic tie-break. A family mismatch is an error. Scores are neither probabilities nor calibrated cosine similarities.

The greedy packer takes whole chunks that fit. It removes exact duplicates only when both the content digest and dependency map match. It preserves a decision entry for omitted aliases; `get` can recover the original. It does not use approximate similarity to drop changed content.

Without `token_counter`, a packet is bounded by UTF-8 bytes. Use the target tokenizer for a real token guarantee. The full packet, including headers and separators, is counted at each selection. System instructions, the question, protocol overhead and generated output are outside this packet and require separate space.

`min_score` is an optional retrieval threshold; there is no universal threshold that works across encoders and tasks. Validate on your corpus. Empty packets are legitimate results when all items are stale, below threshold or too large.

The reference implementation prioritizes simplicity. Projection costs scale with embedding dimension times bit width; retrieval scans the collection; packing repeatedly counts the candidate packet. Measure before using large collections.

## Operational boundaries

Use one store per trust boundary. This is a local library with no authentication, encryption, network service, tenant ACLs or background capture. Its MCP adapter uses stdio. If embedding or generation uses a network adapter, the caller controls what data is sent.

Use one `ContextMemory` instance per owning thread and close it when finished. SQLite handles persistence, but this package is not a high-concurrency server. Source text is stored in cleartext. `forget` removes a record and invalidates dependents; it does not guarantee secure erasure from SQLite pages, backups or snapshots.

Retrieve authoritative evidence before consequential actions. Memory content is data, not permission or trusted instructions. Stamps cannot certify that information is true.

## Migration

The `cs1` serialization includes the full SHA-256 family identity, width and hexadecimal code. Its string representation is substantially larger than the raw code. Never shorten the family ID when deciding compatibility.

Persist trained families with `Family.save`. A stored SQLite family is checked at reopen. To change encoder, seed, bit width, learned mean or projection, create a new store and re-ingest originals. This avoids mixing incompatible indexes. Future format migrations must be explicit and retain golden-vector tests.

# Security policy and threat model

## Secret scanning

CI runs Gitleaks 8.30.1 from a version- and SHA256-pinned archive with all default rules enabled. `.gitleaks.toml` excludes seven exact adjacent public ArguAna identifier pairs which caused nine `generic-api-key` findings, and only in `evidence/replication-v1/per-query.jsonl`. The rule, path and entire matched pair must agree. No directory, line or whole rule is excluded. The underlying evidence stays unchanged.

`experiments/verify_secret_scan.py` checks the original finding, its scoped exclusion, a generated fake credential on the same line, the public pair at another path and a changed neighboring value. Positive controls must remain detectable. See [Gitleaks configuration semantics](https://github.com/gitleaks/gitleaks#configuration). For local checks, run `gitleaks dir . --config .gitleaks.toml --redact` and `python experiments/verify_secret_scan.py --gitleaks /path/to/gitleaks`. Findings outside these exact exclusions require review. Clean scanning is not a guarantee of absence of secrets or application vulnerabilities.

## Progressive routing and receipts

The host authenticates callers and supplies current eligible IDs, roles and revisions. `RoutingPolicy` is a local configuration, not a credential. Scope must identify the actual domain, encoder and schema; undeclared distribution changes are not detected. Returned IDs never authorize tool execution.

Compact exits also require a nonzero calibration count and a certified upper error bound at or below the caller's risk budget. These are statistical controls under the recorded validation assumptions, not adversarial guarantees. `FacetCompiler` treats source text as data and never executes it. Extracted identifiers and relations are routing hints; only host-asserted metadata may set authority, policy or modality, and none of these fields grants access.

`ScopeCertificate` binds a precise hybrid profile to a named validation scope and a digest of paired metric values. A certificate is not authorization and must not be copied across corpus, encoder, schema, eligibility, metric or retriever revisions. Issue it from a disjoint validation partition; using final evaluation labels creates leakage. `HybridScoreProfile` bounds score arrays to one million finite values, but a network service should apply smaller request and corpus limits appropriate to its resource budget.

`RelationMap` accepts declared edges only. Node and relationship labels are data, not executable instructions. The reference limits graphs to 1,000 nodes, 4,096 edges, 32 diffusion hops and 4,096 output dimensions. Its revision digest detects equality under the canonical representation; it does not authenticate the graph. Validate provenance and authorization before adding edges, and treat deliberate hash collisions as a possible ranking attack rather than an access-control bypass.

`ContextSession` is process-local and serializes graph mutations/cache access. Mutations invalidate every cached packet whose dependency closure contains the changed node or relationship source, including permission and same-version content changes. Unrelated receipts remain valid. A reverse source-to-receipt index is cleaned on expiry and eviction. Random 32-byte receipts are separate from spherical stamps and require the original live session and trusted host authorization. Do not expose resolution as an unauthenticated endpoint. TTL, eviction and invalid receipt failures disclose no source details. Source text remains untrusted; retrieval does not prevent prompt injection. Revocation cannot recall delivered plaintext. Remote use needs authentication, transport security and transaction semantics beyond this library.

The residual index verifies fetched vectors against its snapshot digests and rejects stale or reordered data. The eligibility list and snapshot require host authentication. The 32-byte code has no source/version/authentication envelope: enforce these bindings outside the code under a trusted shared schema. `render_integer_assignments` emits bounded declarative constants and never executes generated code.

Context Stamps is a local, single-user context store. It is not a sandbox, access-control system, encrypted vault, or public network service. No software can promise immunity from exploitation. The public v0.5.0 release is an early implementation; independent security review has not been performed.

## Protections

- MCP uses stdio and opens no network listener. Only recall, pack, inspect_source, select and explain are exposed by default. `cstamps --db context.sqlite serve --allow-writes` explicitly adds remember, invalidate and forget. Grant this only to a trusted agent with appropriate host permissions.
- SQL values are bound parameters. Stores reject triggers, views and virtual tables on opening; SQLite trusted schemas and memory mapping are disabled, cell checking and secure deletion enabled. No SQL or shell execution tool is exposed.
- Text is limited to 64 KiB per record, queries to 16 KiB, source IDs to 512 UTF-8 bytes, dependencies to 128, revision maps to 1,000 and stores to 1,000 records. Recall returns at most 100 records. Packing budgets cannot exceed 1,048,576 units. These are small-store limits, not a denial-of-service guarantee.
- New database files use owner-only permissions on POSIX. Existing POSIX stores with group/other permissions are rejected. Windows files inherit the directory ACL: use a directory accessible only to your Windows account. Symbolic-link database paths are rejected. Parent directories must be trusted; concurrent hostile filesystem mutation is outside this boundary.
- CLI text reads are bounded. Projection dimensions and training matrix sizes are capped. NumPy loading disables pickle. The optional neural adapter requires a pinned model revision and disables remote Python code; neural weights must use safetensors. A pinned revision establishes reproducibility, not model trust: use reviewed models and maintained inference dependencies.

## Integrate safely

Treat every retrieved chunk as untrusted evidence, even when its digest and revision match. Hashes establish equality, not authenticity. Prompt injection is not removed by fingerprinting, JSON headers, a token budget, or freshness checks. Keep system instructions separate; require independent policy checks for tool calls, file writes, credentials, network requests and destructive actions. Never grant retrieved text authority to request those actions.

Use one store per trust boundary. Any process with file access can read or modify the database. MCP read tools disclose the selected store to the connected client; read-only tools do not enforce per-record authorization. The process can access files allowed by its OS identity. Run it with minimal permissions, keep credentials out of text, and protect backups. Original text is stored without encryption; fingerprints can reveal similarity and are not anonymization. Secure deletion does not guarantee forensic erasure on SSDs, backups or snapshots.

Do not open databases, projection files or checkpoints received from untrusted parties. Size checks reduce accidental resource exhaustion; they do not isolate native parsers or contain a compromised dependency. Use maintained Python/SQLite and isolate model downloads and training. Neural inference can use substantial memory independently of store limits.

## Optional controller and computation results

Filter unauthorized and stale evidence before building neural tensors; a score or mask supplied by an untrusted client is not authorization. Recheck source access before returning text or cached results. Serving rejects non-finite inputs/outputs and unvalidated recurrent-depth overrides. Checkpoint loading uses safetensors, bounded file size and expected shapes/dtypes; it does not load pickle or execute downloaded Python. These checks do not sandbox native model dependencies.

`ComputationCache` is in-process and binds caller-supplied tenant/principal, model/prompt/tool/policy revisions, request digest and exact source revisions/content digests. The host must capture every relevant parameter and state, including seeds, conversation, time-dependent inputs and tool environments. A digest is not authentication, a signature or encryption. Do not cache side-effecting actions as if returning their old result executes them again. Revoked access requires a new policy binding and a host authorization check before lookup. Capacity, byte limits and TTL constrain the cache; it is not a distributed revocation service.

The experimental neural checkpoints are not enabled by default. The 32-byte similarity stamp is never used as an exact computation-cache key. `exact_decimal` performs only allowlisted arithmetic on bounded decimal strings and rejects operations requiring rounding; it never evaluates source code.

The v2 contractive controller supports trained serving depths 2/4/8 and bounds its score correction to ±1. Its mathematical convergence bound is not an authorization or adversarial-robustness guarantee. Query/document features must be built from the host's already authorized evidence pool. Optional dynamic-int8 evaluation uses local CPU operators and does not publish executable/pickle model files. Public-data preparation uses revision-pinned local model loading and whitelisted archive destinations; cached arrays disable pickle. Run these research scripts with ordinary user privileges and reviewed data/model sources.

## Reporting

Report vulnerabilities privately to the repository owner, Prashant Jagtap, through an existing private contact channel or GitHub private vulnerability reporting when available. Do not post credentials, private text or working exploit details in a public issue. Include the affected commit, minimal reproduction, impact and environment. No response-time guarantee is currently offered.

## References

- [SQLite security guidance](https://www.sqlite.org/security.html)
- [MCP security practices](https://modelcontextprotocol.io/docs/2025-11-25/tutorials/security/security_best_practices)

## Spherical candidate boundaries

Compact stamps expose approximate similarity and are not encryption or authentication. Only trusted hosts should provide role labels, source revisions, exact facet metadata and dependency edges. `ContextGraph.affected()` is an administrative API and does not filter node identifiers by role. `PackedStampIndex.search()` requires explicit eligible row indices; the host must derive them from current authorization and freshness rules. Do not expose an arbitrary client-supplied role or eligible-row list directly over a network. Exported model coefficients use JSON, not executable serialization.

Do not publish raw experiment manifests without reviewing local model-cache paths. No prompt-injection prevention, secret classifier or adversarial multi-tenant guarantee is claimed.
## Runtime and optional reranker boundaries

Use separate runtime/reranker instances for separate trusted isolation boundaries. Filter source authorization before passing text to the optional scorer. Its exact-content token cache is not an access-control or result-validity mechanism. `ContextRuntime` never executes retrieved instructions; its adapters and verifier are registered by the trusted host. Deadline checks cannot forcibly cancel a Python/network callback, so external adapters must enforce their own timeouts. Scope approval is host-owned evaluation policy, not authentication. See [runtime usage](docs/unified-runtime.md).

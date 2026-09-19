# Security policy and threat model

Context Stamps is a local, single-user context store. It is not a sandbox, access-control system, encrypted vault, or public network service. No software can promise immunity from exploitation. Version 0.1 is an early implementation; independent security review has not been performed.

## Protections

- MCP uses stdio and opens no network listener. Only recall, pack and inspect_source are exposed by default. `cstamps --db context.sqlite serve --allow-writes` explicitly adds remember, invalidate and forget. Grant this only to a trusted agent with appropriate host permissions.
- SQL values are bound parameters. Stores reject triggers, views and virtual tables on opening; SQLite trusted schemas and memory mapping are disabled, cell checking and secure deletion enabled. No SQL or shell execution tool is exposed.
- Text is limited to 64 KiB per record, queries to 16 KiB, source IDs to 512 UTF-8 bytes, dependencies to 128, revision maps to 1,000 and stores to 1,000 records. Recall returns at most 100 records. Packing budgets cannot exceed 1,048,576 units. These are small-store limits, not a denial-of-service guarantee.
- New database files use owner-only permissions on POSIX. Existing POSIX stores with group/other permissions are rejected. Windows files inherit the directory ACL: use a directory accessible only to your Windows account. Symbolic-link database paths are rejected. Parent directories must be trusted; concurrent hostile filesystem mutation is outside this boundary.
- CLI text reads are bounded. Projection dimensions and training matrix sizes are capped. NumPy loading disables pickle. The optional neural adapter requires a pinned model revision and disables remote Python code. A pinned revision establishes reproducibility, not model trust: use reviewed models and maintained inference dependencies.

## Integrate safely

Treat every retrieved chunk as untrusted evidence, even when its digest and revision match. Hashes establish equality, not authenticity. Prompt injection is not removed by fingerprinting, JSON headers, a token budget, or freshness checks. Keep system instructions separate; require independent policy checks for tool calls, file writes, credentials, network requests and destructive actions. Never grant retrieved text authority to request those actions.

Use one store per trust boundary. Any process with file access can read or modify the database. MCP read tools disclose the selected store to the connected client; read-only tools do not enforce per-record authorization. The process can access files allowed by its OS identity. Run it with minimal permissions, keep credentials out of text, and protect backups. Original text is stored without encryption; fingerprints can reveal similarity and are not anonymization. Secure deletion does not guarantee forensic erasure on SSDs, backups or snapshots.

Do not open databases, projection files or checkpoints received from untrusted parties. Size checks reduce accidental resource exhaustion; they do not isolate native parsers or contain a compromised dependency. Use maintained Python/SQLite and isolate model downloads and training. Neural inference can use substantial memory independently of store limits.

## Reporting

Report vulnerabilities privately to the repository owner, Prashant Jagtap, through an existing private contact channel or GitHub private vulnerability reporting when available. Do not post credentials, private text or working exploit details in a public issue. Include the affected commit, minimal reproduction, impact and environment. No response-time guarantee is currently offered.

## References

- [SQLite security guidance](https://www.sqlite.org/security.html)
- [MCP security practices](https://modelcontextprotocol.io/docs/2025-11-25/tutorials/security/security_best_practices)

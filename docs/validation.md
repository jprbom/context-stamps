# Validation record

Release 0.2.0, local Windows Python 3.13.13, 2026-09-19.

- 38 automated tests passed. They cover source updates/deletion, path traversal rejection, hard required-source guards, stale evidence, budget accounting, trained coefficient validation, MMR/diallel-related identities, transactional rollback and real MCP stdio sessions with default write-tool denial.
- All seven README Python examples executed. The file-observation example was supplied its documented fictional project file. The pinned neural encoder ran on CPU. The Ollama example was dry-run only; no live generator or edge hardware benchmark is claimed.
- Ruff and Bandit passed. pip-audit found no known vulnerabilities in the isolated environment, including installed neural dependencies. The editable package itself is skipped by advisory lookup. Audit coverage is not a security guarantee.
- Gitleaks 8.30.1 was obtained from its official release and checked against its published SHA-256 checksum. Full existing Git history and the working directory scans reported no secrets. The final committed history is scanned again before publication.
- Seven small selectors were trained: three synthetic relevance seeds, one synthetic restoration-label model, and three SciFact relevance seeds. JSON weights, losses, splits, fixtures/interventions, per-case or per-query results, hashes and limitations are included. Negative outcomes are retained.
- SciFact evaluated all 300 official test queries over 5,183 documents. A shared-library replication reproduced all aggregate nDCG values exactly. The initial and final summaries are retained. These are retrieval and evidence-coverage experiments, not generated-answer evaluation.
- The release wheel and source archive are built from the release commit. SHA-256 checksums and a build manifest accompany the GitHub release. Checksums detect changes; they are not an independent code-signing attestation.
- Workflow diagram validation from 0.1 remains applicable; the new results plot was generated from measured JSON and visually checked.

Run `python experiments/verify_evidence.py` to check experiment source hashes, fixture bytes, split separation and metric aggregation. CI repeats this check on Windows/Linux and Python 3.10/3.13. Source changes require rerunning the affected experiments rather than silently relabeling old evidence.

No independent penetration test, generative-model fine-tune, production-readiness certification, mobile/ARM latency or energy measurement is claimed. [Security boundaries](../SECURITY.md) and [experimental limits](experiments.md) remain part of the release.

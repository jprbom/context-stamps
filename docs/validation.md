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

## Follow-up validation, 2026-09-19

- 42 automated tests passed locally, including four new tests for the restricted model-output grader, boundary correctness, invalid JSON and stale/current fixture preparation. Ruff and the core Bandit scan passed. Independent checks confirmed the new precomputed BM25 implementation matches the existing reference and that graded nDCG handles a known two-document example correctly.
- All 323 NFCorpus and 1,406 ArguAna test queries were evaluated with frozen settings. Both selector regressions are published. Five absent ArguAna positive documents are retained as unretrievable labels in overall metrics and excluded from overlap classification; the evaluator correction is documented.
- 144 calls to a public 1.5B local model completed. All fictional prompts, responses, deterministic grades and actual token counts are retained. No generated code was executed; the grader accepts only a small AST grammar. These are constrained fixtures, not full repository tasks.
- The public v0.2.0 wheel was downloaded, checked against its published SHA-256 hash and installed without dependencies in a new environment. CLI help and the changing-source example passed, with import from the new environment confirmed. Setup including environment creation and download took about 9.85 seconds on this host. This is a [maintainer smoke test](../evidence/usability-smoke-v1/manifest.json), not a human usability study.
- Gitleaks 8.30.1's working-directory scan flagged nine `generic-api-key` matches in ArguAna ranking records. Each matched span consists of public dataset IDs; the IDs were verified against the separately downloaded public corpus. These are reviewed false positives. No broad file exclusion or scanner-rule suppression was added. This review is not a penetration test.
- The cross-dataset figure was generated directly from committed metrics and visually checked. The developer kit contains an invitation, three tasks, neutral facilitator instructions, privacy/consent guidance and a feedback template. No one has been contacted and no participant response is claimed.

`python experiments/verify_followup.py` verifies artifact and source hashes, frozen weights/protocol, query counts, self-match exclusion, recorded bootstrap intervals, all generated-output grades and aggregate metrics. CI runs it alongside the original evidence checks. Raw public corpora and downloaded model weights remain outside the repository. The follow-up changes documentation and experiment tooling; it does not retune or change the released selector.

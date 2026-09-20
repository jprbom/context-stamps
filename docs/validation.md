# Validation record

## Private v0.4.0 follow-up

108 local unit tests pass. The new control record contains eight authored facet cases, 100 exact structured-profile round trips and five router paths. A compact policy whose certified upper error bound exceeds the caller's budget falls back to precise retrieval. These tests validate interface behavior and fail-closed routing; they do not demonstrate facet accuracy, retrieval improvement or a qualified compact exit on public data. [Design and limits](adaptive-capsules.md).

## Private v0.3.3 follow-up

102 local unit tests pass, including nine new partial-query/selective-invalidation tests and 4,000 comparisons with an uncached graph over 500 seeded mutation steps. `verify_mutation_reuse.py` checks 6,600 handoffs, packet equality, cache accounting and 152 receipt checks. The v0.3.2 global-cache source is preserved for comparison. The latest training remains the three-seed ITQ run; no additional model training is claimed here.

Secret scanning now uses exact public-ID/path exclusions, retains every default detection rule and passes five positive/negative controls. The former nine findings described in the historical entries below have been resolved. Current directory and history scans are required before sync. [Current behavior](selective-context.md).

## Private v0.3.2 follow-up

93 local unit tests pass. The new offline verifier checks 18,261 public ranking records (six methods and three-seed replication over 2,029 queries), disjoint policy calibration, 800 recorded context handoffs across two implementations and source hashes. Full quantizer retraining requires the external pinned embeddings. [Usage, metrics and limitations](progressive-routing.md).

## Private v0.3.1 follow-up

- 82 automated tests pass locally, including maximum-dimension near-tie retrieval, stale backing vectors, exact 32-byte round trips and bounded configuration rendering.
- `verify_improvements.py` checks new artifact/source hashes, all 2,029 exact public ranking matches per retrieval run, 21 allocation training runs, 3,600 fixed-bit rankings, 768 workflow stages and 540 comparative scale queries.
- Four workflow iterations retain output-contract failures and latency outliers. The final run uses actual 32-byte routing, matched warmup settings and unchanged controls. Eight unique workflows are repeated twice; they are not 16 independent tasks.
- Ruff and core Bandit pass. Gitleaks reports exactly the same nine reviewed public-ID false positives, with no new matches. The new APIs introduce no new mandatory dependency.
- Full comparisons include stronger Faiss and exact-graph controls, which remain preferable for their measured workloads. See [retrieval repair and efficiency](retrieval-repair.md).

> Historical v0.2 evidence. For the current candidate, see [adaptive capsules](adaptive-capsules.md), [measured v0.3.3 results](selective-context.md), [scenario coverage](scenario-matrix.md) and [failures](failures-and-fixes.md).

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

## Private v0.3 candidate verification

- 72 unit tests pass locally, including 100 seeded random graph comparisons with an independent fixed-point oracle and a geometry control against known angular distances.
- The spherical verifier retrains six ridge models, repeats validation selection and replays 3,840 recorded query/seed/method rankings. The pairwise verifier retrains three selected models with all nine validation trials and replays another 3,600 rankings. Projection seeds are not independent task samples.
- The exact-facet guard passes 480 positive and 480 missing-answer regression cases. The unguarded failures remain recorded.
- Historical source/artifact checks continue to pass. No original public-data benchmark outcome was overwritten by a new synthetic result.
- All current README Python snippets, the spherical example and the `scqr encode` CLI passed local smoke checks. Wheel and source distribution builds succeeded.
- Ruff and core Bandit passed. pip-audit reported no known vulnerabilities in the local core/research environment, skipping the editable package. This does not audit every external model runtime or establish exploit resistance.
- Gitleaks reports the same nine previously reviewed public ArguAna ID false positives and no new matches. Local-cache paths were removed from the new Ollama manifests with the redaction recorded.
- The multimodal pilot reports each completed modality separately. Local generated media and external weights are excluded from the repository; hashes and model revisions support reproduction. Output parity is not perceptual-quality validation.
- The completed v2 multimodal run contains 36 generations and 18 identical direct/routed output pairs across image, speech and video. The verifier checks pair completeness, hashes, token equality and timing arithmetic.
- The new public spherical retrieval run covers 2,029 queries and 6,087 method/query records. Both compact methods underperform dense retrieval on every dataset. Its verifier checks provenance, aggregates and self-exclusion; full metric replay requires external corpora and caches.

Remote CI status must be checked on the pushed candidate commit. The repository remains private; no public-release readiness is implied by local checks.

A clean wheel installation with no optional dependencies loaded the bundled experimental scorer and ran `scqr encode` successfully. Historical workflow source is preserved under `evidence/source-snapshots` when a later capacity fix changes its file hash. Source manifests identify the original runtime path and its exact snapshot; current regression tests exercise the patched implementation.

## Private v0.5 candidate verification

- The hybrid-retrieval-v1 protocol was written after the first three datasets had been inspected but before SciDocs outcomes were computed. It pins the encoder revision, BM25 parameters, 0.75 validation-selected semantic weight, metric and bootstrap seed.
- The run contains 9,087 method/query records: dense MiniLM, BM25 and the frozen hybrid for 300 SciFact, 323 NFCorpus, 1,406 ArguAna and 1,000 SciDocs queries. `verify_hybrid_retrieval.py` checks checksums, source snapshots, record uniqueness, aggregate arithmetic and the expected sign of every paired interval.
- The hybrid improved nDCG@10 on the three previously inspected datasets. It regressed on prospective SciDocs by 0.0121, with a paired 95% interval from −0.0194 to −0.0046. The failure remains in the candidate and is the reason scope certification defaults to dense fallback.
- Relation-map controls check direction, revision sensitivity, normalization and bounds. Product-family controls fit eight independent 32-bit ITQ blocks, concatenate exactly 256 bits and encode through `SphericalStamp`.
- Certificate controls check both a positive-gain admission and a regression abstention. These are deterministic interface tests; they are not a prospective validation of the remediation.
- The animated and interactive workflow diagrams passed their structural and multi-viewport checks. The static cover and benchmark plot were visually inspected at full resolution.

Raw BEIR corpora and embedding caches remain outside the repository. Encoder inference, BM25 construction, network time and capsule candidate generation are excluded from the recorded per-query score timing. No base-model fine-tuning, internal attention modification, mobile-device result or production load claim is made.

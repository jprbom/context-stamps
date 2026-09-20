# Candidate changes

All versions below are research candidates. The repository is private; this file does not announce a public or PyPI release.

## 0.3.3

- Added explicit partial-facet queries and calibration scopes tied to active views, families and weights.
- Replaced global session invalidation with dependency-selective receipt invalidation.
- Added 4,000 mutation-oracle comparisons and a 6,600-handoff replay against exact and global-cache controls; identical evidence retained.
- Rewrote the README around current defaults, three-seed findings, usable examples and measured limits. Historical experiments remain available.
- Kept document stamps at 32 bytes. No new model training or internal attention change in this update.

## 0.3.2

- Added exact-first routing, optional validated compact exits and bounded context sessions.
- Recorded three-seed ITQ training: mean compact relevance improved, dense remained stronger, one ArguAna seed regressed slightly.
- Preserved dense ranking through fallback when compact calibration failed.
- Added narrowly scoped secret-scanner exceptions with five controls; retained all default rules.

## 0.3.1 and earlier

See the [historical retrieval repair](docs/retrieval-repair.md), [spherical results](docs/spherical-results.md) and [failure ledger](docs/failures-and-fixes.md). Earlier losses and unsuccessful runs have not been replaced by newer measurements.

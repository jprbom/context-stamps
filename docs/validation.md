# Validation record

Local validation on 2026-09-19, Windows, Python 3.13.13:

- 31 automated tests passed, including actual MCP stdio sessions, default write-tool denial, source freshness, exact-text retention, budget accounting, transaction rollback, projection parity and resource bounds.
- Ruff passed. Bandit reported no issues in `stamps.py` and `context_stamps/`.
- pip-audit reported no known vulnerabilities in the isolated environment after updating pip to 26.2.1. The editable project itself has no advisory identifier and is skipped by this dependency audit. The optional semantic encoder dependencies were installed in this environment. Advisory coverage is not proof of security.
- All five README Python examples executed successfully, including the pinned neural encoder. The SLM example passed its dry run; a live generation request and edge-device performance were not tested.
- Wheel and source distribution built successfully. A clean environment exercised the dependency-free wheel and command entry point before the security changes; CI builds the current source on Windows and Linux with Python 3.10 and 3.13.
- The synthetic benchmark was rerun after projection validation changes. Results include dataset and code hashes; these are not real-world semantic quality measurements.
- Workflow diagram passed all nine Archify validation stages with zero errors or warnings. Light/dark rendering and four desktop sizes were inspected. The static overview infographic was visually checked.

No independent penetration test, trained language-model release, production readiness claim or public package release is implied. See [SECURITY.md](../SECURITY.md) for deployment boundaries. GitHub Actions provides validation tied to each pushed commit.

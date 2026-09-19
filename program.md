# Context Stamps experiment protocol

Author: Prashant Jagtap

Improve retrieval or context selection under a fixed resource budget. Start with `README.md`, `docs/research.md` and a clean checkout. Run the existing tests and benchmark before changing an algorithm.

For a projection experiment, edit `context_stamps/learning.py`. Keep `benchmarks/retrieval.py`, the seed, held-out inputs and metric definitions fixed. For an application study, define and freeze independent acceptance tests before changing the context policy.

Before running, record the allowed wall time, number of trials and hardware. Stop when the agreed budget is reached; do not start an unbounded training loop. This file grants no permission to publish, upload data, alter repository visibility or purchase compute.

For each trial record the code revision, dataset digest, family, seed, environment, quality metrics and total latency. Compare against exact deduplication and the unchanged baseline. Keep failures and null results. A synthetic recall gain is not proof of lower application cost or improved task completion.

Accept a change only when it passes the correctness suite and improves the prespecified objective without exceeding its quality-loss allowance. Never optimize by editing evaluation data or hiding failed runs. Use a separate untouched test set after model/policy selection.

Do not train a generative model merely to add a model artifact. First show a failure that deterministic policy or an existing encoder cannot adequately resolve. Document encoder/model licenses, input limits and the complete training recipe for any eventual weights release.

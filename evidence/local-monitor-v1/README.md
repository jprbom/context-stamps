# Local adaptation monitor: engineering evidence

By Prashant Jagtap

This is an explicitly simulated local lifecycle, with **zero model calls and zero GPU training runs**. It does not add a model-quality score or an edge-device performance result.

- A two-cell CPU policy is fitted on 60 fixture tasks / 120 paired action observations.
- The candidate passes the fixture's separate adaptation and retention gates on 800 simulated task IDs each.
- The live fixture has 20 successful tasks followed by seven failed tasks. The seventh failure crosses the retention failure threshold, producing a durable alarm after 27 monitored outcomes.
- Restarting the monitor retains the alarm; enforcing it rolls the registry back to its anchor. Every fixture training record, evaluation record and live outcome is retained in `fixture.json.gz`.

The live mixture averages four alternative Bernoulli rates and all possible start positions. It uses constant arithmetic state per metric/cohort. This is an established statistical construction; see the [derivation, assumptions and integration limits](../../docs/local-regression-monitor.md).

Validation includes 14 new core tests, a rational-arithmetic oracle over every binary sequence of length eight, exhaustive finite-horizon optional-stopping paths under a conditional null, and late-deterioration, retention, deadline, missing-data, restart, competing-writer and stale-rollback tests. The full local core suite passes 402 tests without skips. Two replay checks ensure float tolerance cannot hide changed decisions, types, structure or materially changed numerical results.

The initial export compared Python tuples directly with their JSON list form and failed replay. Its original sources and records remain in [the preparation evidence](../local-monitor-preparation-v1/README.md). The next replay passed Windows but failed exact floating comparison on Linux: 16 intermediate log statistics differed at the last few binary digits. The original measured records are unchanged. The replay now compares exact structure, types and decisions, with finite-float relative/absolute tolerance `1e-12`. Both original verifier versions were archived before correction. `engineering.json` records the final replay-tool fingerprints and validation.

```bash
python experiments/verify_local_monitor.py
python experiments/test_local_monitor_evidence.py -v
python -m unittest discover -s tests -p test_local_monitor.py -v
```

This reproduction needs only the installed core and Python standard library. It does not call a local or remote model. Simulation labels do not establish valid IID deployment samples, truthful verification, sustained model improvement, measured power savings or production reliability. Existing failed model candidates remain inactive.

A later clean-shell check found that the new CLI scripts relied on a development `PYTHONPATH`. The scripts now explicitly locate this reviewed repository, matching existing experiment entry points. Windows and Linux replay pass with `PYTHONPATH` absent, and the regenerated fixture bytes are identical. `cli-validation.json` records this follow-up; the previous tool source remains archived.

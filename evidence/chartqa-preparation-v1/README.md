# ChartQA local visual-memory preparation

By Prashant Jagtap.

This record contains **no actual ChartQA model calls, trained policy, evaluation
score or model improvement**. It prepares a local comparison of direct image
reading, reusable model-extracted chart memory and bounded arithmetic. The
[protocol and local commands](../../docs/chartqa-local-learning-draft.md) describe
the remaining authored canary, training, validation and evaluation steps.

- Publisher revision: `vis-nlp/ChartQA@044eabfc306abfe9340c5741f0093aefc5973d06`.
- Training: 64 images / 143 questions; validation: 32 / 78; test: 64 / 145.
- Two to four distinct questions per image; selection never uses their answers.
- Zero identical decoded-pixel groups across the selected cohorts; near-duplicates
  and base-model pretraining exposure remain possible.
- The 160 images occupy 7,018,856 bytes. All 172 prepared files were checked against
  the retained manifest. Reader inputs have no answer fields; keys remain separate.
- Thirty experiment tests pass. Canary-lifecycle inference is mocked; a printed
  canary success inside those unit tests is **not an actual model result**.
- Twenty-one authored scorer cases match the reviewed pinned Pix2Struct function,
  including its percentage and zero-target behavior. Exact string agreement is a
  separate metric. Full upstream source and license are retained locally.

The record preserves the failed Parquet preparation source (blocked native reader),
the first native preparation source (overstrict filename validation), and the
successful JSON/PNG preparation source. Windows Application Control's native-module
block was not bypassed. Dataset, raw questions, chart annotations and answer keys
are not included. The publisher's GPL-3.0 notice and original chart-source terms
remain applicable to separately downloaded material; this MIT code release does
not relicense those data.

`prepared-manifest.json` and `registration.json` identify the external preparation.
The published inventory uses filename/digest records and also binds the original
manifest's exact SHA-256; it contains no reference answers.
`source-snapshots.json.gz` retains exact historical code bytes, including original
line endings. `manifest.json` hashes the published records and separately checks
the current experiment sources after line-ending normalization for portability.
`scorer-canaries.json` records the reference source revision, URLs, hashes and
authored comparisons. `engineering.json` records the local commands and outcomes.

```powershell
python -m pip install -e . Pillow==12.3.0
python -m unittest discover -s experiments -p 'test_chartqa_*.py' -v
python experiments/verify_chartqa_preparation.py
```

The offline verifier checks the published fingerprints and replays the authored
scorer cases. It does not independently regenerate image cohorts or validate
model answers. Use the separately documented downloader and native preparation
commands to reproduce the data preparation from the pinned public sources.

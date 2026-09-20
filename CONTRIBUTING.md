# Contributing

Discuss substantial interface or storage-format changes in an issue first. Keep the standard-library core usable without optional model or training packages.

```bash
python -m pip install -e ".[dev,mcp,tokens]"
python -m unittest discover -s tests -v
python -m ruff check .
python -m build
```

For retrieval or packing changes, provide the failing case and a before/after comparison on frozen evaluation inputs. Include latency and correctness, not only token reduction. New projection formats require a new version or method identity, explicit compatibility behavior and serialization tests.

Do not commit source corpora, private traces, base-model weights, credentials, database files or generated caches. Small reviewable experimental quantizers need provenance, source hashes and appropriate data-derived license notices, as in `evidence/quantizer-seeds-v1`. Add small redistributable fixtures instead. Keep optional adapters explicit about first-use downloads and model licenses.

Preserve the Prashant Jagtap copyright notice and the applicable third-party notices. Contributions are submitted under the repository's MIT License. Do not add unearned author or co-author credits.

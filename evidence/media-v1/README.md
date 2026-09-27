# Canonical media engineering checks

By Prashant Jagtap. MIT License.

The source adapter represents external image, audio and video bytes, extracted
views, producer revisions and source-relative regions in the canonical context
graph. It adds no learned extractor or model benchmark score.

`example.json` retains the complete output of `examples/media_context.py`: an
authored 250 ms WAV signal (8,044 bytes), two overlapping derived views, their
temporal-order node, exact-byte resolution and source revocation. There are zero
model calls. The example validates the WAV frame count and sample rate with the
standard-library decoder. It does not measure perceptual accuracy.

Seventeen targeted tests cover altered bytes, hidden sources, model-output
typing, extraction identity, invalid geometry/time, unknown metadata, expiry,
supersession, callback errors and revocation during a fetch. Integration tests
also cover reopening the durable store, revocation from an independent writer
and a working-set quota too small for a view's source dependency. The first
quota test expected `ValueError`; the store correctly raised `OverflowError`.
Only the test expectation changed.

The first evidence-export helper resolved an older installed package when launched
outside the checkout and stopped before writing records. Explicitly selecting the
reviewed source checkout fixed that helper; the example and library were unchanged.

`core-tests.txt` records the command and completion summary of the full local suite. `manifest.json` binds the retained
outputs and measured source files. Replay on a reviewed clone:

```bash
python experiments/verify_media.py
python -m unittest discover -s tests -p test_media_evidence.py -v
```

No callback sandbox, offline-provider guarantee, VLM accuracy, image/audio/video
token reduction, throughput or autonomous weight improvement is claimed. A host
must validate extracted content independently before using it as a training label.

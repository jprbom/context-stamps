# Retained relation preparation, version 1

This preparation was superseded before any model requests. Ordered relation
packets lacked the local table/section ancestor needed to distinguish containers
on the same page. The source bytes were frozen before changing the extractor.

The initial index contained 192,674 packets, representing 1,001,089 occurrences.
Two episodes exceeded the occurrence bound and were excluded from the relation
index; their evaluation questions were not dropped. No answer-quality scores
exist for this preparation.

Registration, preparation hashes, index counts and the supersession note retain
their original bytes. The [manifest](manifest.json) names the immutable source
archive. Raw histories, prepared prompts and the index remain local. Replay:

```powershell
python experiments/verify_relation_preparation.py
```

The corrected [version-2 experiment](../lme-relations-v2/README.md) includes local
ancestor identity and retained all 200 episodes. It still regressed in answer
quality and request latency; fixing source ownership did not establish utility.

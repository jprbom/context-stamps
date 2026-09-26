# Historical engineering sources

These deterministic gzip JSON archives retain the exact source bytes named by
earlier engineering manifests. The capture command checks every source digest
before archiving it. Content is indexed by SHA-256; the index also pins the
compressed archive digest and source/base commit. Captures from a changed
worktree explicitly record that fact; per-file hashes identify the actual code.

Evidence verifiers use current source when its hash matches, otherwise an archived
copy. They bound decompression and verify the archived bytes without extracting
or executing code. This preserves old measurements when implementation changes.
It does **not** apply historical test results to the current implementation.
Current tests and CI must pass separately. Archives contain repository source,
not user evidence, model prompts, credentials or training corpora.

Capture: `python experiments/archive_engineering_sources.py --name NAME --manifests MANIFEST ...`

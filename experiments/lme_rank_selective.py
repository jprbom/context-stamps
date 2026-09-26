"""Retain the original query plan for sparse path views; remeasure all lookups.

The first optimization's tail regression remains in search/. This fixed choice
uses the index channel, never a question answer or model outcome. It is a local
query-plan refinement, not a general learned optimizer or quality improvement.
"""

import argparse
from pathlib import Path

import lme_rank_fast
from lme_memory import ranked
from lme_rank_fast import ranked_fast as original_fast
from ruler_native import outside_repo, sha, write_new


def ranked_selective(db, question, domain, channel, allowed):
    search = ranked if channel == "path" else original_fast
    return search(db, question, domain, channel, allowed)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("data", "run", "output"):
        parser.add_argument("--"+name, required=True, type=Path)
    args = parser.parse_args()
    output = outside_repo(args.output)
    # Saved before parent profile creates its output or executes any lookup.
    write_new(output.with_suffix(".selection.json"), dict(
        strategy="original for path; tie-preserving native-rank cursor for state/change",
        source_sha256=sha(__file__), parent_source_sha256=sha(Path(lme_rank_fast.__file__)),
        reason="The first profile regressed on sparse path searches; no answer-quality labels used",
        qualification="Same-query engineering remeasurement, not independent generalization"))
    lme_rank_fast.ranked_fast = ranked_selective
    lme_rank_fast.profile(args.data, args.run, output)

"""Complete local-tokenizer example; no model requests or downloads.

Copyright (c) 2026 Prashant Jagtap. MIT License.
"""

import argparse
import json
from pathlib import Path

from context_stamps.greedy_budget import pack_in_order


def main(path):
    from tokenizers import Tokenizer

    tokenizer = Tokenizer.from_file(str(path))
    tokenizer.no_truncation()
    tokenizer.no_padding()
    fragments = ["Fictional maintenance record: pump P7 was inspected on Monday.",
                 "Irrelevant example detail. "*1000,
                 "Fictional maintenance record: P7 requires a seal inspection before restart.",
                 "Fictional scheduling record: the inspection is assigned to team A."]

    def render(indices):
        return ("<|im_start|>user\nSummarize these fictional records; do not infer missing facts.\n"
                + json.dumps([fragments[i] for i in indices])
                + "<|im_end|>\n<|im_start|>assistant\n")

    result = pack_in_order(len(fragments), render,
        lambda texts: [len(e.ids) for e in tokenizer.encode_batch_fast(texts, add_special_tokens=False)],
        lambda text: len(tokenizer.encode(text, add_special_tokens=False).ids),
        budget=256, max_selected=4, batch_size=8)
    assert result.indices == (0, 2, 3)
    print(json.dumps(dict(selected=result.indices, tokens=result.tokens,
                          counted_candidates=result.counted_candidates, count_batches=result.batches)))
    print("No model was invoked. The application must authorize and verify real source records.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokenizer", required=True, type=Path)
    main(parser.parse_args().tokenizer)

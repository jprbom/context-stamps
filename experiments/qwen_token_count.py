"""Read one JSON prompt from stdin and return pinned-tokenizer metadata/count.

Called in the existing permitted tokenizer environment; no tensor/GPU imports.
"""

import argparse
import hashlib
import importlib.metadata
import json
import platform
import sys
from pathlib import Path


def main():
    from tokenizers import Tokenizer

    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer", type=Path, required=True)
    args = parser.parse_args()
    raw = sys.stdin.buffer.read(1024 * 1024 + 1)
    if len(raw) > 1024 * 1024:
        raise ValueError("Prompt input exceeds limit")
    prompt = json.loads(raw)
    if not isinstance(prompt, str):
        raise ValueError("One prompt string required")
    tokenizer = Tokenizer.from_file(str(args.tokenizer))
    tokenizer.no_truncation()
    tokenizer.no_padding()
    print(json.dumps({"tokens": len(tokenizer.encode(prompt, add_special_tokens=False).ids),
                      "tokenizer_sha256": hashlib.sha256(args.tokenizer.read_bytes()).hexdigest(),
                      "python_version": platform.python_version(),
                      "tokenizers_version": importlib.metadata.version("tokenizers")}))


if __name__ == "__main__":
    main()

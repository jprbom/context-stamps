"""Command-line interface; output is JSON for use by people and agent tools."""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys

from stamps import Family, HashingEncoder, demo

from .memory import ContextMemory
from .security import read_text


def _mapping(path: str | None) -> dict | None:
    if path is None:
        return None
    value = json.loads(read_text(path, 1048576))
    if not isinstance(value, dict) or any(
        not isinstance(k, str) or not isinstance(v, str) for k, v in value.items()
    ):
        raise ValueError("version file must be a JSON object mapping strings to strings")
    return value


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Local context fingerprints and evidence selection")
    parser.add_argument("--db", default="context.sqlite", help="SQLite store (default: context.sqlite)")
    parser.add_argument(
        "--family", help="projection family JSON; required only when creating a trained store"
    )
    parser.add_argument("--model", help="optional sentence-transformers model identifier")
    parser.add_argument("--revision", help="immutable model commit SHA")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("demo", help="run without a database or model download")
    add = sub.add_parser("add", help="store one text chunk; use stable source IDs")
    add.add_argument("--source", required=True)
    inputs = add.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--text")
    inputs.add_argument("--file")
    add.add_argument("--dependencies", help="JSON version map")
    for name in ("recall", "pack"):
        command = sub.add_parser(name)
        command.add_argument("query")
        command.add_argument("--revisions", help="complete current source/dependency version map")
        if name == "recall":
            command.add_argument("--limit", type=int, default=5)
        else:
            command.add_argument("--budget", type=int, default=2048)
            command.add_argument("--tokenizer", help="optional tiktoken encoding name, e.g. cl100k_base")
            command.add_argument("--min-score", type=float, default=0.0)
    for name in ("get", "invalidate", "forget"):
        sub.add_parser(name).add_argument("source")
    observe = sub.add_parser("observe", help="observe only explicitly listed files under a trusted root")
    observe.add_argument("--root", required=True)
    observe.add_argument("paths", nargs="+")
    explain = sub.add_parser("explain", help="explain changed or missing source versions")
    explain.add_argument("source")
    explain.add_argument("--revisions", required=True)
    select = sub.add_parser("select", help="select evidence with freshness and required-source guards")
    select.add_argument("query")
    select.add_argument("--budget", type=int, default=2048)
    select.add_argument("--revisions")
    select.add_argument("--required", action="append", default=[])
    select.add_argument("--selector", help="optional experimental JSON reranker")
    fit = sub.add_parser("fit", help="fit a projection on a training-only NumPy embedding matrix")
    fit.add_argument("vectors")
    fit.add_argument("--encoder-id", required=True)
    fit.add_argument("--out", required=True)
    fit.add_argument("--bits", type=int, default=128)
    fit.add_argument("--method", choices=["centered", "itq"], default="itq")
    server = sub.add_parser("serve", help="serve local read tools over MCP stdio")
    server.add_argument(
        "--allow-writes", action="store_true", help="explicitly enable remember, invalidate and forget"
    )
    args = parser.parse_args(argv)
    try:
        if args.command == "demo":
            result = demo()
        elif args.command == "fit":
            import numpy as np

            from .learning import fit_family

            family = fit_family(
                np.load(args.vectors, allow_pickle=False, mmap_mode="r"),
                encoder=args.encoder_id,
                bits=args.bits,
                method=args.method,
            )
            family.save(args.out)
            result = {"family_id": family.identity, "path": args.out, "method": family.method}
        else:
            if args.model:
                from .encoders import SentenceTransformerEncoder

                encoder = SentenceTransformerEncoder(args.model, revision=args.revision or "")
            else:
                encoder = HashingEncoder()
            family = Family.load(args.family) if args.family else None
            with ContextMemory(args.db, encoder=encoder, family=family) as memory:
                if args.command == "serve":
                    from .server import serve

                    serve(memory, allow_writes=args.allow_writes)
                    return 0
                if args.command == "observe":
                    from .sources import observe_files

                    result = observe_files(memory, args.root, args.paths)
                elif args.command == "explain":
                    from .sources import explain_versions

                    result = explain_versions(memory, args.source, _mapping(args.revisions))
                elif args.command == "select":
                    from .selection import LinearSelector, select_evidence

                    selector = LinearSelector.load(args.selector) if args.selector else None
                    result = select_evidence(
                        memory,
                        args.query,
                        budget=args.budget,
                        revisions=_mapping(args.revisions),
                        required=args.required,
                        reranker=selector.score if selector else None,
                    ).to_dict()
                elif args.command == "add":
                    text = read_text(args.file) if args.file else args.text
                    result = memory.add(text, source=args.source, dependencies=_mapping(args.dependencies))
                elif args.command == "get":
                    result = memory.get(args.source)
                elif args.command == "invalidate":
                    result = {"invalidated": memory.invalidate(args.source)}
                elif args.command == "forget":
                    result = {"deleted": memory.forget(args.source)}
                elif args.command == "recall":
                    result = memory.recall(args.query, limit=args.limit, revisions=_mapping(args.revisions))
                else:
                    counter = None
                    if args.tokenizer:
                        import tiktoken

                        encoding = tiktoken.get_encoding(args.tokenizer)

                        def counter(s):
                            return len(encoding.encode(s, disallowed_special=()))

                    result = memory.pack(
                        args.query,
                        token_budget=args.budget,
                        token_counter=counter,
                        revisions=_mapping(args.revisions),
                        min_score=args.min_score,
                    ).to_dict()
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    except (ValueError, OSError, ImportError, sqlite3.Error, TypeError) as exc:
        print(json.dumps({"error": str(exc)}), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

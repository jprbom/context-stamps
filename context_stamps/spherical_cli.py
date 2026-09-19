"""Small offline CLI for spherical payloads; no model or network required."""

import argparse
import json

from stamps import Family, HashingEncoder

from .activation import StampSchema
from .security import bounded_text, read_text
from .spherical import SphericalStamp


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    encode = commands.add_parser("encode", help="encode an explicit JSON object of text facets")
    encode.add_argument("facets_file")
    inspect = commands.add_parser("inspect", help="validate a portable JSON spherical stamp file")
    inspect.add_argument("payload_file")
    args = parser.parse_args()
    try:
        if args.command == "inspect":
            stamp = SphericalStamp.from_payload(read_text(args.payload_file, 8192))
            print(json.dumps({"bits": stamp.bits, "views": [n for n, _ in stamp.views]}))
        else:
            facets = json.loads(read_text(args.facets_file, 16384))
            if not isinstance(facets, dict) or not 1 <= len(facets) <= 8:
                raise ValueError("one to eight named text facets required")
            for text in facets.values():
                bounded_text(text, 2048)
            encoder = HashingEncoder(64)
            families = {name: Family(encoder.identity, 64, 64, 17 + i) for i, name in enumerate(sorted(facets))}
            stamp = SphericalStamp.encode({n: encoder.encode(t) for n, t in facets.items()}, families)
            schema = StampSchema.for_stamp(stamp)
            print(json.dumps({"portable": json.loads(stamp.to_payload()), "compact": schema.pack(stamp),
                              "schema": {"id": schema.identity, "views": schema.views}}, indent=2))
    except (ValueError, TypeError, OSError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()

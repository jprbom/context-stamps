"""Verify recorded source bytes, using bounded immutable archives when code evolves.

Archives are data only. They are never extracted or executed by this module.
"""

import base64
import gzip
import hashlib
import io
import json
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]
ARCHIVES = ROOT / "evidence/engineering-sources-v1"
MAX_ARCHIVE = 16 * 1024 * 1024


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def checked_path(name, root=ROOT):
    if not isinstance(name, str):
        raise ValueError("relative source path required")
    path = PurePosixPath(name)
    if (path.is_absolute() or not path.parts
            or any(p in (".", "..") or ":" in p or "\\" in p for p in path.parts)):
        raise ValueError("relative source path required")
    resolved = (root / name).resolve()
    if not resolved.is_relative_to(root.resolve()):
        raise ValueError("source path outside archive root")
    return resolved


def verify_sources(hashes):
    if type(hashes) is not dict or not 1 <= len(hashes) <= 512:
        raise ValueError("bounded recorded source map required")
    archived, payloads, index = [], {}, None
    for name, expected in hashes.items():
        if not isinstance(expected, str) or len(expected) != 64 or any(c not in "0123456789abcdef" for c in expected):
            raise ValueError("source digest required")
        path = checked_path(name)
        if path.is_file() and sha(path.read_bytes()) == expected:
            continue
        if index is None:
            index = json.loads((ARCHIVES / "index.json").read_text(encoding="utf-8"))
            if type(index) is not dict or type(index.get("archives")) is not list or not 1 <= len(index["archives"]) <= 64:
                raise ValueError("bounded archive index required")
        found = False
        for item in index["archives"]:
            archive_path = checked_path(item["file"], ARCHIVES)
            if item["file"] not in payloads:
                if archive_path.stat().st_size > MAX_ARCHIVE:
                    raise ValueError("source archive size limit")
                compressed = archive_path.read_bytes()
                if len(compressed) > MAX_ARCHIVE or sha(compressed) != item["sha256"]:
                    raise ValueError("source archive integrity failure")
                with gzip.GzipFile(fileobj=io.BytesIO(compressed)) as stream:
                    raw = stream.read(MAX_ARCHIVE + 1)
                if len(raw) > MAX_ARCHIVE:
                    raise ValueError("source archive expansion limit")
                payloads[item["file"]] = json.loads(raw)
            encoded = payloads[item["file"]]["content_by_sha256"].get(expected)
            if encoded is not None:
                content = base64.b64decode(encoded, validate=True)
                if sha(content) != expected:
                    raise ValueError("archived source digest mismatch")
                archived.append(name)
                found = True
                break
        if not found:
            raise ValueError("recorded source unavailable: " + name)
    return tuple(archived)

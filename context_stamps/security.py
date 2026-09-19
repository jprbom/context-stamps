"""Resource limits for a small local store, not a multi-tenant service boundary."""

import os
from pathlib import Path

MAX_TEXT = 65536
MAX_RECORDS = 1000


def bounded_text(value, maximum=MAX_TEXT):
    if not isinstance(value, str) or len(value) > maximum or len(value.encode("utf-8")) > maximum:
        raise ValueError(f"text must be a string of at most {maximum} UTF-8 bytes")


def identifier(value):
    bounded_text(value, 512)
    if not value or any(ord(c) < 32 or ord(c) == 127 for c in value):
        raise ValueError("identifier must be nonempty and contain no control characters")


def version_map(value, maximum=128):
    if value is None:
        return
    if len(value) > maximum:
        raise ValueError(f"version map exceeds {maximum} entries")
    for key, revision in value.items():
        identifier(key)
        bounded_text(revision, 512)
        if not revision:
            raise ValueError("version must not be empty")


def read_text(path, maximum=MAX_TEXT):
    target = Path(path)
    if target.is_symlink() or not target.is_file():
        raise ValueError("input must be a regular file, not a symbolic link")
    with target.open("rb") as stream:
        data = stream.read(maximum + 1)
    if len(data) > maximum:
        raise ValueError(f"file exceeds {maximum} bytes")
    return data.decode("utf-8")


def prepare_database(path):
    if str(path) == ":memory:":
        return
    target = Path(path)
    if target.is_symlink():
        raise ValueError("database symbolic links are not accepted")
    try:
        descriptor = os.open(target, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError:
        if not target.is_file():
            raise ValueError("database must be a regular file") from None
        if os.name == "posix" and target.stat().st_mode & 0o077:
            raise ValueError("database permissions must be owner-only (chmod 600)") from None
    else:
        os.close(descriptor)

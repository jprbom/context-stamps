"""Explicit allowlisted file observation. No recursive crawling or background watcher."""

from pathlib import Path

from stamps import content_digest

from .security import identifier, read_text


def observe_files(memory, root, paths):
    """Observe current bytes, invalidate missing sources and return a revision map.

    Call again immediately before selection. Source IDs are relative POSIX paths;
    private paths and absolute machine locations are not recorded in the store.
    """
    root = Path(root).resolve(strict=True)
    paths = list(paths)
    if not 1 <= len(paths) <= 128:
        raise ValueError("provide 1..128 explicitly allowed relative paths")
    prepared = []
    for value in paths:
        relative = Path(value)
        if relative.anchor or ".." in relative.parts or ":" in str(value):
            raise ValueError("source must be a relative path inside root")
        source = relative.as_posix()
        identifier(source)
        target = root / relative
        if not target.resolve().is_relative_to(root):
            raise ValueError("source escapes root")
        cursor = target
        while cursor != root:
            if cursor.is_symlink():
                raise ValueError("symbolic links are not allowed in source paths")
            cursor = cursor.parent
        if not target.resolve().is_relative_to(root):
            raise ValueError("source escapes root")
        prepared.append((source, read_text(target) if target.exists() else None))
    revisions, events = {}, []
    for source, text in prepared:
        if text is None:
            events.append({"source": source, "status": "missing", "invalidated": memory.invalidate(source)})
        else:
            events.append(memory.add(text, source=source))
            revisions[source] = content_digest(text)
    return {"revisions": revisions, "events": events}


def explain_versions(memory, source, revisions):
    item = memory.get(source)
    if item is None:
        return {"source": source, "status": "missing", "changes": []}
    expected = {**item["dependencies"], source: item["digest"]}
    changes = [
        {
            "source": key,
            "expected": value,
            "observed": revisions.get(key),
            "reason": "missing" if key not in revisions else "changed",
        }
        for key, value in expected.items()
        if revisions.get(key) != value
    ]
    status = (
        "stale"
        if item["stale"] or any(x["reason"] == "changed" for x in changes)
        else ("unknown" if changes else "current")
    )
    return {"source": source, "status": status, "changes": changes}

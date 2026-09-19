"""Local context memory. Source updates are explicit; there is no hidden file watcher."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import asdict, dataclass
from typing import Callable, Mapping, Protocol

from stamps import Family, HashingEncoder, Stamp, content_digest, similarity, stamp_vector


class Encoder(Protocol):
    identity: str
    dim: int

    def encode(self, text: str) -> list[float]: ...


@dataclass
class Packet:
    text: str
    tokens: int
    budget: int
    counting: str
    decisions: list[dict]

    def to_dict(self) -> dict:
        return asdict(self)


class ContextMemory:
    """One encoder family and one active revision per source in a SQLite file.

    Text stays local. No source is treated as unchanged based on approximate
    similarity. Full chunks are packed; chunks that do not fit are omitted,
    not silently truncated. Partition untrusted tenants into separate stores.
    """

    def __init__(
        self, path: str = ":memory:", *, encoder: Encoder | None = None, family: Family | None = None
    ) -> None:
        self.encoder = encoder or HashingEncoder()
        self.db = sqlite3.connect(str(path))
        self.db.row_factory = sqlite3.Row
        try:
            self.db.executescript("""
                CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS items (
                    source TEXT PRIMARY KEY, text TEXT NOT NULL, digest TEXT NOT NULL,
                    stamp TEXT NOT NULL, dependencies TEXT NOT NULL,
                    stale INTEGER NOT NULL DEFAULT 0
                );
            """)
            row = self.db.execute("SELECT value FROM metadata WHERE key='family'").fetchone()
            stored = Family.from_json(row[0]) if row else None
            self.family = family or stored or Family(self.encoder.identity, self.encoder.dim)
            if (self.family.encoder, self.family.dim) != (self.encoder.identity, self.encoder.dim):
                raise ValueError("encoder identity/dimension differs from family; use a new store")
            if stored and stored.identity != self.family.identity:
                raise ValueError("store uses a different family; create a new store and re-ingest")
            version = self.db.execute("SELECT value FROM metadata WHERE key='schema'").fetchone()
            if version and version[0] != "1":
                raise ValueError("unsupported memory schema")
            with self.db:
                self.db.execute("INSERT OR IGNORE INTO metadata VALUES ('schema', '1')")
                self.db.execute(
                    "INSERT OR IGNORE INTO metadata VALUES ('family', ?)", (self.family.to_json(),)
                )
        except Exception:
            self.db.close()
            raise

    def __enter__(self) -> ContextMemory:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def close(self) -> None:
        self.db.close()

    def add(self, text: str, *, source: str, dependencies: Mapping[str, str] | None = None) -> dict:
        if not source or not isinstance(source, str) or any(ord(c) < 32 for c in source):
            raise ValueError("source must be a nonempty identifier without control characters")
        if not isinstance(text, str) or not text.strip():
            raise ValueError("text must not be empty")
        deps = dict(dependencies or {})
        if any(not isinstance(k, str) or not isinstance(v, str) or not k or not v for k, v in deps.items()):
            raise ValueError("dependencies must map nonempty names to version strings")
        digest = content_digest(text)
        previous = self.db.execute("SELECT * FROM items WHERE source=?", (source,)).fetchone()
        encoded_deps = json.dumps(deps, sort_keys=True)
        # A re-observation can restore an invalidated record, without another embedding.
        if previous and previous["digest"] == digest:
            code = previous["stamp"]
            status = (
                "unchanged"
                if previous["dependencies"] == encoded_deps and not previous["stale"]
                else "refreshed"
            )
        else:
            code = str(stamp_vector(self.encoder.encode(text), self.family))
            status = "updated" if previous else "added"
        with self.db:
            if previous and (previous["digest"] != digest or previous["dependencies"] != encoded_deps):
                # Follow only declared links; discovery of undeclared dependencies is not implied.
                self._invalidate(source)
            self.db.execute(
                """INSERT INTO items VALUES (?, ?, ?, ?, ?, 0)
                ON CONFLICT(source) DO UPDATE SET text=excluded.text, digest=excluded.digest,
                stamp=excluded.stamp, dependencies=excluded.dependencies, stale=0""",
                (source, text, digest, code, encoded_deps),
            )
        return {"source": source, "digest": digest, "stamp": code, "status": status}

    def get(self, source: str) -> dict | None:
        row = self.db.execute("SELECT * FROM items WHERE source=?", (source,)).fetchone()
        if not row:
            return None
        item = dict(row)
        item["dependencies"] = json.loads(item["dependencies"])
        item["stale"] = bool(item["stale"])
        return item

    def invalidate(self, source: str) -> int:
        """Mark a source and its explicitly declared transitive dependents stale."""
        with self.db:
            return self._invalidate(source)

    def _invalidate(self, source: str) -> int:
        """Participate in the caller's transaction without an intermediate commit."""
        affected = {source}
        rows = self.db.execute("SELECT source, dependencies FROM items").fetchall()
        while True:
            more = {r["source"] for r in rows if affected.intersection(json.loads(r["dependencies"]))}
            if more.issubset(affected):
                break
            affected.update(more)
        count = 0
        for name in affected:
            count += self.db.execute("UPDATE items SET stale=1 WHERE source=? AND stale=0", (name,)).rowcount
        return count

    def forget(self, source: str) -> bool:
        with self.db:
            self._invalidate(source)
            return bool(self.db.execute("DELETE FROM items WHERE source=?", (source,)).rowcount)

    @staticmethod
    def _freshness(item: dict, revisions: Mapping[str, str] | None) -> str:
        if item["stale"]:
            return "invalidated"
        if revisions is None:
            return "unchecked"
        expected = {**json.loads(item["dependencies"]), item["source"]: item["digest"]}
        if any(key not in revisions for key in expected):
            return "unknown_version"
        if any(revisions[key] != version for key, version in expected.items()):
            return "stale_version"
        return "current"

    def _rank(self, query: str, revisions: Mapping[str, str] | None) -> list[dict]:
        q = stamp_vector(self.encoder.encode(query), self.family)
        result = []
        for row in self.db.execute("SELECT * FROM items"):
            item = dict(row)
            item["freshness"] = self._freshness(item, revisions)
            item["score"] = similarity(q, Stamp.parse(item["stamp"]))
            result.append(item)
        return sorted(result, key=lambda r: (-r["score"], r["source"]))

    def recall(self, query: str, *, limit: int = 5, revisions: Mapping[str, str] | None = None) -> list[dict]:
        if limit < 0:
            raise ValueError("limit must be nonnegative")
        return [
            {k: item[k] for k in ("source", "text", "digest", "score", "freshness")}
            for item in self._rank(query, revisions)
            if item["freshness"] in {"current", "unchecked"}
        ][:limit]

    def pack(
        self,
        query: str,
        *,
        token_budget: int = 2048,
        token_counter: Callable[[str], int] | None = None,
        revisions: Mapping[str, str] | None = None,
        min_score: float = 0.0,
    ) -> Packet:
        if token_budget < 0 or not 0 <= min_score <= 1:
            raise ValueError("budget must be nonnegative; min_score must be in [0, 1]")
        # Byte count is a conservative surrogate for conventional byte-based tokenizers.
        # The caller's tokenizer is required for a model-specific token guarantee.
        counter = token_counter or (lambda s: len(s.encode("utf-8")))
        counting = "supplied tokenizer" if token_counter else "UTF-8 byte budget (not model tokens)"
        selected: list[str] = []
        decisions: list[dict] = []
        seen: dict[tuple[str, str], str] = {}
        for item in self._rank(query, revisions):
            decision = {k: item[k] for k in ("source", "digest", "score", "freshness")}
            key = (item["digest"], item["dependencies"])
            if item["freshness"] not in {"unchecked", "current"}:
                decision["reason"] = item["freshness"]
            elif item["score"] < min_score:
                decision["reason"] = "below_threshold"
            elif key in seen:
                decision.update(reason="exact_duplicate", duplicate_of=seen[key])
            else:
                # JSON headers make source identifiers unambiguous. They are data, not instructions.
                header = json.dumps({"source": item["source"], "sha256": item["digest"]}, ensure_ascii=False)
                chunk = header + "\n" + item["text"]
                candidate = "\n\n".join([*selected, chunk])
                count = counter(candidate)
                if not isinstance(count, int) or count < 0:
                    raise ValueError("token_counter must return a nonnegative integer")
                if count > token_budget:
                    decision["reason"] = "budget"
                else:
                    selected.append(chunk)
                    seen[key] = item["source"]
                    decision["reason"] = "selected"
            decisions.append(decision)
        text = "\n\n".join(selected)
        return Packet(text, counter(text), token_budget, counting, decisions)

"""Context Stamps: a standalone reference implementation.

Copyright (c) 2026 Prashant Jagtap. MIT License; see LICENSE.
Run `python stamps.py --demo`. Python 3.10+, standard library only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
from dataclasses import asdict, dataclass
from functools import cached_property, lru_cache
from pathlib import Path
from typing import Sequence

FORMAT_VERSION = 1


def content_digest(text: str) -> str:
    """SHA-256 over exact UTF-8 bytes, without whitespace normalization."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _vector(values: Sequence[float], dim: int) -> tuple[float, ...]:
    result = tuple(float(x) for x in values)
    if len(result) != dim or not all(math.isfinite(x) for x in result):
        raise ValueError(f"expected {dim} finite numbers")
    return result


@dataclass(frozen=True)
class Family:
    """Projection space. Encoder identity includes preprocessing and model revision."""

    encoder: str
    dim: int
    bits: int = 128
    seed: int = 20260919
    method: str = "gaussian-v1"
    mean: tuple[float, ...] = ()
    planes: tuple[tuple[float, ...], ...] = ()
    version: int = FORMAT_VERSION

    def __post_init__(self) -> None:
        if not self.encoder or not isinstance(self.encoder, str):
            raise ValueError("encoder identity is required")
        if not isinstance(self.dim, int) or not 1 <= self.dim <= 16384:
            raise ValueError("dimension must be between 1 and 16384")
        if not isinstance(self.bits, int) or not 8 <= self.bits <= 512 or self.bits % 8:
            raise ValueError("bits must be a multiple of 8 between 8 and 512")
        if self.dim * self.bits > 1048576:
            raise ValueError("projection exceeds 1048576 coefficients")
        if self.version != FORMAT_VERSION:
            raise ValueError("unsupported family version")
        if self.method not in {"gaussian-v1", "centered-v1", "itq-v1"}:
            raise ValueError("unsupported projection method")
        if self.mean:
            object.__setattr__(self, "mean", _vector(self.mean, self.dim))
        if self.method != "gaussian-v1" and not self.mean:
            raise ValueError("trained families require a mean")
        if self.planes:
            if len(self.planes) != self.bits:
                raise ValueError("projection must contain one row per bit")
            object.__setattr__(self, "planes", tuple(_vector(p, self.dim) for p in self.planes))
        if self.method == "itq-v1" and not self.planes:
            raise ValueError("ITQ requires stored projection rows")

    @cached_property
    def identity(self) -> str:
        return content_digest(self.to_json())

    def to_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True, separators=(",", ":"), allow_nan=False)

    @classmethod
    def from_json(cls, value: str) -> Family:
        if len(value) > 33554432:
            raise ValueError("family JSON exceeds 32 MiB")
        data = json.loads(value)
        data["mean"] = tuple(data.get("mean", ()))
        data["planes"] = tuple(tuple(row) for row in data.get("planes", ()))
        return cls(**data)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(self.to_json() + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> Family:
        with Path(path).open("r", encoding="utf-8") as stream:
            return cls.from_json(stream.read(33554433))


@lru_cache(maxsize=8)
def _planes(family: Family) -> tuple[tuple[float, ...], ...]:
    if family.planes:
        return family.planes
    # Gaussian normals are rotationally symmetric, unlike uniform coordinates.
    # Reproducible projections; this RNG is not used for cryptography.
    rng = random.Random(family.seed)  # nosec B311
    return tuple(tuple(rng.gauss(0, 1) for _ in range(family.dim)) for _ in range(family.bits))


@dataclass(frozen=True)
class Stamp:
    family_id: str
    bits: int
    value: int

    def __post_init__(self) -> None:
        if not re.fullmatch(r"[0-9a-f]{64}", self.family_id):
            raise ValueError("family id must be a SHA-256 hexadecimal digest")
        if not 8 <= self.bits <= 512 or self.bits % 8:
            raise ValueError("invalid stamp width")
        if not 0 <= self.value < (1 << self.bits):
            raise ValueError("stamp value exceeds its width")

    def __str__(self) -> str:
        return f"cs1:{self.family_id}:{self.bits}:{self.value:0{self.bits // 4}x}"

    @classmethod
    def parse(cls, text: str) -> Stamp:
        try:
            prefix, family_id, width, payload = text.split(":")
            bits = int(width)
            if prefix != "cs1" or len(payload) != bits // 4 or not re.fullmatch(r"[0-9a-f]+", payload):
                raise ValueError("invalid stamp encoding")
            return cls(family_id, bits, int(payload, 16))
        except (TypeError, ValueError) as exc:
            raise ValueError("invalid cs1 stamp") from exc


def stamp_vector(vector: Sequence[float], family: Family) -> Stamp:
    values = _vector(vector, family.dim)
    if not any(values):
        raise ValueError("a zero vector has no direction")
    centered = tuple(x - m for x, m in zip(values, family.mean)) if family.mean else values
    code = 0
    for index, plane in enumerate(family.planes or _planes(family)):
        if math.fsum(x * p for x, p in zip(centered, plane)) >= 0:
            code |= 1 << index
    return Stamp(family.identity, family.bits, code)


def hamming(a: Stamp, b: Stamp) -> int:
    if (a.family_id, a.bits) != (b.family_id, b.bits):
        raise ValueError("stamps belong to different families; re-encode before comparing")
    return (a.value ^ b.value).bit_count()


def similarity(a: Stamp, b: Stamp) -> float:
    """Bit agreement in [0, 1], not a calibrated cosine or probability."""
    return 1.0 - hamming(a, b) / a.bits


class HashingEncoder:
    """Deterministic lexical baseline, NOT a neural semantic embedding model.

    Case-sensitive tokens, punctuation and adjacent token pairs are feature-hashed.
    Exact reuse must still use content_digest, never this lossy representation.
    """

    def __init__(self, dim: int = 384) -> None:
        if not 8 <= dim <= 16384:
            raise ValueError("hashing dimension must be between 8 and 16384")
        self.dim = dim
        self.identity = f"lexical-sha256-token-bigram-v1:dim={dim}"

    def encode(self, text: str) -> list[float]:
        tokens = re.findall(r"\w+|[^\w\s]", text, flags=re.UNICODE)
        if not tokens:
            raise ValueError("text must contain a non-whitespace token")
        features = ["t:" + t for t in tokens]
        features.extend("b:" + a + "\0" + b for a, b in zip(tokens, tokens[1:]))
        values = [0.0] * self.dim
        for feature in features:
            digest = hashlib.sha256(feature.encode("utf-8")).digest()
            values[int.from_bytes(digest[:4], "big") % self.dim] += 1 if digest[4] & 1 else -1
        norm = math.sqrt(math.fsum(x * x for x in values))
        if not norm:
            raise ValueError("lexical features cancelled; use a larger dimension")
        return [x / norm for x in values]


def demo() -> dict:
    encoder = HashingEncoder()
    family = Family(encoder.identity, encoder.dim)
    original = "if balance >= amount: transfer(amount)"
    changed = "if balance > amount: transfer(amount)"
    a = stamp_vector(encoder.encode(original), family)
    b = stamp_vector(encoder.encode(changed), family)
    return {
        "encoder": "lexical baseline (no downloaded model)",
        "bits": family.bits,
        "exact_repeat": content_digest(original) == content_digest(original),
        "changed_content_is_duplicate": content_digest(original) == content_digest(changed),
        "changed_content_bit_agreement": similarity(a, b),
        "decision": "keep changed content even when fingerprints are similar",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--demo", action="store_true", help="run an offline change-detection example")
    args = parser.parse_args()
    if args.demo:
        print(json.dumps(demo(), indent=2))
    else:
        parser.print_help()

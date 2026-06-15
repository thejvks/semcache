"""Embedding providers.

`LocalHashingEmbedder` is a dependency-free, deterministic embedder (a hashing
bag-of-words projected to a fixed dimension, L2-normalized). It makes the whole
system runnable and testable offline — semantically-similar prompts land close
in vector space without any API call.

In production, swap in `RemoteEmbedder` backed by a real embeddings API; the
interface (`embed(text) -> list[float]`) is identical.
"""
from __future__ import annotations

import hashlib
import math
import re
from typing import Callable, Protocol


class Embedder(Protocol):
    dim: int

    def embed(self, text: str) -> list[float]:
        ...


_TOKEN = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


class LocalHashingEmbedder:
    """Hashing vectorizer. Stable across runs (uses sha1, not Python's salted hash)."""

    def __init__(self, dim: int = 256):
        self.dim = dim

    def _bucket(self, token: str) -> tuple[int, float]:
        h = hashlib.sha1(token.encode()).digest()
        idx = int.from_bytes(h[:4], "big") % self.dim
        sign = 1.0 if h[4] & 1 else -1.0
        return idx, sign

    def embed(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        tokens = _tokenize(text)
        for tok in tokens:
            idx, sign = self._bucket(tok)
            vec[idx] += sign
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]


class RemoteEmbedder:
    """Adapter for a real embeddings API. `embed_fn` maps text -> vector."""

    def __init__(self, embed_fn: Callable[[str], list[float]], dim: int):
        self._embed_fn = embed_fn
        self.dim = dim

    def embed(self, text: str) -> list[float]:
        return self._embed_fn(text)


def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)) or 1.0
    nb = math.sqrt(sum(y * y for y in b)) or 1.0
    return dot / (na * nb)

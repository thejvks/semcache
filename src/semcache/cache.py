"""SemanticCache — ties the embedder + vector store + similarity threshold
together and tracks hit-rate / savings analytics.
"""
from __future__ import annotations

import datetime as dt
import hashlib
from dataclasses import dataclass, field
from typing import Any

from .embedding import Embedder, LocalHashingEmbedder
from .store import CacheEntry, VectorStore


@dataclass
class CacheStats:
    lookups: int = 0
    hits: int = 0
    misses: int = 0
    # Tokens we *did not* send upstream because of cache hits (for savings math).
    tokens_saved: int = 0

    @property
    def hit_rate(self) -> float:
        return round(self.hits / self.lookups, 4) if self.lookups else 0.0


@dataclass
class LookupResult:
    hit: bool
    response: Any | None
    similarity: float
    matched_prompt: str | None = None


@dataclass
class SemanticCache:
    embedder: Embedder = field(default_factory=LocalHashingEmbedder)
    store: VectorStore = field(default_factory=VectorStore)
    threshold: float = 0.92
    ttl_seconds: int | None = 3600
    stats: CacheStats = field(default_factory=CacheStats)

    def _key(self, prompt: str) -> str:
        return hashlib.sha256(prompt.encode()).hexdigest()[:32]

    def lookup(self, prompt: str, now: dt.datetime | None = None) -> LookupResult:
        now = now or dt.datetime.utcnow()
        self.stats.lookups += 1
        emb = self.embedder.embed(prompt)
        entry, sim = self.store.search(emb, self.threshold, now)
        if entry is not None:
            entry.hits += 1
            self.stats.hits += 1
            return LookupResult(hit=True, response=entry.response, similarity=round(sim, 4),
                                matched_prompt=entry.prompt)
        self.stats.misses += 1
        return LookupResult(hit=False, response=None, similarity=round(sim, 4))

    def store_response(self, prompt: str, response: Any, model: str = "unknown",
                       tokens: int = 0, now: dt.datetime | None = None) -> None:
        now = now or dt.datetime.utcnow()
        expires = now + dt.timedelta(seconds=self.ttl_seconds) if self.ttl_seconds else None
        self.store.add(CacheEntry(
            key=self._key(prompt),
            prompt=prompt,
            embedding=self.embedder.embed(prompt),
            response=response,
            model=model,
            created_at=now,
            expires_at=expires,
        ))

    def record_saved_tokens(self, tokens: int) -> None:
        self.stats.tokens_saved += tokens

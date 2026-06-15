"""Vector cache store with TTL + nearest-neighbor lookup.

In-memory and brute-force for the MVP (correct and simple). The interface is
ANN-ready: swap in FAISS / pgvector / a managed vector DB behind `search`
without touching the cache logic.
"""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from typing import Any

from .embedding import cosine


@dataclass
class CacheEntry:
    key: str
    prompt: str
    embedding: list[float]
    response: Any
    model: str
    created_at: dt.datetime
    expires_at: dt.datetime | None
    hits: int = 0


@dataclass
class VectorStore:
    max_entries: int = 10_000
    _entries: dict[str, CacheEntry] = field(default_factory=dict)

    def _evict_if_needed(self):
        if len(self._entries) <= self.max_entries:
            return
        # Evict the least-recently-created entry (simple FIFO-ish policy).
        oldest = min(self._entries.values(), key=lambda e: e.created_at)
        self._entries.pop(oldest.key, None)

    def add(self, entry: CacheEntry) -> None:
        self._entries[entry.key] = entry
        self._evict_if_needed()

    def purge_expired(self, now: dt.datetime) -> int:
        expired = [k for k, e in self._entries.items() if e.expires_at and e.expires_at <= now]
        for k in expired:
            self._entries.pop(k, None)
        return len(expired)

    def search(self, embedding: list[float], threshold: float, now: dt.datetime) -> tuple[CacheEntry | None, float]:
        best: CacheEntry | None = None
        best_sim = -1.0
        for entry in self._entries.values():
            if entry.expires_at and entry.expires_at <= now:
                continue
            sim = cosine(embedding, entry.embedding)
            if sim > best_sim:
                best, best_sim = entry, sim
        if best is not None and best_sim >= threshold:
            return best, best_sim
        return None, best_sim if best is not None else 0.0

    def __len__(self) -> int:
        return len(self._entries)

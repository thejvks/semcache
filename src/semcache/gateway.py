"""FastAPI semantic-cache proxy.

POST /v1/chat — checks the cache by semantic similarity. On hit, returns the
cached response instantly (no upstream call). On miss, forwards to the upstream
LLM, caches the result, and returns it. GET /v1/stats exposes hit-rate/savings.
"""
from __future__ import annotations

from typing import Any, Callable

import httpx
from fastapi import FastAPI

from .cache import SemanticCache
from .config import Settings, get_settings
from .embedding import LocalHashingEmbedder

Forwarder = Callable[[dict], "httpx.Response"]


def canonical_prompt(body: dict) -> str:
    """Collapse a chat request into a single string for embedding."""
    if "prompt" in body:
        return str(body["prompt"])
    msgs = body.get("messages", [])
    return "\n".join(f"{m.get('role','')}: {m.get('content','')}" for m in msgs)


async def _default_forwarder(body: dict) -> httpx.Response:
    settings = get_settings()
    async with httpx.AsyncClient(timeout=120.0) as client:
        return await client.post(settings.upstream_url, json=body)


def create_app(settings: Settings | None = None, cache: SemanticCache | None = None,
               forwarder: Forwarder | None = None) -> FastAPI:
    settings = settings or get_settings()
    cache = cache or SemanticCache(
        embedder=LocalHashingEmbedder(dim=settings.embedding_dim),
        threshold=settings.similarity_threshold,
        ttl_seconds=settings.ttl_seconds,
    )
    forward = forwarder or _default_forwarder

    app = FastAPI(title="SemCache", version="0.1.0")
    app.state.cache = cache

    @app.get("/healthz")
    def healthz():
        return {"status": "ok", "entries": len(cache.store)}

    @app.post("/v1/chat")
    async def chat(body: dict):
        prompt = canonical_prompt(body)
        result = cache.lookup(prompt)
        if result.hit:
            # Estimate tokens saved from the cached response's usage, if present.
            saved = (result.response or {}).get("usage", {}).get("total_tokens", 0)
            cache.record_saved_tokens(int(saved or 0))
            return {"cached": True, "similarity": result.similarity, "response": result.response}

        upstream = await forward(body)
        try:
            data = upstream.json()
        except Exception:
            data = {"error": "non-json upstream response"}
        cache.store_response(prompt, data, model=str(body.get("model", "unknown")))
        return {"cached": False, "similarity": result.similarity, "response": data}

    @app.get("/v1/stats")
    def stats():
        s = cache.stats
        return {
            "lookups": s.lookups,
            "hits": s.hits,
            "misses": s.misses,
            "hit_rate": s.hit_rate,
            "tokens_saved": s.tokens_saved,
            "entries": len(cache.store),
            "threshold": cache.threshold,
        }

    return app

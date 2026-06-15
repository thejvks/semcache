import httpx
from fastapi.testclient import TestClient

from semcache.cache import SemanticCache
from semcache.config import Settings
from semcache.gateway import create_app


def make_client(call_counter):
    async def fake_forward(body):
        call_counter["n"] += 1
        return httpx.Response(200, json={
            "model": body.get("model", "x"),
            "content": "the answer",
            "usage": {"total_tokens": 200},
        })

    # Lexical embedder: near-duplicates land ~0.83, unrelated ~0.0 -> 0.8 separates them.
    settings = Settings(similarity_threshold=0.8)
    cache = SemanticCache(threshold=0.8)
    app = create_app(settings=settings, cache=cache, forwarder=fake_forward)
    return TestClient(app), call_counter


def test_miss_then_semantic_hit_avoids_second_upstream_call():
    counter = {"n": 0}
    client, _ = make_client(counter)

    r1 = client.post("/v1/chat", json={"messages": [{"role": "user", "content": "how do I reset my password"}]})
    assert r1.json()["cached"] is False
    assert counter["n"] == 1

    # Near-duplicate phrasing -> should hit cache, NOT call upstream again.
    r2 = client.post("/v1/chat", json={"messages": [{"role": "user", "content": "how can I reset my password"}]})
    assert r2.json()["cached"] is True
    assert counter["n"] == 1  # no new upstream call

    stats = client.get("/v1/stats").json()
    assert stats["hits"] == 1
    assert stats["tokens_saved"] == 200


def test_unrelated_prompt_calls_upstream():
    counter = {"n": 0}
    client, _ = make_client(counter)
    client.post("/v1/chat", json={"prompt": "reset my password please"})
    client.post("/v1/chat", json={"prompt": "translate this poem into French"})
    assert counter["n"] == 2

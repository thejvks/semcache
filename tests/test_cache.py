import datetime as dt

from semcache.cache import SemanticCache
from semcache.embedding import LocalHashingEmbedder, cosine


def test_embedder_is_deterministic_and_normalized():
    e = LocalHashingEmbedder(dim=128)
    v1 = e.embed("hello world")
    v2 = e.embed("hello world")
    assert v1 == v2
    # near unit norm
    assert abs(sum(x * x for x in v1) ** 0.5 - 1.0) < 1e-9


def test_similar_prompts_are_closer_than_unrelated():
    e = LocalHashingEmbedder(dim=512)
    base = e.embed("how do I reset my password")
    similar = e.embed("how can I reset my password")
    unrelated = e.embed("what is the capital of France")
    assert cosine(base, similar) > cosine(base, unrelated)


def test_exact_repeat_is_a_hit():
    cache = SemanticCache(threshold=0.9)
    cache.store_response("ping?", {"answer": "pong", "usage": {"total_tokens": 10}})
    r = cache.lookup("ping?")
    assert r.hit and r.similarity > 0.99
    assert r.response["answer"] == "pong"


def test_unrelated_prompt_is_a_miss():
    cache = SemanticCache(threshold=0.9)
    cache.store_response("how do I reset my password and recover access", {"answer": "x"})
    r = cache.lookup("what is the weather in Tokyo tomorrow")
    assert r.hit is False


def test_ttl_expiry():
    cache = SemanticCache(threshold=0.9, ttl_seconds=60)
    t0 = dt.datetime(2026, 1, 1, 0, 0, 0)
    cache.store_response("cached q", {"a": 1}, now=t0)
    assert cache.lookup("cached q", now=t0 + dt.timedelta(seconds=30)).hit is True
    assert cache.lookup("cached q", now=t0 + dt.timedelta(seconds=120)).hit is False


def test_stats_hit_rate():
    cache = SemanticCache(threshold=0.9)
    cache.store_response("alpha beta gamma", {"a": 1})
    cache.lookup("alpha beta gamma")     # hit
    cache.lookup("totally different text here")  # miss
    assert cache.stats.lookups == 2
    assert cache.stats.hits == 1
    assert cache.stats.hit_rate == 0.5

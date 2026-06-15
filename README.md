# SemCache — Semantic Caching Layer for LLM APIs

> A proxy that returns cached answers for **semantically-similar** prompts (embedding cosine similarity above a threshold), cutting LLM spend and latency on repetitive traffic.

![status](https://img.shields.io/badge/tests-8%20passing-brightgreen) ![python](https://img.shields.io/badge/python-3.10%2B-blue) ![license](https://img.shields.io/badge/license-MIT-black)

Exact-match caching barely helps LLM workloads — users phrase the same question a hundred ways. SemCache caches on **meaning**, not string equality, so "how do I reset my password" hits the cache for "how can I reset my password." Concrete cost savings you can put a number on.

---

## The problem

Support bots, RAG assistants, and internal tools answer the same handful of questions thousands of times a day in slightly different wording. Every variation is a fresh, paid, multi-second LLM call. A traditional key-value cache never hits because the strings differ. SemCache turns that repetitive long tail into instant, free cache hits.

## Who uses it

- Teams running **high-volume, repetitive** LLM traffic (support, FAQ, RAG).
- Platform teams who want a **drop-in proxy** that lowers spend without app changes.
- Anyone needing **latency** wins on common queries (cache hit ≈ instant).

## What it proves (skills)

Embeddings + vector similarity, nearest-neighbor cache design with **TTL and eviction**, a clean provider abstraction (local vs. remote embedder), proxy plumbing, and **hit-rate/savings analytics**. Real backend-systems work, not a wrapper.

---

## Architecture

```
  client ──POST /v1/chat──►  SemCache
                              │ 1. embed(prompt)
                              │ 2. nearest neighbor in vector store
                              │      cosine ≥ threshold ?
                              │        ├── HIT  → return cached response (no upstream)
                              │        └── MISS → forward upstream, cache result, return
                              ▼
                        VectorStore (TTL + eviction)   ── /v1/stats: hit-rate, tokens saved
                              │
            (swap brute-force for FAISS / pgvector at scale — same interface)
```

| Module | Responsibility |
|--------|----------------|
| `embedding.py` | `LocalHashingEmbedder` (offline, deterministic) + `RemoteEmbedder` adapter; `cosine` |
| `store.py` | TTL-aware vector store with nearest-neighbor `search` + eviction |
| `cache.py` | `SemanticCache`: lookup/store + hit-rate & savings stats |
| `gateway.py` | FastAPI `/v1/chat` proxy + `/v1/stats` |
| `cli.py` | `serve`, `selftest` (offline demo) |

### A note on the bundled embedder (honesty matters)

The default `LocalHashingEmbedder` is a **lexical** hashing vectorizer — zero dependencies, fully offline, deterministic, perfect for tests/CI and demos. It captures word overlap, so set the threshold around **0.8**. In production, plug a real embeddings model via `RemoteEmbedder`; those resolve genuine semantics and you'd raise the threshold to **~0.92+**. The cache, store, TTL, eviction, and analytics are identical either way.

---

## Tech stack

**Python 3.10+ · FastAPI · httpx · Pydantic v2 · Typer + Rich · pytest · Docker**

---

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt && pip install -e .

semcache selftest          # offline demo of semantic hits
semcache serve             # run the proxy  (or: docker compose up)
```

```bash
# point your app at it
curl -s localhost:8000/v1/chat -d '{"prompt":"how do I reset my password"}'
curl -s localhost:8000/v1/chat -d '{"prompt":"how can I reset my password"}'   # cached:true
curl -s localhost:8000/v1/stats
```

---

## Screenshots

> _Replace with real captures._

```
$ semcache selftest
HIT  sim=1.000  q='How do I reset my password?'
HIT  sim=0.833  q='how can i reset my password'      ← semantic hit, no upstream call
MISS sim=0.000  q='What is the capital of France?'
stats: hit_rate=67% over 3 lookups
```

---

## MVP vs. Advanced

**MVP (this repo):** semantic lookup with threshold, TTL + eviction, FastAPI proxy, hit-rate/savings stats, offline embedder, tested (incl. proxy path proving the 2nd similar call skips upstream).

**Advanced roadmap:**
- Real embeddings backend + **FAISS/pgvector** ANN index for millions of entries.
- Per-tenant namespaces & cache isolation; cache-control headers (bypass/refresh).
- **Negative caching** and staleness controls; semantic invalidation on knowledge updates.
- Savings dashboard wired to **PromptLedger** (show $ saved per feature).
- Guardrail: confidence-gated hits (fall back to upstream when top-2 neighbors disagree).

## Testing

```bash
pytest -q   # 8 tests: embedder determinism, similarity ordering, TTL, hit-rate, full proxy path
```

## Resume bullets

- *Built **SemCache**, a semantic caching proxy for LLM APIs that serves cached responses for paraphrased prompts via embedding cosine similarity — the proxy test proves a second, reworded query returns from cache with **zero upstream calls**.*
- *Designed a TTL-aware vector store with eviction and a pluggable embedder interface (offline lexical embedder for CI, real embeddings in prod), plus hit-rate and tokens-saved analytics.*

## Why a recruiter cares

LLM cost reduction is a board-level concern in 2026, and "30–60% fewer calls" is a quantifiable resume number. This shows embeddings/vector-search fluency *and* production proxy engineering. Composes with **PromptLedger** (measure the savings) and **EvalForge** (verify cached answers stay correct).

## License

MIT

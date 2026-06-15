"""CLI — run the proxy or a quick offline self-test of the cache behavior."""
from __future__ import annotations

import typer
from rich.console import Console

from .cache import SemanticCache
from .config import get_settings

app = typer.Typer(help="SemCache — semantic caching for LLM APIs.")
console = Console()


@app.command()
def serve(host: str = "0.0.0.0", port: int = 8000):
    """Run the gateway."""
    import uvicorn
    uvicorn.run("semcache.app:app", host=host, port=port)


@app.command()
def selftest():
    """Demonstrate semantic hits offline (no API needed)."""
    s = get_settings()
    cache = SemanticCache(threshold=s.similarity_threshold)
    cache.store_response("How do I reset my password?", {"answer": "Use the reset link.", "usage": {"total_tokens": 120}})

    for q in [
        "How do I reset my password?",          # exact
        "how can i reset my password",           # near-duplicate
        "What is the capital of France?",        # unrelated
    ]:
        r = cache.lookup(q)
        console.print(f"[{'green' if r.hit else 'red'}]{'HIT ' if r.hit else 'MISS'}[/] "
                      f"sim={r.similarity:.3f}  q={q!r}")
    console.print(f"\nstats: hit_rate={cache.stats.hit_rate:.0%} over {cache.stats.lookups} lookups")


def main():
    app()


if __name__ == "__main__":
    main()

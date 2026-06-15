from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="SEMCACHE_", env_file=".env", extra="ignore")

    upstream_url: str = "https://api.anthropic.com/v1/messages"
    embedding_dim: int = 256
    # Cosine similarity above this is treated as a cache hit. Tune per workload
    # AND per embedder: the bundled LocalHashingEmbedder is lexical (use ~0.8),
    # while real semantic embeddings resolve much higher (use ~0.92+).
    similarity_threshold: float = 0.8
    ttl_seconds: int | None = 3600


def get_settings() -> Settings:
    return Settings()

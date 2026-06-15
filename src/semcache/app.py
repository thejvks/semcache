"""ASGI entrypoint: `uvicorn semcache.app:app`."""
from .gateway import create_app

app = create_app()

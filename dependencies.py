import json
from contextlib import asynccontextmanager

import numpy as np
from fastapi import FastAPI
from sentence_transformers import SentenceTransformer
from slowapi import Limiter
from slowapi.util import get_remote_address

from app_config import settings


def create_limiter() -> Limiter:
    return Limiter(
        key_func=get_remote_address,
        default_limits=[settings.default_rate_limit],
    )


class CatalogRuntime:
    def __init__(self) -> None:
        self.embed_model = None
        self.catalog: list[dict] = []
        self.catalog_vecs: np.ndarray | None = None

    def load(self) -> None:
        self.embed_model = SentenceTransformer(settings.model_name, device=settings.device)
        raw = json.loads(settings.embeddings_file.read_text(encoding="utf-8"))
        self.catalog = raw
        self.catalog_vecs = np.array([item["embedding"] for item in raw], dtype=np.float32)

    def clear(self) -> None:
        self.embed_model = None
        self.catalog = []
        self.catalog_vecs = None


runtime = CatalogRuntime()


@asynccontextmanager
async def lifespan(app: FastAPI):
    runtime.load()
    print(
        f"[api] env={settings.app_env} loaded {len(runtime.catalog)} products, device={settings.device}"
    )
    yield
    runtime.clear()

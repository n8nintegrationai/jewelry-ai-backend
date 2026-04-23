import json
import time
from contextlib import asynccontextmanager

import httpx
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
        # FIX 1: Cache vector file in memory at startup with timing
        vec_start = time.time()
        raw = json.loads(settings.embeddings_file.read_text(encoding="utf-8"))
        self.catalog = raw
        self.catalog_vecs = np.array(
            [item["embedding"] for item in raw], dtype=np.float32)
        vec_time_ms = round((time.time() - vec_start) * 1000, 1)
        print(
            f"[api] Vector store loaded: {len(self.catalog)} items in {vec_time_ms}ms")

        # Load embedding model
        self.embed_model = SentenceTransformer(
            settings.model_name, device=settings.device)

        # FIX 2: Warm up the embedding model at startup
        try:
            self.embed_model.encode("warmup")
            print("[api] Embedding model warmed up")
        except Exception as e:
            print(f"[api] Warning: Failed to warm up embedding model: {e}")

        # FIX 3: Warm up Ollama/Gemma at startup
        try:
            with httpx.Client(timeout=10) as client:
                response = client.post(
                    settings.ollama_url,
                    json={"model": settings.ollama_model,
                          "prompt": "hi", "stream": False},
                )
                if response.status_code == 200:
                    print(f"[api] {settings.ollama_model} warmed up and ready")
                else:
                    print(
                        f"[api] Warning: Ollama warmup returned status {response.status_code}")
        except Exception as e:
            print(f"[api] Warning: Failed to warm up Ollama: {e}")

    def clear(self) -> None:
        self.embed_model = None
        self.catalog = []
        self.catalog_vecs = None


runtime = CatalogRuntime()


@asynccontextmanager
async def lifespan(app: FastAPI):
    runtime.load()
    print(
        f"[api] env={settings.app_env}, device={settings.device}, startup complete"
    )
    yield
    runtime.clear()

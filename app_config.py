import os
from dataclasses import dataclass
from pathlib import Path

import torch


def _parse_csv(value: str | None) -> list[str]:
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


@dataclass(frozen=True)
class Settings:
    app_env: str
    model_name: str
    ollama_url: str
    ollama_model: str
    products_url: str
    embeddings_file: Path
    top_k: int
    max_input_chars: int
    similarity_threshold: float
    default_whatsapp: str
    default_rate_limit: str
    chat_rate_limit: str
    llm_connect_timeout_seconds: float
    llm_read_timeout_seconds: float
    llm_write_timeout_seconds: float
    llm_pool_timeout_seconds: float
    cors_allow_origins: list[str]
    device: str

    @property
    def is_production(self) -> bool:
        return self.app_env == "prod"


def load_settings() -> Settings:
    app_env = os.getenv("APP_ENV", "dev").strip().lower()
    if app_env not in {"dev", "prod"}:
        raise ValueError("APP_ENV must be 'dev' or 'prod'")

    # In dev mode, allow all origins. In prod, require explicit configuration.
    if app_env == "dev":
        cors_allow_origins = ["*"]
    else:
        raw_origins = os.getenv("CORS_ALLOW_ORIGINS", "")
        cors_allow_origins = _parse_csv(
            raw_origins) if raw_origins.strip() else []

    default_ollama_url = (
        "http://172.17.0.1:11434/api/generate"
        if app_env == "prod"
        else "http://localhost:11434/api/generate"
    )

    return Settings(
        app_env=app_env,
        model_name=os.getenv(
            "MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2"),
        ollama_url=os.getenv("OLLAMA_URL", default_ollama_url),
        ollama_model=os.getenv("OLLAMA_MODEL", "phi3"),
        # ollama_model=os.getenv("OLLAMA_MODEL", "gemma4:e2b"),
        products_url=os.getenv(
            "PRODUCTS_URL",
            "https://raw.githubusercontent.com/n8nintegrationai/luvz-collection-dev/refs/heads/main/public/data/products.json",
        ),
        embeddings_file=Path(os.getenv("EMBEDDINGS_FILE", "embeddings.json")),
        top_k=int(os.getenv("TOP_K", "3")),
        max_input_chars=int(os.getenv("MAX_INPUT_CHARS", "400")),
        similarity_threshold=float(os.getenv("SIMILARITY_THRESHOLD", "0.40")),
        default_whatsapp=os.getenv("DEFAULT_WHATSAPP", "919876543210"),
        default_rate_limit=os.getenv("DEFAULT_RATE_LIMIT", "30/minute"),
        chat_rate_limit=os.getenv("CHAT_RATE_LIMIT", "15/minute"),
        llm_connect_timeout_seconds=float(
            os.getenv("LLM_CONNECT_TIMEOUT_SECONDS", "10")),
        llm_read_timeout_seconds=float(
            os.getenv("LLM_READ_TIMEOUT_SECONDS", "180")),
        llm_write_timeout_seconds=float(
            os.getenv("LLM_WRITE_TIMEOUT_SECONDS", "10")),
        llm_pool_timeout_seconds=float(
            os.getenv("LLM_POOL_TIMEOUT_SECONDS", "5")),
        cors_allow_origins=cors_allow_origins,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )


settings = load_settings()

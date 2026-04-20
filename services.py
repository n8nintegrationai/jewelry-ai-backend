import re
from collections import Counter

import numpy as np

from app_config import settings
from app_constants import NO_RESULTS_MESSAGE, SYSTEM_PROMPT
from dependencies import runtime


def _normalize_text(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", value.lower()).strip()


def _format_price(raw_price: object) -> str:
    value = str(raw_price or "").strip()
    normalized = value.lower()
    if normalized in {"", "0", "none", "consult"}:
        return "Available on Request"

    numeric = value.replace(",", "")
    try:
        if float(numeric) <= 0:
            return "Available on Request"
    except ValueError:
        return "Available on Request"

    return f"Rs. {value}"


def _format_product_line(hit: dict) -> str:
    name = hit.get("name", "Product")
    return f"{name} - {_format_price(hit.get('price'))}"


def _product_identity(hit: dict) -> tuple[object, str, str, str]:
    return (
        hit.get("id"),
        str(hit.get("name", "")).strip(),
        str(hit.get("price", "")).strip(),
        str(hit.get("url", "")).strip(),
    )


def find_exact_name_matches(query: str) -> list[dict]:
    normalized_query = _normalize_text(query)
    if not normalized_query:
        return []

    return [
        product
        for product in runtime.catalog
        if _normalize_text(str(product.get("name", ""))) == normalized_query
    ]


def cosine_search(query: str, k: int | None = None) -> list[dict]:
    if runtime.embed_model is None or runtime.catalog_vecs is None:
        raise RuntimeError("Catalog runtime is not initialized")

    limit = k or settings.top_k
    qvec = runtime.embed_model.encode([query], normalize_embeddings=True)[0]
    scores = runtime.catalog_vecs @ qvec
    valid_indices = np.where(scores >= settings.similarity_threshold)[0]
    if len(valid_indices) == 0:
        return []

    top_idx = valid_indices[np.argsort(scores[valid_indices])[::-1][:limit]]
    return [runtime.catalog[i] for i in top_idx]


def build_context(hits: list[dict]) -> str:
    return "\n".join(_format_product_line(hit) for hit in hits)


def build_sources(hits: list[dict]) -> list[dict]:
    unique_sources: list[dict] = []
    seen_products: set[tuple[object, str, str, str]] = set()

    for hit in hits:
        name = hit.get("name")
        identity = _product_identity(hit)
        if name and identity not in seen_products:
            unique_sources.append({"name": name, "url": hit.get("url")})
            seen_products.add(identity)

    return unique_sources


def build_llm_payload(message: str, context: str, exact_match: bool) -> dict:
    mode = "exact_match" if exact_match else "suggestions"
    return {
        "model": settings.ollama_model,
        "prompt": (
            f"MODE: {mode}\n"
            f"INVENTORY:\n{context}\n\n"
            f"Customer: {message}\nAssistant:"
        ),
        "system": SYSTEM_PROMPT,
        "stream": False,
        "options": {
            "temperature": 0.0,
            "stop": ["User:", "Assistant:", "INSTRUCTIONS:", "Inventory:", "User Request:", "\n\n\n", "---", "<a", "http"],
            "num_predict": 150,
        },
    }


def validate_llm_answer(answer: str, hits: list[dict], exact_match: bool) -> str:
    del exact_match

    allowed_lines = [_format_product_line(hit) for hit in hits]
    if not allowed_lines:
        return NO_RESULTS_MESSAGE

    allowed_counter = Counter(allowed_lines)
    allowed_lookup = {_normalize_text(line): line for line in allowed_counter}

    raw_lines = [line.strip() for line in answer.splitlines() if line.strip()]
    if not raw_lines:
        return build_context(hits)

    selected: list[str] = []
    selected_counter: Counter[str] = Counter()

    for line in raw_lines:
        if " - " not in line:
            continue

        allowed_line = allowed_lookup.get(_normalize_text(line))
        if allowed_line and selected_counter[allowed_line] < allowed_counter[allowed_line]:
            selected.append(allowed_line)
            selected_counter[allowed_line] += 1

    if not selected:
        selected = allowed_lines

    return "\n".join(selected)

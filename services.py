import re
from collections import Counter

import numpy as np

from app_config import settings
from app_constants import NO_RESULTS_MESSAGE, SYSTEM_PROMPT
from dependencies import runtime
from query_classifier import QueryTier, classify_query, get_clarifying_question, should_add_clarifying_question


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


def _tokenize_text(value: str) -> list[str]:
    normalized = _normalize_text(value)
    return [token for token in normalized.split() if token]


def _lexical_boost(query: str, product: dict) -> float:
    normalized_query = _normalize_text(query)
    if not normalized_query:
        return 0.0

    product_name = str(product.get("name", ""))
    normalized_name = _normalize_text(product_name)
    if not normalized_name:
        return 0.0

    boost = 0.0
    if normalized_query in normalized_name:
        boost += 0.35

    query_tokens = set(_tokenize_text(query))
    if not query_tokens:
        return boost

    name_tokens = set(_tokenize_text(product_name))
    overlap = query_tokens & name_tokens
    if overlap:
        boost += 0.25 * (len(overlap) / len(query_tokens))
        if overlap == query_tokens:
            boost += 0.20

    return boost


def find_exact_name_matches(query: str) -> list[dict]:
    normalized_query = _normalize_text(query)
    if not normalized_query:
        return []

    return [
        product
        for product in runtime.catalog
        if _normalize_text(str(product.get("name", ""))) == normalized_query
    ]


def cosine_search(query: str, k: int | None = None, tier: QueryTier | None = None) -> list[dict]:
    """
    Search catalog using cosine similarity.

    Args:
        query: The search query (may be enriched)
        k: Number of results to return (defaults to settings.top_k)
        tier: Query tier (TIER1_EXACT, TIER2_SEMANTIC, or TIER3_AMBIGUOUS)
              For TIER3, always returns top K items even if similarity is low

    Returns:
        List of matching products, never empty for TIER3
    """
    if runtime.embed_model is None or runtime.catalog_vecs is None:
        raise RuntimeError("Catalog runtime is not initialized")

    limit = k or settings.top_k
    qvec = runtime.embed_model.encode([query], normalize_embeddings=True)[0]
    semantic_scores = runtime.catalog_vecs @ qvec
    lexical_scores = np.array([_lexical_boost(query, product) for product in runtime.catalog], dtype=np.float32)
    scores = semantic_scores + lexical_scores

    # For TIER3 (ambiguous), always return top K items regardless of threshold
    if tier == QueryTier.TIER3_AMBIGUOUS:
        top_idx = np.argsort(scores)[::-1][:limit]
        return [runtime.catalog[i] for i in top_idx]

    # For TIER1 and TIER2, apply threshold
    valid_indices = np.where(semantic_scores >= settings.similarity_threshold)[0]
    if len(valid_indices) == 0:
        valid_indices = np.where(lexical_scores > 0)[0]
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


def build_llm_payload(message: str, context: str, exact_match: bool, tier: QueryTier | None = None) -> dict:
    """
    Build LLM payload in Phi-3 format.

    Args:
        message: Original user message
        context: Product inventory context
        exact_match: Whether this was an exact name match
        tier: Query tier (used for mode setting)
    """
    mode = "exact_match" if exact_match else "suggestions"

    # Use Phi-3 format: <|system|>...<|end|><|user|>...<|end|><|assistant|>
    prompt = (
        f"<|system|>{SYSTEM_PROMPT}<|end|>"
        f"<|user|>MODE: {mode}\n"
        f"INVENTORY:\n{context}\n\n"
        f"Customer: {message}<|end|>"
        f"<|assistant|>"
    )

    return {
        "model": settings.ollama_model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.0,
            "stop": ["<|end|>", "User:", "Assistant:", "INSTRUCTIONS:", "Inventory:", "User Request:", "\n\n\n", "---", "<a", "http"],
            "num_predict": 150,
        },
    }


def validate_llm_answer(answer: str, hits: list[dict], exact_match: bool, tier: QueryTier | None = None) -> str:
    del exact_match

    allowed_lines = [_format_product_line(hit) for hit in hits]
    if not allowed_lines:
        return NO_RESULTS_MESSAGE

    allowed_counter = Counter(allowed_lines)
    allowed_lookup = {_normalize_text(line): line for line in allowed_counter}

    raw_lines = [line.strip() for line in answer.splitlines() if line.strip()]
    if not raw_lines:
        result = build_context(hits)
    else:
        selected: list[str] = []
        selected_counter: Counter[str] = Counter()

        for line in raw_lines:
            if " - " not in line:
                continue

            allowed_line = allowed_lookup.get(_normalize_text(line))
            if allowed_line and selected_counter[allowed_line] < allowed_counter[allowed_line]:
                selected.append(allowed_line)
                selected_counter[allowed_line] += 1

        result = "\n".join(selected) if selected else build_context(hits)

    # Add clarifying question for TIER 3 (ambiguous queries)
    if tier == QueryTier.TIER3_AMBIGUOUS and should_add_clarifying_question(tier):
        question = get_clarifying_question(tier)
        result = f"{result}\n\n{question}"

    return result

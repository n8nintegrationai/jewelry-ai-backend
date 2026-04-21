import re

import numpy as np

from app_config import settings
from app_constants import NO_RESULTS_MESSAGE, SYSTEM_PROMPT
from dependencies import runtime
from query_classifier import QueryTier, get_clarifying_question, should_add_clarifying_question


def _normalize_text(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", value.lower()).strip()


def extract_constraints(message: str, previous_assistant_message: str = "") -> dict:
    """
    Extract price constraints from user message.

    Handles patterns:
    - "budget 500" or "budget of 500"
    - "500 budget"
    - "my budget is 500"
    - "budget around 500"
    - "around 500"
    - "roughly 500"
    - "500 rupees" / "500 rs" / "rs 500" / "₹500" / "500₹"
    - Standalone number when previous message mentions budget/price keywords

    Args:
        message: User's message (case-insensitive matching)
        previous_assistant_message: Previous assistant response for context

    Returns:
        Dict with 'max_price' key (or empty dict if no price found)
    """
    msg_lower = message.lower().strip()

    # Pattern 1: "budget 500" or "budget of 500"
    match = re.search(r'budget\s+(?:of\s+)?(\d+)', msg_lower)
    if match:
        return {"max_price": int(match.group(1))}

    # Pattern 2: "500 budget"
    match = re.search(r'(\d+)\s+budget', msg_lower)
    if match:
        return {"max_price": int(match.group(1))}

    # Pattern 3: "my budget is 500"
    match = re.search(r'my\s+budget\s+is\s+(\d+)', msg_lower)
    if match:
        return {"max_price": int(match.group(1))}

    # Pattern 4: "budget around 500"
    match = re.search(r'budget\s+around\s+(\d+)', msg_lower)
    if match:
        return {"max_price": int(match.group(1))}

    # Pattern 5: "around 500" or "roughly 500"
    match = re.search(r'(?:around|roughly|approximately)\s+(\d+)', msg_lower)
    if match:
        return {"max_price": int(match.group(1))}

    # Pattern 6a: "rs 500" or "rs. 500" or "rupees 500"
    match = re.search(r'(?:rs\.?|rupees?)\s+(\d+)', msg_lower)
    if match:
        return {"max_price": int(match.group(1))}

    # Pattern 6b: "500 rupees" or "500 rs" or "500₹" or "₹500"
    match = re.search(r'(?:₹\s*)?(\d+)\s*(?:rupees?|rs\.?|₹)?', msg_lower)
    if match:
        # Only return if we found rupees/rs in the message to avoid false positives
        if re.search(r'(\d+)\s*(?:rupees?|rs\.?|₹)|₹\s*(\d+)|(\d+)\s*₹', msg_lower):
            return {"max_price": int(match.group(1))}

    # Pattern 7: Standalone number when previous message mentioned budget keywords
    budget_keywords = ["budget", "price", "range", "narrow", "style"]
    prev_lower = previous_assistant_message.lower()
    if any(keyword in prev_lower for keyword in budget_keywords):
        # Check if message is just a number
        stripped = msg_lower.strip()
        if re.match(r'^\d+$', stripped):
            return {"max_price": int(stripped)}

    # Pattern 8: "under 500"
    match = re.search(r'under\s+(\d+)', msg_lower)
    if match:
        return {"max_price": int(match.group(1))}

    return {}


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


def cosine_search(query: str, k: int | None = None, tier: QueryTier | None = None, original_query: str | None = None, constraints: dict | None = None) -> list[dict]:
    """
    Search catalog using cosine similarity.

    Args:
        query: The search query (may be enriched)
        k: Number of results to return (defaults to settings.top_k)
        tier: Query tier (TIER1_EXACT, TIER2_SEMANTIC, or TIER3_AMBIGUOUS)
              For TIER3, always returns top K items even if similarity is low
        original_query: The original unmodified query (for TIER2 validation)
        constraints: Dict with max_price, category, etc. for filtering

    Returns:
        List of matching products, never empty for TIER3
    """
    if runtime.embed_model is None or runtime.catalog_vecs is None:
        raise RuntimeError("Catalog runtime is not initialized")

    limit = k or settings.top_k
    qvec = runtime.embed_model.encode([query], normalize_embeddings=True)[0]
    semantic_scores = runtime.catalog_vecs @ qvec
    lexical_scores = np.array([_lexical_boost(query, product)
                              for product in runtime.catalog], dtype=np.float32)
    scores = semantic_scores + lexical_scores

    # For TIER3 (ambiguous), always return top K items regardless of threshold
    if tier == QueryTier.TIER3_AMBIGUOUS:
        top_idx = np.argsort(scores)[::-1][:limit]
        hits = [runtime.catalog[i] for i in top_idx]
    else:
        # For TIER1 and TIER2, apply threshold
        valid_indices = np.where(
            semantic_scores >= settings.similarity_threshold)[0]
        if len(valid_indices) == 0:
            valid_indices = np.where(lexical_scores > 0)[0]
        if len(valid_indices) == 0:
            hits = []
        else:
            # For TIER2 (semantic/occasion), validate against original query to avoid irrelevant results
            if tier == QueryTier.TIER2_SEMANTIC and original_query:
                original_qvec = runtime.embed_model.encode(
                    [original_query], normalize_embeddings=True)[0]
                original_semantic_scores = runtime.catalog_vecs @ original_qvec

                # Filter results to only include items that are relevant to BOTH the enriched and original query
                # This prevents "car price" from matching jewelry just because it has "silver"
                min_original_threshold = max(
                    0.35, settings.similarity_threshold - 0.1)
                valid_indices = valid_indices[original_semantic_scores[valid_indices]
                                              >= min_original_threshold]

                if len(valid_indices) == 0:
                    hits = []
                else:
                    top_idx = valid_indices[np.argsort(
                        scores[valid_indices])[::-1][:limit]]
                    hits = [runtime.catalog[i] for i in top_idx]
            else:
                top_idx = valid_indices[np.argsort(
                    scores[valid_indices])[::-1][:limit]]
                hits = [runtime.catalog[i] for i in top_idx]

    # Apply price filter if constraints provided
    if constraints and constraints.get("max_price"):
        max_price = constraints["max_price"]
        filtered_hits = []
        for hit in hits:
            price_str = str(hit.get("price", "")).replace(",", "")
            try:
                price = float(price_str)
                if price <= max_price:
                    filtered_hits.append(hit)
            except ValueError:
                # If price not parseable, include it
                filtered_hits.append(hit)

        # If filter leaves fewer than 2 results, relax by 20%
        if len(filtered_hits) < 2 and max_price > 0:
            relaxed_price = max_price * 1.2
            filtered_hits = []
            for hit in hits:
                price_str = str(hit.get("price", "")).replace(",", "")
                try:
                    price = float(price_str)
                    if price <= relaxed_price:
                        filtered_hits.append(hit)
                except ValueError:
                    filtered_hits.append(hit)

        # Never return empty results - if still empty, return original hits
        if not filtered_hits:
            filtered_hits = hits

        hits = filtered_hits

    return hits


def build_context(hits: list[dict]) -> str:
    entries: list[str] = []
    for index, hit in enumerate(hits, start=1):
        entries.append(
            "\n".join(
                [
                    f"Item {index}",
                    f"Name: {hit.get('name', 'Product')}",
                    f"Price: {_format_price(hit.get('price'))}",
                    f"Description: {_format_description(hit.get('description'))}",
                ]
            )
        )
    return "\n\n".join(entries)


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
    mode = "exact_match" if exact_match else "suggestions"
    tier_name = tier.value if tier else "semantic"
    prompt = (
        "system\n"
        f"{SYSTEM_PROMPT}\n\n"
        "user\n"
        f"TIER: {tier_name}\n"
        f"MODE: {mode}\n"
        f"CUSTOMER_MESSAGE: {message}\n\n"
        f"INVENTORY:\n{context}\n\n"
        "assistant\n"
    )

    return {
        "model": settings.ollama_model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.2,
            "stop": ["\nuser\n", "\nsystem\n"],
            "num_predict": 220,
        },
    }


def _fallback_reason(hit: dict) -> str:
    description = _format_description(hit.get("description"))
    if description == "No description provided.":
        return "This is a lovely option from our silver jewelry collection."
    return description


def _build_fallback_answer(hits: list[dict], tier: QueryTier | None = None) -> str:
    if not hits:
        return NO_RESULTS_MESSAGE

    if tier == QueryTier.TIER3_AMBIGUOUS:
        lines = ["Here are 5 popular picks you can start with:"]
        lines.extend(_format_product_line(hit) for hit in hits[:5])
        question = get_clarifying_question(tier)
        if question:
            lines.append("")
            lines.append(question)
        return "\n".join(lines)

    if tier == QueryTier.TIER2_SEMANTIC:
        lines = ["These are lovely curated picks for your occasion:"]
        for hit in hits:
            lines.append(
                f"{_format_product_line(hit)} - {_fallback_reason(hit)}")
        lines.append(
            "If you want, I can narrow these down for a specific style or budget.")
        return "\n".join(lines)

    lines = ["These are beautiful matches from our collection:"]
    lines.extend(_format_product_line(hit) for hit in hits)
    lines.append(
        "If you'd like, I can also show matching pieces in a similar style.")
    return "\n".join(lines)


def _mentions_known_product(answer: str, hits: list[dict]) -> bool:
    normalized_answer = _normalize_text(answer)
    return any(_normalize_text(str(hit.get("name", ""))) in normalized_answer for hit in hits)


def validate_llm_answer(answer: str, hits: list[dict], exact_match: bool, tier: QueryTier | None = None) -> str:
    del exact_match
    if not hits:
        return NO_RESULTS_MESSAGE

    cleaned = re.sub(r"<[^>]+>", "", answer or "").strip()
    if not cleaned:
        return _build_fallback_answer(hits, tier=tier)

    lowered = cleaned.lower()
    banned_phrases = ["i couldn't find", "i could not find",
                      "no items", "no products", "sorry"]
    if any(phrase in lowered for phrase in banned_phrases):
        return _build_fallback_answer(hits, tier=tier)

    if "http://" in lowered or "https://" in lowered:
        return _build_fallback_answer(hits, tier=tier)

    if not _mentions_known_product(cleaned, hits):
        return _build_fallback_answer(hits, tier=tier)

    result = cleaned
    if tier == QueryTier.TIER3_AMBIGUOUS and should_add_clarifying_question(tier):
        question = get_clarifying_question(tier)
        if question.lower() not in lowered:
            result = f"{result}\n\n{question}"

    return result

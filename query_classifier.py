"""
3-Tier Query Classification System
Classifies user queries into: TIER 1 (Exact), TIER 2 (Semantic), TIER 3 (Ambiguous)
"""

from enum import Enum
import re

from dependencies import runtime

# Category keywords - TIER 1
TIER1_CATEGORIES = {
    "jhumka", "jhumkas",
    "bangle", "bangles", "churi", "churis", "kada", "kadas",
    "necklace", "necklaces", "mala", "malas",
    "ring", "rings",
    "earring", "earrings",
    "bracelet", "bracelets",
    "chain", "chains",
    "pendant", "pendants",
    "tikka", "tikkas",
    "payal", "payals", "anklet", "anklets",
}

# Occasion/context keywords - TIER 2
TIER2_OCCASIONS = {
    "wedding", "weddings", "bridal", "bride",
    "festive", "festival",
    "party", "parties",
    "daily", "everyday", "casual",
    "gift", "gifts",
    "traditional", "ethnic",
}

# Ambiguous/vague keywords - TIER 3
TIER3_AMBIGUOUS = {
    "set", "sets", "collection", "collections",
    "something", "anything",
    "jewellery", "jewelry", "jewel", "jewels",
    "item", "items", "product", "products",
    "design", "designs",
}

PRODUCT_INTENT_TERMS = TIER1_CATEGORIES | TIER2_OCCASIONS | TIER3_AMBIGUOUS | {
    "silver", "price", "cost", "rs", "under", "show", "looking", "want", "need",
    "buy", "wear", "bridal", "daily", "latest", "new", "matching", "pearl", "drop",
}

OFF_TOPIC_PHRASES = {
    "what is your name",
    "who are you",
    "tell me your name",
    "lol",
    "haha",
    "hahaha",
    "ok",
    "okay",
    "hi",
    "hello",
    "hey",
}


class QueryTier(Enum):
    TIER1_EXACT = "exact"
    TIER2_SEMANTIC = "semantic"
    TIER3_AMBIGUOUS = "ambiguous"


def _tokenize(query: str) -> set[str]:
    """Normalize and tokenize query into lowercase words."""
    return set(word.lower() for word in query.split())


def _matches_catalog_tokens(tokens: set[str]) -> bool:
    if not tokens or not runtime.catalog:
        return False

    for product in runtime.catalog:
        name_tokens = _tokenize(str(product.get("name", "")).lower())
        if tokens & name_tokens:
            return True

    return False


def is_product_query(query: str) -> bool:
    normalized = re.sub(r"\s+", " ", query.strip().lower())
    if not normalized:
        return False

    if normalized in OFF_TOPIC_PHRASES:
        return False

    tokens = _tokenize(normalized)
    if tokens & PRODUCT_INTENT_TERMS:
        return True

    if _matches_catalog_tokens(tokens):
        return True

    if any(char.isdigit() for char in normalized):
        return True

    return len(tokens) >= 2 and any(token in {"for", "with", "under", "show"} for token in tokens)


def classify_query(query: str) -> tuple[QueryTier, str | None]:
    """
    Classify query into 3 tiers and optionally return enriched query.

    Returns:
        tuple of (QueryTier, enriched_query_or_none)
        - TIER1: (QueryTier.TIER1_EXACT, None) - no enrichment needed
        - TIER2: (QueryTier.TIER2_SEMANTIC, enriched_query) - enriched with occasion
        - TIER3: (QueryTier.TIER3_AMBIGUOUS, enriched_query) - enriched to specific phrase
    """
    tokens = _tokenize(query)

    # Check TIER 1 - Exact category match
    if tokens & TIER1_CATEGORIES:
        return QueryTier.TIER1_EXACT, None

    # Check TIER 2 - Occasion/semantic keywords
    if tokens & TIER2_OCCASIONS:
        occasion = list(tokens & TIER2_OCCASIONS)[0]
        enriched = f"{query} silver jewelry {occasion}"
        return QueryTier.TIER2_SEMANTIC, enriched

    # Check TIER 3 - Ambiguous/vague keywords
    if tokens & TIER3_AMBIGUOUS:
        specific_tokens = tokens - TIER3_AMBIGUOUS
        if specific_tokens:
            enriched = f"{query} silver jewelry"
        else:
            enriched = "popular silver jewelry sets traditional designs"
        return QueryTier.TIER3_AMBIGUOUS, enriched

    # Default to TIER 2 for unknown queries (treat as semantic)
    enriched = f"{query} silver jewelry"
    return QueryTier.TIER2_SEMANTIC, enriched


def should_add_clarifying_question(tier: QueryTier) -> bool:
    """Determine if we should add a clarifying question to the response."""
    return tier == QueryTier.TIER3_AMBIGUOUS


def get_clarifying_question(tier: QueryTier) -> str:
    """Get appropriate clarifying question based on tier."""
    if tier == QueryTier.TIER3_AMBIGUOUS:
        return "Are you looking for a bridal set or everyday wear?"
    return ""

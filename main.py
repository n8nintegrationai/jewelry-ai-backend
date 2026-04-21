import httpx

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded

from app_config import settings
from app_constants import NO_RESULTS_MESSAGE, OFF_TOPIC_MESSAGE
from dependencies import create_limiter, lifespan
from query_classifier import classify_query, is_product_query
from schemas import Query
from services import (
    build_context,
    build_llm_payload,
    build_sources,
    cosine_search,
    find_exact_name_matches,
    validate_llm_answer,
)

limiter: Limiter = create_limiter()

app = FastAPI(title="Silver Jewelry AI", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_allow_origins,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)


@app.post("/chat")
@limiter.limit(settings.chat_rate_limit)
async def chat(request: Request, query: Query):
    if not is_product_query(query.message):
        return {"answer": OFF_TOPIC_MESSAGE, "sources": []}

    # Classify query into 3 tiers
    tier, enriched_query = classify_query(query.message)
    search_query = enriched_query if enriched_query else query.message

    # Try exact name match first
    exact_hits = find_exact_name_matches(query.message)

    # If no exact match, use cosine search with tier-aware logic
    hits = exact_hits or cosine_search(search_query, tier=tier)

    unique_sources = build_sources(hits)

    # For TIER3 (ambiguous), we should never return empty
    # cosine_search ensures this, but as safety check:
    if not hits:
        return {"answer": NO_RESULTS_MESSAGE, "sources": []}

    context = build_context(hits)
    if not context:
        raise HTTPException(500, "Unable to format product results")

    payload = build_llm_payload(
        query.message, context, exact_match=bool(exact_hits), tier=tier)

    async with httpx.AsyncClient(timeout=settings.llm_timeout_seconds) as client:
        resp = await client.post(settings.ollama_url, json=payload)
        if resp.status_code != 200:
            raise HTTPException(502, "LLM unavailable")

    result = resp.json()
    answer = validate_llm_answer(result.get(
        "response", ""), hits, exact_match=bool(exact_hits), tier=tier)
    return {"answer": answer, "sources": unique_sources}

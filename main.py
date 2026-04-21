import json

import httpx

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded

from app_config import settings
from app_constants import NO_RESULTS_MESSAGE, OFF_TOPIC_MESSAGE
from dependencies import create_limiter, lifespan
from query_classifier import QueryTier, classify_query, is_product_query
from schemas import Query
from services import (
    build_context,
    build_llm_payload,
    build_sources,
    cosine_search,
    find_exact_name_matches,
    validate_llm_answer,
)
from session_store import session_store

limiter: Limiter = create_limiter()

app = FastAPI(title="Silver Jewelry AI", lifespan=lifespan)

# Configure CORS - allow all origins in dev, specific in prod
cors_origins = settings.cors_allow_origins if settings.cors_allow_origins else [
    "*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)


def _sse_event(event_type: str, data: dict) -> str:
    return f"event: {event_type}\ndata: {json.dumps(data)}\n\n"


async def _stream_ollama_response(payload: dict, sources: list[dict]):
    async with httpx.AsyncClient(timeout=settings.llm_timeout_seconds) as client:
        async with client.stream(
            "POST",
            settings.ollama_url,
            json={**payload, "stream": True},
        ) as resp:
            if resp.status_code != 200:
                yield _sse_event("error", {"message": "LLM unavailable"})
                return

            async for line in resp.aiter_lines():
                if not line:
                    continue

                try:
                    chunk = json.loads(line)
                except json.JSONDecodeError:
                    continue

                text = chunk.get("response", "")
                if text:
                    yield _sse_event("delta", {"text": text})

                if chunk.get("done"):
                    break

    yield _sse_event("sources", {"sources": sources})
    yield _sse_event("done", {"ok": True})


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
    hits = exact_hits or cosine_search(
        search_query,
        k=5 if tier == QueryTier.TIER3_AMBIGUOUS else None,
        tier=tier,
        original_query=query.message,
    )

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

    if query.stream:
        return StreamingResponse(
            _stream_ollama_response(payload, unique_sources),
            media_type="text/event-stream",
        )

    async with httpx.AsyncClient(timeout=settings.llm_timeout_seconds) as client:
        resp = await client.post(settings.ollama_url, json=payload)
        if resp.status_code != 200:
            raise HTTPException(502, "LLM unavailable")

    result = resp.json()
    answer = validate_llm_answer(result.get(
        "response", ""), hits, exact_match=bool(exact_hits), tier=tier)
    return {"answer": answer, "sources": unique_sources}

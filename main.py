import json
import logging
import re
import time

import httpx

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded

from app_config import settings
from app_constants import NO_RESULTS_MESSAGE, OFF_TOPIC_MESSAGE
from dependencies import create_limiter, lifespan, runtime
from query_classifier import QueryTier, classify_query, is_product_query
from schemas import Query
from services import (
    build_context,
    build_llm_payload,
    build_sources,
    cosine_search,
    extract_constraints,
    find_exact_name_matches,
    validate_llm_answer,
)
from session_store import session_store

# Configure logging for timing measurements
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

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


def _sse_event(event_name: str, data: dict) -> str:
    return f"event: {event_name}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


async def _stream_static_response(answer: str, sources: list[dict]):
    yield _sse_event("delta", {"text": answer})
    yield _sse_event("sources", {"sources": sources})
    yield _sse_event("done", {"ok": True})


def _count_tokens(text: str) -> int:
    """Simple word-based token counting."""
    return len(text.split())


def _extract_previous_category(chat_history: list[dict]) -> str | None:
    """
    Extract the category keyword from the previous user or assistant message.
    Returns the category (e.g., 'jhumka', 'bangle') if found, None otherwise.
    """
    if not chat_history:
        return None

    # Look through history from most recent backwards
    for i in range(len(chat_history) - 1, -1, -1):
        msg = chat_history[i]
        content = msg.get("content", "").lower()

        # Check for category keywords (from query_classifier.py TIER1_CATEGORIES)
        categories = {
            "jhumka", "jhumkas", "bangle", "bangles", "churi", "churis",
            "kada", "kadas", "necklace", "necklaces", "mala", "malas",
            "ring", "rings", "earring", "earrings", "bracelet", "bracelets",
            "chain", "chains", "pendant", "pendants", "tikka", "tikkas",
            "payal", "payals", "anklet", "anklets",
        }

        for category in categories:
            if category in content:
                return category

    return None


def _get_top_similarity_score(query: str) -> float:
    """
    Calculate the maximum cosine similarity score for a query against catalog.
    Returns the score of the best matching item.
    """
    if runtime.embed_model is None or runtime.catalog_vecs is None:
        return 0.0

    import numpy as np
    qvec = runtime.embed_model.encode([query], normalize_embeddings=True)[0]
    semantic_scores = runtime.catalog_vecs @ qvec
    if len(semantic_scores) > 0:
        return float(np.max(semantic_scores))
    return 0.0


async def _stream_ollama_response(payload: dict, sources: list[dict]):
    start_request_time = time.perf_counter()
    first_chunk_time = None
    response_text = ""

    async with httpx.AsyncClient(timeout=settings.llm_timeout_seconds) as client:
        try:
            try:
                async with client.stream(
                    "POST",
                    settings.ollama_url,
                    json={**payload, "stream": True},
                ) as resp:
                    if resp.status_code != 200:
                        logger.error(
                            f"[TIMING] ollama_error: status {resp.status_code}")
                        yield _sse_event("error", {"message": "LLM unavailable"})
                        return

                    async for line in resp.aiter_lines():
                        if not line:
                            continue

                        # Record time to first token
                        if first_chunk_time is None:
                            first_chunk_time = time.perf_counter() - start_request_time
                            logger.info(
                                f"[TIMING] ollama_first_token: {first_chunk_time*1000:.0f}ms")

                        try:
                            chunk = json.loads(line)
                        except json.JSONDecodeError:
                            continue

                        text = chunk.get("response", "")
                        if text:
                            response_text += text
                            yield _sse_event("delta", {"text": text})

                        if chunk.get("done"):
                            break
            finally:
                pass
        except httpx.TimeoutException:
            logger.error("[TIMING] ollama_timeout")
            yield _sse_event("error", {"message": "LLM timeout"})
            return
        except httpx.ConnectError:
            logger.error("[TIMING] ollama_connection_error")
            yield _sse_event("error", {"message": "LLM unavailable"})
            return
        except Exception as e:
            logger.error(f"[TIMING] streaming_error: {str(e)}")
            yield _sse_event("error", {"message": "Streaming error"})
            return

    # Calculate total streaming time and token count
    total_stream_time = time.perf_counter() - start_request_time
    token_count = _count_tokens(response_text)
    if token_count > 0:
        tokens_per_sec = token_count / total_stream_time if total_stream_time > 0 else 0
        logger.info(f"[TIMING] ollama_total: {total_stream_time*1000:.0f}ms")
        logger.info(
            f"[TIMING] response_tokens: {token_count} words in {total_stream_time*1000:.0f}ms = {tokens_per_sec:.1f} words/sec")
    else:
        logger.info(f"[TIMING] ollama_total: {total_stream_time*1000:.0f}ms")

    yield _sse_event("sources", {"sources": sources})
    yield _sse_event("done", {"ok": True})


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "model": settings.ollama_model,
        "vector_store_loaded": runtime.catalog_vecs is not None and len(runtime.catalog) > 0,
    }


@app.post("/chat")
@limiter.limit(settings.chat_rate_limit)
async def chat(request: Request, query: Query):
    total_start_time = time.perf_counter()

    if not is_product_query(query.message):
        return StreamingResponse(
            _stream_static_response(OFF_TOPIC_MESSAGE, []),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
            },
        )

    # TIMING 1: Query classification
    t_start = time.perf_counter()
    tier, enriched_query = classify_query(query.message)
    t_classify = (time.perf_counter() - t_start) * 1000
    logger.info(f"[TIMING] query_classification: {t_classify:.0f}ms")

    search_query = enriched_query if enriched_query else query.message

    # TIMING 2: Exact name match
    t_start = time.perf_counter()
    exact_hits = find_exact_name_matches(query.message)
    t_exact = (time.perf_counter() - t_start) * 1000
    logger.info(f"[TIMING] exact_name_match: {t_exact:.0f}ms")

    # Extract constraints to check for follow-up intent
    constraints = extract_constraints(query.message)
    is_followup = constraints.get("is_followup", False)

    # TIMING 3: Cosine search with vector search + price filtering
    t_start = time.perf_counter()
    hits = exact_hits or cosine_search(
        search_query,
        k=3 if tier == QueryTier.TIER3_AMBIGUOUS else None,
        tier=tier,
        original_query=query.message,
        constraints=constraints,
    )
    t_vector_search = (time.perf_counter() - t_start) * 1000
    logger.info(f"[TIMING] vector_search: {t_vector_search:.0f}ms")

    # TIMING 3b: Context-aware fallback for low-confidence follow-up queries
    if is_followup and not exact_hits and hits:
        t_start_fallback = time.perf_counter()
        top_score = _get_top_similarity_score(search_query)

        # If low confidence and we have session history with a previous category
        if top_score < 0.3:
            previous_category = _extract_previous_category(query.chat_history)
            if previous_category:
                logger.info(
                    f"[FALLBACK] low confidence query (score={top_score:.2f}), using previous category: {previous_category}")

                # Re-run vector search with the previous category
                fallback_hits = cosine_search(
                    previous_category,
                    k=5 if tier == QueryTier.TIER3_AMBIGUOUS else None,
                    tier=tier,
                    original_query=previous_category,
                    constraints=constraints,
                )
                if fallback_hits:
                    hits = fallback_hits

        t_fallback = (time.perf_counter() - t_start_fallback) * 1000
        logger.info(f"[TIMING] fallback_check: {t_fallback:.0f}ms")

    # TIMING 4: Build sources
    t_start = time.perf_counter()
    unique_sources = build_sources(hits)
    t_sources = (time.perf_counter() - t_start) * 1000
    logger.info(f"[TIMING] build_sources: {t_sources:.0f}ms")

    # For TIER3 (ambiguous), we should never return empty
    # cosine_search ensures this, but as safety check:
    if not hits:
        return StreamingResponse(
            _stream_static_response(NO_RESULTS_MESSAGE, []),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
            },
        )

    # TIMING 5: Build context
    t_start = time.perf_counter()
    context = build_context(hits)
    t_context = (time.perf_counter() - t_start) * 1000
    logger.info(f"[TIMING] prompt_build: {t_context:.0f}ms")

    if not context:
        raise HTTPException(500, "Unable to format product results")

    # TIMING 6: Build LLM payload
    t_start = time.perf_counter()
    payload = build_llm_payload(
        query.message, context, exact_match=bool(exact_hits), tier=tier)
    t_payload = (time.perf_counter() - t_start) * 1000
    logger.info(f"[TIMING] build_llm_payload: {t_payload:.0f}ms")

    logger.info(f"[TIMING] total_request_pre_stream: {(time.perf_counter() - total_start_time) * 1000:.0f}ms")

    return StreamingResponse(
        _stream_ollama_response(payload, unique_sources),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )

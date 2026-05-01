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
    apply_price_filter,
    build_context,
    build_price_filter_no_results_message,
    build_llm_payload,
    build_sources,
    cosine_search,
    extract_constraints,
    find_exact_name_matches,
    get_previous_context,
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

BASE_RULES = (
    "You are Luvz Style Assistant for a silver jewelry store in India. "
    "Use only INVENTORY products. Never invent names, prices, facts. "
    "No links, URLs, markdown, emojis. Plain text only. "
    "Short, warm shopkeeper sentences. Ships across India. Returns via WhatsApp."
)

TIER_INSTRUCTIONS = {
    "tier1": (
        "Customer wants a specific item or has a budget. "
        "Present matching item name and price naturally. "
        "End with exactly one of: "
        "\"Would you like to filter by budget?\" "
        "\"Shall I show you more [category] options?\""
    ),
    "tier2": (
        "Customer has occasion or style in mind. "
        "Present items as curated picks, one sentence per item explaining why it fits using only the item description. "
        "End with: \"Would you like to filter these by budget or occasion?\""
    ),
    "tier3": (
        "Vague request. "
        "Show top 2 items from INVENTORY with name and price. "
        "Ask exactly ONE clarifying question about budget or occasion. "
        "Never say \"I couldn't find\" or any variation."
    ),
}


def _sse_event(event_name: str, data: dict) -> str:
    return f"event: {event_name}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


async def _stream_static_response(answer: str, sources: list[dict]):
    yield _sse_event("delta", {"text": answer})
    yield _sse_event("sources", {"sources": sources})
    yield _sse_event("done", {"ok": True})


def _count_tokens(text: str) -> int:
    """Simple word-based token counting."""
    return len(text.split())


def _resolve_session_history(query: Query) -> list[dict]:
    request_history = query.chat_history or []
    stored_history = session_store.get_history(query.session_id)
    return request_history or stored_history


def _format_history_fallback_answer(hits: list[dict]) -> str:
    lines = ["Here are more options similar to what we were looking at:"]
    for hit in hits:
        name = str(hit.get("name", "Product")).strip() or "Product"
        price = str(hit.get("price", "")).strip()
        lines.append(f"{name} ({price})" if price else name)
    return "\n".join(lines)


def _budget_context_prompt() -> str:
    return "What type of jewelry are you looking for within that budget?"


def build_system_prompt(tier: str, context: str) -> str:
    # Builds the final system prompt from shared rules, tier guidance, and inventory.
    """Build the grounded system prompt for the selected query tier."""
    return f"{BASE_RULES}\n\n{TIER_INSTRUCTIONS[tier]}\n\nINVENTORY:\n{context}"


def classify_tier(query: str, hits: list[dict]) -> str:
    # Chooses the response tier from exact product names, budgets, and vague terms.
    """Classify the customer query into the prompt tier used for response style."""
    query_lower = query.lower()
    if any(
        (name := str(hit.get("name", "")).strip().lower()) and name in query_lower
        for hit in hits
    ):
        return "tier1"
    if re.search(r"\d", query):
        return "tier1"

    vague_terms = [
        "something",
        "anything",
        "gift",
        "suggest",
        "idea",
        "help",
        "nice",
        "good",
        "popular",
        "best",
        "show me",
    ]
    if len(query.split()) <= 2 or any(term in query_lower for term in vague_terms):
        return "tier3"

    return "tier2"


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


def _llm_timeout() -> httpx.Timeout:
    return httpx.Timeout(
        connect=settings.llm_connect_timeout_seconds,
        read=settings.llm_read_timeout_seconds,
        write=settings.llm_write_timeout_seconds,
        pool=settings.llm_pool_timeout_seconds,
    )


async def _stream_ollama_response(payload: dict, sources: list[dict]):
    start_request_time = time.perf_counter()
    first_chunk_time = None
    response_text = ""

    async with httpx.AsyncClient(timeout=_llm_timeout()) as client:
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
    session_history = _resolve_session_history(query)
    previous_assistant_message = ""
    for history_message in reversed(session_history):
        if str(history_message.get("role", "")).lower() == "assistant":
            previous_assistant_message = str(history_message.get("content", ""))
            break

    constraints = extract_constraints(query.message, previous_assistant_message)
    is_followup = constraints.get("is_followup", False)
    is_price_filter = constraints.get("is_price_filter", False)
    previous_context = (
        get_previous_context(session_history)
        if session_history else {
            "last_category": None,
            "last_query": None,
            "last_items_shown": [],
            "last_max_price": None,
        }
    )
    history_category = previous_context.get("last_category")
    history_query = previous_context.get("last_query") or history_category
    active_max_price = constraints.get("max_price")
    if active_max_price is None and is_followup:
        active_max_price = previous_context.get("last_max_price")
        if active_max_price is not None:
            constraints["max_price"] = active_max_price

    if not is_product_query(query.message):
        if session_history:
            if history_query:
                logger.info(
                    f"[FALLBACK] using session history context: {history_query}")
                fallback_tier, fallback_enriched_query = classify_query(history_query)
                fallback_search_query = fallback_enriched_query or history_query
                fallback_hits = find_exact_name_matches(history_query) or cosine_search(
                    fallback_search_query,
                    tier=fallback_tier,
                    original_query=history_query,
                    constraints=constraints,
                )
                if fallback_hits:
                    fallback_sources = build_sources(fallback_hits)
                    return StreamingResponse(
                        _stream_static_response(
                            _format_history_fallback_answer(fallback_hits),
                            fallback_sources,
                        ),
                        media_type="text/event-stream",
                        headers={
                            "Cache-Control": "no-cache",
                            "X-Accel-Buffering": "no",
                        },
                    )

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
    search_seed = query.message
    top_k_override = None
    needs_history_context = bool(active_max_price is not None or is_followup)
    if needs_history_context and not history_category:
        return StreamingResponse(
            _stream_static_response(_budget_context_prompt(), []),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
            },
        )

    if (is_price_filter or is_followup) and history_category:
        if active_max_price is not None:
            search_seed = f"{history_category} under {active_max_price}"
            logger.info(
                f"[PRICE_FILTER] applying max_price:{active_max_price} to previous category:{history_category}")
            top_k_override = settings.top_k + 6
        else:
            search_seed = history_query or history_category
            logger.info(
                f"[FOLLOWUP] detected, searching with previous context: {search_seed}")
            top_k_override = settings.top_k + 3
    elif active_max_price is not None:
        top_k_override = settings.top_k + 6

    tier, enriched_query = classify_query(search_seed)
    if top_k_override is None and tier == QueryTier.TIER3_AMBIGUOUS:
        top_k_override = 2
    t_classify = (time.perf_counter() - t_start) * 1000
    logger.info(f"[TIMING] query_classification: {t_classify:.0f}ms")

    search_query = enriched_query if enriched_query else search_seed

    # TIMING 2: Exact name match
    t_start = time.perf_counter()
    exact_hits = find_exact_name_matches(search_seed)
    t_exact = (time.perf_counter() - t_start) * 1000
    logger.info(f"[TIMING] exact_name_match: {t_exact:.0f}ms")

    # TIMING 3: Cosine search with vector search + price filtering
    t_start = time.perf_counter()
    hits = exact_hits or cosine_search(
        search_query,
        k=top_k_override,
        tier=tier,
        original_query=search_seed,
    )
    t_vector_search = (time.perf_counter() - t_start) * 1000
    logger.info(f"[TIMING] vector_search: {t_vector_search:.0f}ms")

    # TIMING 3b: Context-aware fallback for low-confidence follow-up queries
    if (is_followup or is_price_filter) and not exact_hits and hits:
        t_start_fallback = time.perf_counter()
        top_score = _get_top_similarity_score(search_query)

        # If low confidence and we have session history with a previous category
        if top_score < 0.3:
            previous_category = history_category
            if previous_category:
                logger.info(
                    f"[FALLBACK] low confidence query (score={top_score:.2f}), using previous category: {previous_category}")

                # Re-run vector search with the previous category
                fallback_query = previous_category
                if active_max_price is not None:
                    fallback_query = f"{previous_category} under {active_max_price}"
                fallback_hits = cosine_search(
                    fallback_query,
                    k=settings.top_k + 3,
                    tier=tier,
                    original_query=fallback_query,
                )
                if fallback_hits:
                    hits = fallback_hits

        t_fallback = (time.perf_counter() - t_start_fallback) * 1000
        logger.info(f"[TIMING] fallback_check: {t_fallback:.0f}ms")

    if active_max_price is not None and hits:
        t_start = time.perf_counter()
        price_filter_result = apply_price_filter(hits, active_max_price)
        t_price_filter = (time.perf_counter() - t_start) * 1000
        logger.info(f"[TIMING] price_filtering: {t_price_filter:.0f}ms")

        if price_filter_result["hits"]:
            hits = price_filter_result["hits"]
            if price_filter_result["relaxed"]:
                relaxed_price = price_filter_result["relaxed_price"]
                if relaxed_price is not None:
                    logger.info(
                        f"[PRICE_FILTER] no hits under {active_max_price}, relaxed to {relaxed_price:.0f}")
        else:
            category_label = history_category or search_seed
            closest_sources = build_sources(hits)
            answer = build_price_filter_no_results_message(
                category_label,
                active_max_price,
                price_filter_result["lowest_price"],
            )
            return StreamingResponse(
                _stream_static_response(answer, closest_sources),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "X-Accel-Buffering": "no",
                },
            )

    # TIMING 4: Build sources
    t_start = time.perf_counter()
    unique_sources = build_sources(hits)
    t_sources = (time.perf_counter() - t_start) * 1000
    logger.info(f"[TIMING] build_sources: {t_sources:.0f}ms")

    # For TIER3 (ambiguous), we should never return empty
    # cosine_search ensures this, but as safety check:
    if not hits:
        if active_max_price is not None and history_category:
            answer = build_price_filter_no_results_message(
                history_category,
                active_max_price,
                None,
            )
            return StreamingResponse(
                _stream_static_response(answer, []),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "X-Accel-Buffering": "no",
                },
            )
        if needs_history_context:
            return StreamingResponse(
                _stream_static_response(_budget_context_prompt(), []),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "X-Accel-Buffering": "no",
                },
            )
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
    response_tier = classify_tier(query.message, hits)
    context = build_context(hits)
    system_prompt = build_system_prompt(response_tier, context)
    t_context = (time.perf_counter() - t_start) * 1000
    logger.info(f"[TIMING] prompt_build: {t_context:.0f}ms")

    if not context:
        raise HTTPException(500, "Unable to format product results")

    # TIMING 6: Build LLM payload
    t_start = time.perf_counter()
    prompt = (
        "system\n"
        f"{system_prompt}\n\n"
        "user\n"
        f"CUSTOMER_MESSAGE: {query.message}\n\n"
        "assistant\n"
    )
    payload = {
        "model": "luvz-fast",
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0,
            "top_k": 1,
            "num_predict": 100,
            "num_ctx": 1024,
        },
    }
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

import httpx

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded

from app_config import settings
from app_constants import NO_RESULTS_MESSAGE
from dependencies import create_limiter, lifespan
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
    exact_hits = find_exact_name_matches(query.message)
    hits = exact_hits or cosine_search(query.message)
    unique_sources = build_sources(hits)
    if not hits:
        return {"answer": NO_RESULTS_MESSAGE, "sources": []}

    context = build_context(hits)
    if not context:
        raise HTTPException(500, "Unable to format product results")

    payload = build_llm_payload(query.message, context, exact_match=bool(exact_hits))

    async with httpx.AsyncClient(timeout=settings.llm_timeout_seconds) as client:
        resp = await client.post(settings.ollama_url, json=payload)
        if resp.status_code != 200:
            raise HTTPException(502, "LLM unavailable")

    result = resp.json()
    answer = validate_llm_answer(result.get("response", ""), hits, exact_match=bool(exact_hits))
    return {"answer": answer, "sources": unique_sources}

import json
import re
import httpx
import numpy as np
import torch
from pathlib import Path
from contextlib import asynccontextmanager  # New import
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, field_validator  # Updated import
from sentence_transformers import SentenceTransformer
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from urllib.parse import quote

# ── Config ────────────────────────────────────────────────
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "phi3"
TOP_K = 3
MAX_INPUT = 400   # characters
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Define your fallback number here (same as your JS constant)
DEFAULT_WHATSAPP = "919876543210"

# ── Rate limiter ──────────────────────────────────────────
limiter = Limiter(key_func=get_remote_address,
                  default_limits=["30/minute"])


# ── Startup: load model + embeddings ─────────────────────
# ── Global State ──
state = {
    "embed_model": None,
    "catalog": [],
    "catalog_vecs": None
}

# ── Modern Lifespan Handler ──


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup Logic
    state["embed_model"] = SentenceTransformer(MODEL_NAME, device=DEVICE)
    raw = json.loads(Path("embeddings.json").read_text())
    state["catalog"] = raw
    state["catalog_vecs"] = np.array(
        [item["embedding"] for item in raw], dtype=np.float32)
    print(f"[api] loaded {len(state['catalog'])} products, device={DEVICE}")

    yield  # The app runs while it "yields"

    # Shutdown Logic (Optional: clear memory)
    state.clear()

app = FastAPI(title="Silver Jewelry AI", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For local testing
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
# ── Input schema with sanitization ───────────────────────
ALLOWED_RE = re.compile(r"[^a-zA-Z0-9 \u00C0-\u024F.,?!'\-áéíóú]+")


class Query(BaseModel):
    message: str

    @field_validator("message")  # Modern Pydantic V2 style
    @classmethod
    def sanitize(cls, v):
        v = v.strip()
        if len(v) > MAX_INPUT:
            raise ValueError(f"Input exceeds {MAX_INPUT} characters")
        v = ALLOWED_RE.sub("", v)
        if not v:
            raise ValueError("Empty message after sanitization")
        return v

# ── RAG lookup ────────────────────────────────────────────


def cosine_search(query: str, k: int = TOP_K):
    qvec = state["embed_model"].encode([query], normalize_embeddings=True)[0]
    scores = state["catalog_vecs"] @ qvec

    # 0.40 is strict; it will block unrelated topics like cars or laptops
    THRESHOLD = 0.40

    valid_indices = np.where(scores >= THRESHOLD)[0]
    if len(valid_indices) == 0:
        return []

    top_idx = valid_indices[np.argsort(scores[valid_indices])[::-1][:k]]
    return [state["catalog"][i] for i in top_idx]


def build_context(hits: list[dict]) -> str:
    lines = []
    seen_names = set()
    # base_wa = "918919359961"  # From your Gemini code

    for h in hits:
        name = h.get("name", "Product")
        if name in seen_names:
            continue
        seen_names.add(name)

        # Create pre-filled message: "Hi I am interested in [Product Name]"
        encoded_msg = quote(f"Hi, I am interested in {name}")
        # wa_url = f"https://wa.me/{base_wa}?text={encoded_msg}"

        raw_price = str(h.get("price", "")).strip().lower()
        price_val = "Available on Request" if raw_price in [
            "", "0", "none", "consult"] else f"Rs. {raw_price}"

        # Format for the AI to easily copy
        # Strip out the SVG and HTML tags here. Just send a clean Markdown link.

        # lines.append(f"{name} - {price_val}. [Enquire on WhatsApp]({wa_url})")
        lines.append(f"{name} - {price_val}")

    return "\n".join(lines)


SYSTEM_PROMPT = """You are the Luvz Style Assistant. Provide a short, friendly response to the customer.

POLICIES:
- Shipping: Across India.
- Returns: Contact WhatsApp.

INSTRUCTIONS:
1. Suggest matching items from the inventory.
2. Format: [Product Name] - ₹[Price]
3. If the user asks for something not in the inventory, politely say we don't have it.
4. Provide only the answer. Do not add commentary about the prompt.
5. Do NOT include any links, URLs, or HTML tags. The system adds the WhatsApp button automatically.
"""

# ── Endpoint ──────────────────────────────────────────────


@app.post("/chat")
@limiter.limit("15/minute")
async def chat(request: Request, query: Query):
    # 1. Run the vector search
    hits = cosine_search(query.message)

    # ── Deduplicate Sources ──
    unique_sources = []
    seen_names = set()
    for h in hits:
        name = h.get("name")
        if name not in seen_names:
            unique_sources.append({"name": name, "url": h.get("url")})
            seen_names.add(name)

    # 2. IF NO HITS: Return a pre-written, safe response immediately
    if not hits:
        return {
            "answer": "I'm sorry, I couldn't find any items in our jewelry collection that match your request. I specialize in traditional silver jewelry like bangles and jhumkas. Would you like to see our latest designs?",
            "sources": []
        }

    # 3. IF HITS EXIST: Only now do we prepare the AI prompt
    context = build_context(hits)
    prompt = SYSTEM_PROMPT.format(context=context)

    payload = {
        "model": OLLAMA_MODEL,
        "prompt": f"INVENTORY:\n{context}\n\nCustomer: {query.message}\nAssistant:",
        "system": prompt,
        "stream": False,
        "options": {
            "temperature": 0.0,
            # <--- CRITICAL
            "stop": ["User:", "Assistant:", "INSTRUCTIONS:", "Inventory:", "User Request:", "\n\n\n", "---", "<a", "http"],
            "num_predict": 150
        },
    }

    async with httpx.AsyncClient(timeout=60) as client:
        resp = await client.post(OLLAMA_URL, json=payload)
        if resp.status_code != 200:
            raise HTTPException(502, "LLM unavailable")

    result = resp.json()
    return {
        "answer": result["response"],
        # "sources": [{"name": h["name"], "url": h.get("url")} for h in hits],
        "sources": unique_sources  # Now only unique items!
    }

import json
import httpx
import torch
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer

# 1. Configuration
PRODUCTS_URL = "https://raw.githubusercontent.com/n8nintegrationai/luvz-collection-dev/refs/heads/main/public/data/products.json"
EMBEDDINGS_FILE = Path("embeddings.json")
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def build_text(p: dict) -> str:
    """Concatenate fields for semantic search."""
    # We combine Name, Description, and Category to give the AI context
    parts = [
        str(p.get("name", "")),
        str(p.get("description", "")),
        str(p.get("category", ""))
    ]
    return " | ".join(filter(None, parts))


def main():
    print(f"[embed] Device detected: {DEVICE}")

    # 2. Fetch the Nested JSON
    resp = httpx.get(PRODUCTS_URL, timeout=30)
    resp.raise_for_status()
    full_data = resp.json()

    # 3. Flatten the Categories
    # Your JSON has keys like 'necklaces', 'earrings', etc.
    # We loop through every key that holds a list of products.
    all_products = []
    for key, value in full_data.items():
        if isinstance(value, list):
            # Add the category name to each product dict so the AI knows what it is
            for item in value:
                if isinstance(item, dict) and "id" in item:
                    item["category"] = key
                    all_products.append(item)

    print(
        f"[embed] Found {len(all_products)} total products across all categories.")

    # 4. Initialize Model
    model = SentenceTransformer(MODEL_NAME, device=DEVICE)

    # 5. Generate Embeddings
    texts = [build_text(p) for p in all_products]
    print(f"[embed] Encoding {len(texts)} items on GPU...")

    vectors = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=True,
        normalize_embeddings=True
    )

    # 6. Prepare Payload
    payload = []
    for i, p in enumerate(all_products):
        payload.append({
            "id": p.get("id"),
            "name": p.get("name"),
            "price": p.get("price"),
            "category": p.get("category"),
            "image": p.get("image"),  # Useful for displaying search results
            "embedding": vectors[i].tolist()
        })

    # 7. Save
    EMBEDDINGS_FILE.write_text(json.dumps(
        payload, ensure_ascii=False, indent=2))
    print(f"[embed] Success! Saved {len(payload)} items to {EMBEDDINGS_FILE}")


if __name__ == "__main__":
    main()

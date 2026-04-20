import json

import httpx
from sentence_transformers import SentenceTransformer

from app_config import settings


def build_text(product: dict) -> str:
    parts = [
        str(product.get("name", "")),
        str(product.get("description", "")),
        str(product.get("category", "")),
    ]
    return " | ".join(filter(None, parts))


def main():
    print(f"[embed] env={settings.app_env} device={settings.device}")

    resp = httpx.get(settings.products_url, timeout=30)
    resp.raise_for_status()
    full_data = resp.json()

    all_products = []
    for key, value in full_data.items():
        if isinstance(value, list):
            for item in value:
                if isinstance(item, dict) and "id" in item:
                    normalized = dict(item)
                    normalized["category"] = key
                    all_products.append(normalized)

    print(f"[embed] Found {len(all_products)} total products across all categories.")

    model = SentenceTransformer(settings.model_name, device=settings.device)
    texts = [build_text(product) for product in all_products]
    print(f"[embed] Encoding {len(texts)} items on {settings.device}...")

    vectors = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=True,
        normalize_embeddings=True,
    )

    payload = []
    for index, product in enumerate(all_products):
        payload.append(
            {
                "id": product.get("id"),
                "name": product.get("name"),
                "price": product.get("price"),
                "category": product.get("category"),
                "image": product.get("image"),
                "embedding": vectors[index].tolist(),
            }
        )

    settings.embeddings_file.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"[embed] Success! Saved {len(payload)} items to {settings.embeddings_file}")


if __name__ == "__main__":
    main()

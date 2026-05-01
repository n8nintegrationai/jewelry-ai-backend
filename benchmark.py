#!/usr/bin/env python3
"""
Benchmark script to measure performance of each layer in the jewelry AI backend.
Measures: catalog load, vector load, embedding generation, search, constraint extraction,
prompt building, and Gemma LLM response times.

Usage: python benchmark.py [--skip-llm]
  --skip-llm : Skip LLM benchmarks (Gemma first token and full response)
"""

import sys
import time
import json
import statistics
from typing import Callable, Any
from pathlib import Path

import httpx
import numpy as np
from sentence_transformers import SentenceTransformer

from app_config import settings
from dependencies import CatalogRuntime, runtime
from query_classifier import QueryTier, classify_query
import services
from services import (
    build_context,
    build_llm_payload,
    cosine_search,
    extract_constraints,
    find_exact_name_matches,
)


# Parse command-line arguments
SKIP_LLM = "--skip-llm" in sys.argv


# Helper functions for formatting (missing from services.py)
def _format_price(price: str | None) -> str:
    """Format price for display."""
    if not price:
        return "Price not available"
    return f"₹{price}"


def _format_description(description: str | None) -> str:
    """Format description for display."""
    if not description or not str(description).strip():
        return "No description provided."
    return str(description).strip()[:100] + "..." if len(str(description).strip()) > 100 else str(description).strip()


def _format_product_line(product: dict) -> str:
    """Format product name and price for display."""
    name = product.get("name", "Product")
    price = product.get("price", "N/A")
    return f"{name} - ₹{price}"


# Inject these functions into services module so build_context can use them
services._format_price = _format_price
services._format_description = _format_description
services._format_product_line = _format_product_line


class BenchmarkTimer:
    """Simple timer using perf_counter for accurate measurements."""

    def __init__(self, label: str = ""):
        self.label = label
        self.start_time = None
        self.elapsed_ms = None

    def __enter__(self):
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        end_time = time.perf_counter()
        self.elapsed_ms = (end_time - self.start_time) * 1000


def measure_layer(layer_name: str, func: Callable, *args, **kwargs) -> list[float]:
    """Run a function 3 times and return list of millisecond timings."""
    timings = []
    for _ in range(3):
        with BenchmarkTimer(layer_name) as timer:
            result = func(*args, **kwargs)
        timings.append(timer.elapsed_ms)
    return timings


def benchmark_catalog_load() -> list[float]:
    """Measure time to load JSON catalog file."""
    def load_catalog():
        embeddings_file = settings.embeddings_file
        if not embeddings_file.exists():
            raise FileNotFoundError(
                f"Embeddings file not found at {embeddings_file}. "
                "Run python embeddings_gpu.py first."
            )
        data = json.loads(embeddings_file.read_text(encoding="utf-8"))
        return data

    return measure_layer("Catalog load", load_catalog)


def benchmark_vector_load() -> list[float]:
    """Measure time to load vectors from catalog."""
    def load_vectors():
        embeddings_file = settings.embeddings_file
        data = json.loads(embeddings_file.read_text(encoding="utf-8"))
        vecs = np.array([item["embedding"] for item in data], dtype=np.float32)
        return vecs

    return measure_layer("Vector load", load_vectors)


def benchmark_embedding_generation(query: str) -> list[float]:
    """Measure time to generate embedding for a query."""
    def generate_embedding():
        if runtime.embed_model is None:
            raise RuntimeError("Embed model not initialized")
        qvec = runtime.embed_model.encode(
            [query], normalize_embeddings=True)[0]
        return qvec

    return measure_layer("Embedding generation", generate_embedding)


def benchmark_cosine_search(query: str, tier: QueryTier) -> list[float]:
    """Measure time for cosine similarity search."""
    def search():
        results = cosine_search(query, tier=tier, original_query=query)
        return results

    return measure_layer("Similarity search", search)


def benchmark_constraint_extraction(message: str) -> list[float]:
    """Measure time for constraint extraction."""
    def extract():
        constraints = extract_constraints(message)
        return constraints

    return measure_layer("Constraint extraction", extract)


def benchmark_prompt_build(query: str, hits: list[dict]) -> list[float]:
    """Measure time to build LLM prompt."""
    def build_prompt():
        context = build_context(hits)
        payload = build_llm_payload(
            query, context, exact_match=False, tier=QueryTier.TIER2_SEMANTIC
        )
        return payload

    return measure_layer("Prompt build", build_prompt)


def benchmark_gemma_first_token(payload: dict) -> list[float]:
    """Measure time to first token from Gemma (time to start responding)."""
    timings = []
    llm_timeout = 10.0  # Much shorter timeout for benchmarking

    for _ in range(3):
        try:
            with BenchmarkTimer("Gemma first token") as timer:
                with httpx.Client(timeout=llm_timeout) as client:
                    resp = client.post(settings.ollama_url, json=payload)
                    resp.raise_for_status()
                    _ = resp.json()
        except httpx.TimeoutException:
            print(f"  Warning: Gemma first token timeout ({llm_timeout}s)")
            timings.append(0)
            break  # Stop trying if it times out
        except Exception as e:
            print(f"  Warning: Gemma first token failed: {type(e).__name__}")
            timings.append(0)
            break  # Stop trying if it fails
        timings.append(timer.elapsed_ms)

    # Pad to 3 runs if we bailed early
    while len(timings) < 3:
        timings.append(0)
    return timings[:3]


def benchmark_gemma_full_response(payload: dict) -> list[float]:
    """Measure time for complete Gemma response."""
    timings = []
    llm_timeout = 10.0  # Much shorter timeout for benchmarking

    for _ in range(3):
        try:
            with BenchmarkTimer("Gemma full response") as timer:
                with httpx.Client(timeout=llm_timeout) as client:
                    resp = client.post(settings.ollama_url, json=payload)
                    resp.raise_for_status()
                    _ = resp.json()
        except httpx.TimeoutException:
            print(f"  Warning: Gemma full response timeout ({llm_timeout}s)")
            timings.append(0)
            break  # Stop trying if it times out
        except Exception as e:
            print(f"  Warning: Gemma full response failed: {type(e).__name__}")
            timings.append(0)
            break  # Stop trying if it fails
        timings.append(timer.elapsed_ms)

    # Pad to 3 runs if we bailed early
    while len(timings) < 3:
        timings.append(0)
    return timings[:3]


def benchmark_end_to_end(query: str) -> list[float]:
    """Measure complete request processing pipeline."""
    def full_pipeline():
        tier, enriched_query = classify_query(query)
        search_query = enriched_query if enriched_query else query

        exact_hits = find_exact_name_matches(query)
        hits = exact_hits or cosine_search(
            search_query,
            k=5 if tier == QueryTier.TIER3_AMBIGUOUS else None,
            tier=tier,
            original_query=query,
        )

        if not hits:
            return None

        context = build_context(hits)
        payload = build_llm_payload(
            query, context, exact_match=bool(exact_hits), tier=tier
        )

        # Skip LLM call if SKIP_LLM is enabled
        if not SKIP_LLM:
            try:
                with httpx.Client(timeout=10.0) as client:
                    resp = client.post(settings.ollama_url, json=payload)
                    resp.raise_for_status()
                    _ = resp.json()
            except Exception:
                pass

        return payload

    return measure_layer("End to end", full_pipeline)


def format_ms(ms: float) -> str:
    """Format milliseconds as human-readable string."""
    if ms < 1:
        return "<1ms"
    if ms < 1000:
        return f"{int(ms)}ms"
    return f"{ms/1000:.1f}s"


def get_status(ms_avg: float) -> str:
    """Determine status based on average time."""
    if ms_avg < 100:
        return "OK"
    elif ms_avg < 500:
        return "WARN"
    else:
        return "SLOW"


def run_benchmark():
    """Run complete benchmark suite."""
    print("\n" + "=" * 90)
    print("JEWELRY AI BACKEND - PERFORMANCE BENCHMARK")
    if SKIP_LLM:
        print("(LLM benchmarks SKIPPED)")
    print("=" * 90)

    # Initialize runtime
    print("\n[init] Loading runtime...")
    try:
        runtime.load()
        print(f"[init] Loaded {len(runtime.catalog)} products")
    except Exception as e:
        print(f"[init] ERROR: Failed to load runtime: {e}")
        return

    # Define test queries
    test_queries = [
        ("jhumka set", "Tier 1 exact"),
        ("something for wedding", "Tier 2 semantic"),
        ("set", "Tier 3 ambiguous"),
    ]

    results = {}

    # Benchmark each query
    for query, query_type in test_queries:
        print(f"\n{'='*90}")
        print(f"BENCHMARKING: {query!r} ({query_type})")
        print(f"{'='*90}")

        query_results = {}

        # 1. Catalog load (same for all queries)
        if "catalog_load" not in results:
            print("[1/9] Catalog load...", end=" ", flush=True)
            query_results["catalog_load"] = benchmark_catalog_load()
            results["catalog_load"] = query_results["catalog_load"]
            print("✓")
        else:
            query_results["catalog_load"] = results["catalog_load"]

        # 2. Vector load (same for all queries)
        if "vector_load" not in results:
            print("[2/9] Vector load...", end=" ", flush=True)
            query_results["vector_load"] = benchmark_vector_load()
            results["vector_load"] = query_results["vector_load"]
            print("✓")
        else:
            query_results["vector_load"] = results["vector_load"]

        # 3. Embedding generation
        print(
            f"[3/9] Embedding generation ({query!r})...", end=" ", flush=True)
        query_results["embedding_gen"] = benchmark_embedding_generation(query)
        print("✓")

        # 4. Cosine similarity search
        print(
            f"[4/9] Cosine similarity search ({query!r})...", end=" ", flush=True)
        tier, enriched_query = classify_query(query)
        search_query = enriched_query if enriched_query else query
        query_results["cosine_search"] = benchmark_cosine_search(
            search_query, tier)
        print("✓")

        # 5. Constraint extraction
        print(
            f"[5/9] Constraint extraction ({query!r})...", end=" ", flush=True)
        query_results["constraint_extract"] = benchmark_constraint_extraction(
            query)
        print("✓")

        # 6. Prompt build
        print(f"[6/9] Prompt build ({query!r})...", end=" ", flush=True)
        # Get search results for prompt building
        hits = cosine_search(
            search_query,
            k=5 if tier == QueryTier.TIER3_AMBIGUOUS else None,
            tier=tier,
            original_query=query,
        )
        if hits:
            query_results["prompt_build"] = benchmark_prompt_build(query, hits)
        else:
            query_results["prompt_build"] = [0, 0, 0]
        print("✓")

        # 7. Gemma first token
        if not SKIP_LLM:
            print(
                f"[7/9] Gemma first token ({query!r})...", end=" ", flush=True)
            context = build_context(hits) if hits else ""
            payload = build_llm_payload(
                query, context, exact_match=False, tier=tier)
            query_results["gemma_first_token"] = benchmark_gemma_first_token(
                payload)
            print("✓")
        else:
            print("[7/9] Gemma first token (SKIPPED)")
            query_results["gemma_first_token"] = [0, 0, 0]

        # 8. Gemma full response
        if not SKIP_LLM:
            print(
                f"[8/9] Gemma full response ({query!r})...", end=" ", flush=True)
            query_results["gemma_full_response"] = benchmark_gemma_full_response(
                payload)
            print("✓")
        else:
            print("[8/9] Gemma full response (SKIPPED)")
            query_results["gemma_full_response"] = [0, 0, 0]

        # 9. End-to-end
        print(f"[9/9] End-to-end ({query!r})...", end=" ", flush=True)
        query_results["end_to_end"] = benchmark_end_to_end(query)
        print("✓")

        # Print results for this query
        print(f"\n{'─'*90}")
        print(f"RESULTS FOR: {query!r}")
        print(f"{'─'*90}")
        print(
            f"{'Layer':<30} | {'Min':<10} | {'Avg':<10} | {'Max':<10} | {'Status':<8}"
        )
        print("-" * 90)

        for layer_name in [
            "catalog_load",
            "vector_load",
            "embedding_gen",
            "cosine_search",
            "constraint_extract",
            "prompt_build",
            "gemma_first_token",
            "gemma_full_response",
            "end_to_end",
        ]:
            if layer_name in query_results:
                timings = query_results[layer_name]
                min_ms = min(timings)
                max_ms = max(timings)
                avg_ms = statistics.mean(timings)
                status = get_status(avg_ms)

                # Format layer name for display
                display_name = layer_name.replace("_", " ").title()

                print(
                    f"{display_name:<30} | {format_ms(min_ms):<10} | {format_ms(avg_ms):<10} | {format_ms(max_ms):<10} | {status:<8}"
                )

        results[query] = query_results

    # Print overall analysis
    print(f"\n{'='*90}")
    print("ANALYSIS & RECOMMENDATIONS")
    print(f"{'='*90}")

    _print_diagnosis_and_recommendations(results)

    print(f"\n{'='*90}")
    print("Benchmark complete!")
    print(f"{'='*90}\n")

    runtime.clear()


def _print_diagnosis_and_recommendations(results: dict):
    """Analyze results and print diagnosis + recommendations."""

    # Identify bottlenecks from first query results
    first_query_results = None
    for key, value in results.items():
        if isinstance(value, dict) and "embedding_gen" in value:
            first_query_results = value
            break

    if not first_query_results:
        print("No results to analyze.")
        return

    bottlenecks = []

    # Analyze each layer
    layers_to_check = [
        ("embedding_gen", "Embedding generation"),
        ("cosine_search", "Cosine similarity search"),
        ("gemma_first_token", "Gemma first token"),
        ("gemma_full_response", "Gemma full response"),
        ("end_to_end", "End-to-end pipeline"),
    ]

    for layer_key, layer_name in layers_to_check:
        if layer_key in first_query_results:
            timings = first_query_results[layer_key]
            avg_ms = statistics.mean(timings)
            if avg_ms >= 500:
                bottlenecks.append((layer_name, avg_ms, "HIGH"))
            elif avg_ms >= 100:
                bottlenecks.append((layer_name, avg_ms, "MEDIUM"))

    if bottlenecks:
        print("\nDIAGNOSIS")
        print("-" * 90)
        for i, (layer, avg_ms, severity) in enumerate(bottlenecks, 1):
            print(f"Bottleneck {i}: {layer} ({format_ms(avg_ms)} avg)")
            if layer == "Embedding generation":
                print(
                    "  → Model inference is slow. Consider caching embeddings for common queries"
                )
                print("    or switching to a faster embedding model like DistilBERT.")
            elif layer == "Cosine similarity search":
                print(
                    "  → Vector search is slow. Consider using approximate nearest neighbor"
                )
                print("    search (FAISS/Annoy) or reducing catalog size.")
            elif "Gemma" in layer:
                print(
                    "  → LLM inference is slow. Model may be loading cold or running on slow hardware."
                )
                print(
                    "    Consider keeping model warm with periodic ping requests at startup.")
            elif "End-to-end" in layer:
                print(
                    "  → Overall pipeline is slow. Check individual layers for bottlenecks.")
            print()
    else:
        print("\nDIAGNOSIS")
        print("-" * 90)
        print("All layers performing well! No major bottlenecks detected.")
        print()

    print("\nRECOMMENDATIONS (Ranked by Impact)")
    print("-" * 90)

    recommendations = [
        (
            "HIGH IMPACT",
            "Cache vector file in memory at startup instead of loading from embeddings.json on every search",
        ),
        (
            "HIGH IMPACT",
            "Send a warmup request to Ollama at app startup so Gemma model is loaded before first real query",
        ),
        (
            "MEDIUM IMPACT",
            "Implement embeddings cache in Redis for top 100 most common queries to skip re-encoding",
        ),
        (
            "MEDIUM IMPACT",
            "Pre-compute semantic scores for top 50 products and cache them in memory",
        ),
        (
            "LOW IMPACT",
            "Reduce max_new_tokens in Gemma prompt if responses are consistently longer than needed",
        ),
        (
            "LOW IMPACT",
            "Use DistilBERT-based embedding model instead of all-MiniLM-L6-v2 for faster inference",
        ),
    ]

    for i, (impact, rec) in enumerate(recommendations, 1):
        print(f"{i}. {impact} — {rec}")

    print()


if __name__ == "__main__":
    if SKIP_LLM:
        print("[info] Running without LLM benchmarks (Gemma tests will be skipped)")
    run_benchmark()

#!/usr/bin/env python3
"""
Generate precision-recall curve at different thresholds.

This script is EFFICIENT:
1. Loads company embeddings from Neo4j ONCE
2. Pre-computes all context embeddings in ONE batch call
3. Evaluates thresholds with pure math (instant)

The goal: Find the threshold that gives us ≥95% precision
for edges we treat as "facts" in the graph.
"""

import csv
from collections import defaultdict
from pathlib import Path

import numpy as np
from dotenv import load_dotenv

load_dotenv()

from public_company_graph.cli import get_driver_and_database


def load_data(split: str) -> list[dict]:
    """Load a split file."""
    with open(f"data/er_{split}.csv") as f:
        return list(csv.DictReader(f))


def load_company_embeddings(driver, database: str) -> dict[str, tuple[list[float], str]]:
    """Load all company embeddings from Neo4j."""
    print("Loading company embeddings from Neo4j...")

    company_cache = {}
    with driver.session(database=database) as session:
        result = session.run(
            """
            MATCH (c:Company)
            WHERE c.description_embedding IS NOT NULL
              AND c.description IS NOT NULL
            RETURN c.ticker AS ticker,
                   c.description_embedding AS embedding,
                   c.description AS description
            """
        )

        for record in result:
            ticker = record["ticker"]
            embedding = record["embedding"]
            description = record["description"]

            if ticker and embedding:
                company_cache[ticker] = (embedding, description or "")

    print(f"Loaded {len(company_cache)} company embeddings")
    return company_cache


def get_context_embeddings(contexts: list[str], cache_file: Path) -> dict[str, list[float]]:
    """Get embeddings for all contexts (cached)."""
    import hashlib
    import json

    from openai import OpenAI

    client = OpenAI()

    # Load existing cache
    cache = {}
    if cache_file.exists():
        with open(cache_file) as f:
            cache = json.load(f)
        print(f"Loaded {len(cache)} cached context embeddings")

    # Find contexts that need embedding
    def hash_text(text: str) -> str:
        return hashlib.sha256(text.encode()).hexdigest()[:16]

    uncached = []
    uncached_keys = []

    for ctx in contexts:
        text = ctx[:500] if ctx else ""
        key = hash_text(text)
        if key not in cache:
            uncached.append(text)
            uncached_keys.append(key)

    if uncached:
        print(f"Computing embeddings for {len(uncached)} new contexts...")

        # Batch API call (up to 2048 texts per call)
        batch_size = 2048
        for i in range(0, len(uncached), batch_size):
            batch = uncached[i : i + batch_size]
            batch_keys = uncached_keys[i : i + batch_size]

            response = client.embeddings.create(
                model="text-embedding-3-small",
                input=batch,
            )

            for j, emb_data in enumerate(response.data):
                cache[batch_keys[j]] = emb_data.embedding

            print(f"  Processed {min(i + batch_size, len(uncached))}/{len(uncached)}")

        # Save cache
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_file, "w") as f:
            json.dump(cache, f)
        print(f"Saved {len(cache)} context embeddings to cache")

    return cache


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Calculate cosine similarity."""
    a_arr = np.array(a)
    b_arr = np.array(b)
    norm_product = np.linalg.norm(a_arr) * np.linalg.norm(b_arr)
    if norm_product == 0:
        return 0.0
    return float(np.dot(a_arr, b_arr) / norm_product)


def evaluate_at_threshold(
    records: list[dict],
    company_cache: dict[str, tuple[list[float], str]],
    context_cache: dict[str, list[float]],
    threshold: float,
) -> dict:
    """Evaluate at a specific embedding threshold (pure math, instant)."""
    import hashlib

    def hash_text(text: str) -> str:
        return hashlib.sha256(text.encode()).hexdigest()[:16]

    kept_correct = 0
    kept_incorrect = 0
    rejected_correct = 0
    rejected_incorrect = 0
    no_embedding = 0

    for r in records:
        ticker = r["target_ticker"]
        context = r.get("context", "")[:500]
        is_correct = r["ai_label"] == "correct"

        # Get embeddings
        if ticker not in company_cache:
            no_embedding += 1
            # No company embedding - default to accept
            if is_correct:
                kept_correct += 1
            else:
                kept_incorrect += 1
            continue

        company_emb, _ = company_cache[ticker]
        context_key = hash_text(context)

        if context_key not in context_cache:
            no_embedding += 1
            if is_correct:
                kept_correct += 1
            else:
                kept_incorrect += 1
            continue

        context_emb = context_cache[context_key]

        # Compute similarity (instant)
        similarity = cosine_similarity(context_emb, company_emb)

        # Apply threshold
        if similarity >= threshold:
            if is_correct:
                kept_correct += 1
            else:
                kept_incorrect += 1
        else:
            if is_correct:
                rejected_correct += 1
            else:
                rejected_incorrect += 1

    kept_total = kept_correct + kept_incorrect
    precision = kept_correct / kept_total if kept_total > 0 else 0

    total_correct = kept_correct + rejected_correct
    recall = kept_correct / total_correct if total_correct > 0 else 0

    return {
        "threshold": threshold,
        "precision": precision,
        "recall": recall,
        "kept": kept_total,
        "kept_correct": kept_correct,
        "kept_incorrect": kept_incorrect,
        "rejected_correct": rejected_correct,
        "no_embedding": no_embedding,
    }


def evaluate_by_relationship_type(
    records: list[dict],
    company_cache: dict[str, tuple[list[float], str]],
    context_cache: dict[str, list[float]],
    threshold: float,
) -> dict:
    """Evaluate precision by relationship type."""
    import hashlib

    def hash_text(text: str) -> str:
        return hashlib.sha256(text.encode()).hexdigest()[:16]

    by_type = defaultdict(lambda: {"correct": 0, "incorrect": 0})

    for r in records:
        ticker = r["target_ticker"]
        context = r.get("context", "")[:500]
        rel_type = r["relationship_type"]
        is_correct = r["ai_label"] == "correct"

        # Get embeddings
        if ticker not in company_cache:
            if is_correct:
                by_type[rel_type]["correct"] += 1
            else:
                by_type[rel_type]["incorrect"] += 1
            continue

        company_emb, _ = company_cache[ticker]
        context_key = hash_text(context)

        if context_key not in context_cache:
            if is_correct:
                by_type[rel_type]["correct"] += 1
            else:
                by_type[rel_type]["incorrect"] += 1
            continue

        context_emb = context_cache[context_key]
        similarity = cosine_similarity(context_emb, company_emb)

        if similarity >= threshold:
            if is_correct:
                by_type[rel_type]["correct"] += 1
            else:
                by_type[rel_type]["incorrect"] += 1

    results = {}
    for rel_type, counts in by_type.items():
        total = counts["correct"] + counts["incorrect"]
        precision = counts["correct"] / total if total > 0 else 0
        results[rel_type] = {
            "precision": precision,
            "kept": total,
            "correct": counts["correct"],
            "incorrect": counts["incorrect"],
        }

    return results


def main():
    # Get Neo4j connection
    driver, database = get_driver_and_database()

    # Load data
    train = load_data("train")
    validation = load_data("validation")
    all_data = train + validation

    print("=" * 70)
    print("PRECISION-RECALL CURVE (EFFICIENT VERSION)")
    print("=" * 70)
    print(f"Total records: {len(all_data)} (train + validation)")
    print()

    # Load company embeddings from Neo4j (ONCE)
    company_cache = load_company_embeddings(driver, database)

    # Pre-compute all context embeddings (ONE batch call)
    contexts = [r.get("context", "") for r in all_data]
    cache_file = Path("data/embedding_cache/context_embeddings.json")
    context_cache = get_context_embeddings(contexts, cache_file)

    print()
    print("=" * 70)
    print("EVALUATING THRESHOLDS (pure math, instant)")
    print("=" * 70)
    print()

    # Test thresholds from 0.20 to 0.60
    thresholds = [0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]

    print("Threshold | Precision | Recall  | Kept | Correct | Incorrect")
    print("-" * 70)

    for t in thresholds:
        result = evaluate_at_threshold(all_data, company_cache, context_cache, t)
        marker = " ← 95%+" if result["precision"] >= 0.95 else ""
        marker = " ← 90%+" if 0.90 <= result["precision"] < 0.95 else marker
        print(
            f"  {t:.2f}    |  {result['precision']:>6.1%}   | {result['recall']:>6.1%}  | "
            f"{result['kept']:>4} |  {result['kept_correct']:>4}   |    {result['kept_incorrect']:>3}{marker}"
        )

    print()
    print("=" * 70)
    print("PRECISION BY RELATIONSHIP TYPE (at threshold=0.30)")
    print("=" * 70)

    by_type = evaluate_by_relationship_type(all_data, company_cache, context_cache, 0.30)
    print()
    print("Relationship Type    | Precision | Kept | Correct | Incorrect")
    print("-" * 70)

    for rel_type in sorted(by_type.keys()):
        stats = by_type[rel_type]
        print(
            f"{rel_type:<20} |  {stats['precision']:>6.1%}   | {stats['kept']:>4} | "
            f"  {stats['correct']:>3}    |    {stats['incorrect']:>3}"
        )

    # Find the threshold for ~95% precision
    print()
    print("=" * 70)
    print("RECOMMENDATION")
    print("=" * 70)

    for t in sorted(thresholds, reverse=True):
        result = evaluate_at_threshold(all_data, company_cache, context_cache, t)
        if result["precision"] >= 0.90 and result["kept"] >= 50:
            print(f"""
At threshold {t:.2f}:
  - Precision: {result["precision"]:.1%} (only {result["kept_incorrect"]} wrong out of {result["kept"]})
  - Recall: {result["recall"]:.1%}
  - You'd keep {result["kept"]} edges as "facts"
  - {result["rejected_correct"]} valid edges would become "candidates"

TIERED APPROACH:
  HIGH CONFIDENCE (≥{t:.2f}):  Store as facts (HAS_COMPETITOR, etc.)
  MEDIUM (0.30-{t:.2f}):       Store as candidates (POSSIBLY_COMPETES_WITH)
  LOW (<0.30):                 Don't create edges, or store as :MENTIONS
""")
            break

    driver.close()


if __name__ == "__main__":
    main()

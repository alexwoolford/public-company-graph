#!/usr/bin/env python3
"""
Full evaluation of the layered entity validator.

This script evaluates:
1. Embedding similarity layer (using pre-computed embeddings)
2. Rule-based filters (biographical, exchange, corporate, platform)
3. Relationship type verifier

Reports precision/recall for each layer and combined.
"""

import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from dotenv import load_dotenv

load_dotenv()

from public_company_graph.cli import get_driver_and_database
from public_company_graph.entity_resolution.candidates import Candidate
from public_company_graph.entity_resolution.filters import (
    BiographicalContextFilter,
    CorporateStructureFilter,
    ExchangeReferenceFilter,
    PlatformDependencyFilter,
)
from public_company_graph.entity_resolution.relationship_verifier import (
    RelationshipVerifier,
    VerificationResult,
)


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


def load_context_embeddings(cache_file: Path) -> dict[str, list[float]]:
    """Load context embeddings from cache."""
    if cache_file.exists():
        with open(cache_file) as f:
            cache = json.load(f)
        print(f"Loaded {len(cache)} cached context embeddings")
        return cache
    return {}


def hash_text(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def cosine_similarity(a: list[float], b: list[float]) -> float:
    a_arr = np.array(a)
    b_arr = np.array(b)
    norm_product = np.linalg.norm(a_arr) * np.linalg.norm(b_arr)
    if norm_product == 0:
        return 0.0
    return float(np.dot(a_arr, b_arr) / norm_product)


def evaluate_layers(
    records: list[dict],
    company_cache: dict,
    context_cache: dict,
    embedding_threshold: float,
) -> dict:
    """Evaluate each layer's contribution to precision/recall."""
    # Initialize filters
    bio_filter = BiographicalContextFilter()
    exchange_filter = ExchangeReferenceFilter()
    corporate_filter = CorporateStructureFilter()
    platform_filter = PlatformDependencyFilter()
    relationship_verifier = RelationshipVerifier()

    # Track rejections by layer
    results = {
        "baseline": {"tp": 0, "fp": 0, "fn": 0, "tn": 0},
        "embedding": {"tp": 0, "fp": 0, "fn": 0, "tn": 0, "rejects": 0},
        "biographical": {"tp": 0, "fp": 0, "fn": 0, "tn": 0, "rejects": 0},
        "exchange": {"tp": 0, "fp": 0, "fn": 0, "tn": 0, "rejects": 0},
        "corporate": {"tp": 0, "fp": 0, "fn": 0, "tn": 0, "rejects": 0},
        "platform": {"tp": 0, "fp": 0, "fn": 0, "tn": 0, "rejects": 0},
        "rel_type": {"tp": 0, "fp": 0, "fn": 0, "tn": 0, "rejects": 0},
        "all_layers": {"tp": 0, "fp": 0, "fn": 0, "tn": 0},
    }

    for r in records:
        ticker = r["target_ticker"]
        context = r.get("context", "")[:500]
        rel_type = r["relationship_type"]
        mention = r.get("raw_mention", "")
        is_correct = r["ai_label"] == "correct"

        # Baseline
        if is_correct:
            results["baseline"]["tp"] += 1
        else:
            results["baseline"]["fp"] += 1

        # Track which layers would reject
        rejected_by = []

        # 1. Embedding check
        if ticker in company_cache:
            company_emb, _ = company_cache[ticker]
            context_key = hash_text(context)

            if context_key in context_cache:
                context_emb = context_cache[context_key]
                similarity = cosine_similarity(context_emb, company_emb)

                if similarity < embedding_threshold:
                    rejected_by.append("embedding")

        # 2. Biographical filter
        candidate = Candidate(
            text=mention,
            sentence=context,
            start_pos=0,
            end_pos=len(mention),
            source_pattern="extraction",
        )

        if not bio_filter.filter(candidate).passed:
            rejected_by.append("biographical")

        # 3. Exchange filter
        if not exchange_filter.filter(candidate).passed:
            rejected_by.append("exchange")

        # 4. Corporate filter
        if not corporate_filter.filter(candidate).passed:
            rejected_by.append("corporate")

        # 5. Platform filter
        if not platform_filter.filter(candidate).passed:
            rejected_by.append("platform")

        # 6. Relationship type verifier
        rel_result = relationship_verifier.verify(
            claimed_type=rel_type,
            context=context,
            mention=mention,
        )
        if rel_result.result == VerificationResult.CONTRADICTED:
            rejected_by.append("rel_type")

        # Track per-layer effectiveness
        for layer in ["embedding", "biographical", "exchange", "corporate", "platform", "rel_type"]:
            if layer in rejected_by:
                results[layer]["rejects"] += 1
                if is_correct:
                    # False negative - rejected a correct relationship
                    results[layer]["fn"] += 1
                else:
                    # True negative - correctly rejected
                    results[layer]["tn"] += 1
            else:
                if is_correct:
                    results[layer]["tp"] += 1
                else:
                    results[layer]["fp"] += 1

        # Combined (all layers)
        if rejected_by:
            if is_correct:
                results["all_layers"]["fn"] += 1
            else:
                results["all_layers"]["tn"] += 1
        else:
            if is_correct:
                results["all_layers"]["tp"] += 1
            else:
                results["all_layers"]["fp"] += 1

    return results


def calc_metrics(tp, fp, fn, tn):
    """Calculate precision, recall, F1."""
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    return precision, recall, f1


def evaluate_by_rel_type(
    records: list[dict],
    company_cache: dict,
    context_cache: dict,
    embedding_threshold: float,
) -> dict:
    """Evaluate performance by relationship type with full layered validation."""
    bio_filter = BiographicalContextFilter()
    exchange_filter = ExchangeReferenceFilter()
    corporate_filter = CorporateStructureFilter()
    platform_filter = PlatformDependencyFilter()
    relationship_verifier = RelationshipVerifier()

    by_type = defaultdict(
        lambda: {
            "baseline_tp": 0,
            "baseline_fp": 0,
            "layered_tp": 0,
            "layered_fp": 0,
            "layered_fn": 0,
        }
    )

    for r in records:
        ticker = r["target_ticker"]
        context = r.get("context", "")[:500]
        rel_type = r["relationship_type"]
        mention = r.get("raw_mention", "")
        is_correct = r["ai_label"] == "correct"

        # Baseline
        if is_correct:
            by_type[rel_type]["baseline_tp"] += 1
        else:
            by_type[rel_type]["baseline_fp"] += 1

        # Check all layers
        rejected = False

        # Embedding
        if ticker in company_cache:
            company_emb, _ = company_cache[ticker]
            context_key = hash_text(context)
            if context_key in context_cache:
                similarity = cosine_similarity(context_cache[context_key], company_emb)
                if similarity < embedding_threshold:
                    rejected = True

        # Filters
        candidate = Candidate(
            text=mention,
            sentence=context,
            start_pos=0,
            end_pos=len(mention),
            source_pattern="extraction",
        )
        if not rejected and not bio_filter.filter(candidate).passed:
            rejected = True
        if not rejected and not exchange_filter.filter(candidate).passed:
            rejected = True
        if not rejected and not corporate_filter.filter(candidate).passed:
            rejected = True
        if not rejected and not platform_filter.filter(candidate).passed:
            rejected = True

        # Relationship verifier
        if not rejected:
            rel_result = relationship_verifier.verify(rel_type, context, mention)
            if rel_result.result == VerificationResult.CONTRADICTED:
                rejected = True

        # Track result
        if rejected:
            if is_correct:
                by_type[rel_type]["layered_fn"] += 1
        else:
            if is_correct:
                by_type[rel_type]["layered_tp"] += 1
            else:
                by_type[rel_type]["layered_fp"] += 1

    return by_type


def main():
    # Get Neo4j connection
    driver, database = get_driver_and_database()

    # Load data
    train = load_data("train")
    validation = load_data("validation")
    all_data = train + validation

    print("=" * 70)
    print("FULL LAYERED VALIDATION EVALUATION")
    print("=" * 70)
    print(f"Total records: {len(all_data)} (train + validation)")
    print()

    # Load embeddings
    company_cache = load_company_embeddings(driver, database)
    context_cache = load_context_embeddings(Path("data/embedding_cache/context_embeddings.json"))

    print()
    print("=" * 70)
    print("LAYER-BY-LAYER ANALYSIS")
    print("=" * 70)

    results = evaluate_layers(all_data, company_cache, context_cache, 0.30)

    print()
    print("Layer            | Precision | Recall  |   F1   | Rejects | Impact")
    print("-" * 70)

    baseline_prec, baseline_rec, baseline_f1 = calc_metrics(
        results["baseline"]["tp"], results["baseline"]["fp"], 0, 0
    )
    print(
        f"Baseline         |  {baseline_prec:>6.1%}   | {baseline_rec:>6.1%}  | {baseline_f1:>5.1%}  |    -    | (no filtering)"
    )

    for layer in ["embedding", "biographical", "exchange", "corporate", "platform", "rel_type"]:
        prec, rec, f1 = calc_metrics(
            results[layer]["tp"], results[layer]["fp"], results[layer]["fn"], results[layer]["tn"]
        )
        rejects = results[layer]["rejects"]
        delta_p = prec - baseline_prec
        impact = f"+{delta_p:.1%}" if delta_p >= 0 else f"{delta_p:.1%}"
        print(
            f"{layer:<16} |  {prec:>6.1%}   | {rec:>6.1%}  | {f1:>5.1%}  |  {rejects:>4}   | {impact} prec"
        )

    all_prec, all_rec, all_f1 = calc_metrics(
        results["all_layers"]["tp"],
        results["all_layers"]["fp"],
        results["all_layers"]["fn"],
        results["all_layers"]["tn"],
    )
    delta_p = all_prec - baseline_prec
    print("-" * 70)
    print(
        f"ALL LAYERS       |  {all_prec:>6.1%}   | {all_rec:>6.1%}  | {all_f1:>5.1%}  |    -    | +{delta_p:.1%} prec"
    )

    print()
    print("=" * 70)
    print("PRECISION BY RELATIONSHIP TYPE (with all layers)")
    print("=" * 70)

    by_type = evaluate_by_rel_type(all_data, company_cache, context_cache, 0.30)

    print()
    print("Relationship     | Baseline Prec | Layered Prec | Improvement | Kept")
    print("-" * 70)

    for rel_type in sorted(by_type.keys()):
        stats = by_type[rel_type]
        baseline_total = stats["baseline_tp"] + stats["baseline_fp"]
        baseline_prec = stats["baseline_tp"] / baseline_total if baseline_total > 0 else 0

        layered_total = stats["layered_tp"] + stats["layered_fp"]
        layered_prec = stats["layered_tp"] / layered_total if layered_total > 0 else 0

        delta = layered_prec - baseline_prec
        sign = "+" if delta >= 0 else ""

        print(
            f"{rel_type:<16} |   {baseline_prec:>6.1%}      |   {layered_prec:>6.1%}     |  {sign}{delta:>5.1%}     | {layered_total}"
        )

    print()
    print("=" * 70)
    print("KEY INSIGHTS")
    print("=" * 70)
    print("""
The layered approach combines:
1. Embedding similarity - catches semantic mismatches
2. Biographical filter - catches career/director mentions
3. Exchange filter - catches stock exchange references
4. Corporate filter - catches parent/subsidiary mentions
5. Platform filter - catches app store/OS dependencies
6. Relationship verifier - catches wrong relationship types

Each layer catches different error types. The rel_type verifier is
particularly important for SUPPLIER/CUSTOMER where the extraction
might pick up competitor mentions from sentences containing "vendor".
""")

    driver.close()


if __name__ == "__main__":
    main()

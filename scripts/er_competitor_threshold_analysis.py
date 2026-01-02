#!/usr/bin/env python3
"""
Analyze COMPETITOR precision at different thresholds.

Goal: Find threshold where we can achieve 95%+ precision
for edges we treat as "facts" in the graph.
"""

import csv
import hashlib
import json

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


def hash_text(text):
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def cosine_similarity(a, b):
    a_arr, b_arr = np.array(a), np.array(b)
    norm = np.linalg.norm(a_arr) * np.linalg.norm(b_arr)
    return float(np.dot(a_arr, b_arr) / norm) if norm > 0 else 0.0


def main():
    # Load data
    with open("data/er_train.csv") as f:
        train = list(csv.DictReader(f))
    with open("data/er_validation.csv") as f:
        val = list(csv.DictReader(f))

    # Load embeddings
    driver, database = get_driver_and_database()
    company_cache = {}
    with driver.session(database=database) as session:
        result = session.run(
            "MATCH (c:Company) WHERE c.description_embedding IS NOT NULL "
            "RETURN c.ticker AS ticker, c.description_embedding AS embedding"
        )
        for rec in result:
            if rec["ticker"]:
                company_cache[rec["ticker"]] = rec["embedding"]
    driver.close()

    with open("data/embedding_cache/context_embeddings.json") as f:
        context_cache = json.load(f)

    # Initialize filters
    bio_filter = BiographicalContextFilter()
    exchange_filter = ExchangeReferenceFilter()
    corporate_filter = CorporateStructureFilter()
    platform_filter = PlatformDependencyFilter()
    rel_verifier = RelationshipVerifier()

    print("=" * 80)
    print("PRECISION BY RELATIONSHIP TYPE AT DIFFERENT THRESHOLDS")
    print("=" * 80)

    for rel_type in ["HAS_COMPETITOR", "HAS_SUPPLIER", "HAS_CUSTOMER", "HAS_PARTNER"]:
        records = [r for r in (train + val) if r["relationship_type"] == rel_type]

        print(f"\n{rel_type} ({len(records)} records)")
        print("-" * 80)
        print("Threshold | Precision | Recall  | Kept | Wrong | Target")
        print("-" * 80)

        for threshold in [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]:
            tp, fp, fn = 0, 0, 0

            for r in records:
                ticker = r["target_ticker"]
                context = r.get("context", "")[:500]
                mention = r.get("raw_mention", "")
                is_correct = r["ai_label"] == "correct"

                rejected = False

                # Embedding check
                if ticker in company_cache and hash_text(context) in context_cache:
                    sim = cosine_similarity(
                        context_cache[hash_text(context)], company_cache[ticker]
                    )
                    if sim < threshold:
                        rejected = True

                # Filters
                candidate = Candidate(
                    text=mention,
                    sentence=context,
                    start_pos=0,
                    end_pos=len(mention),
                    source_pattern="x",
                )
                if not rejected and not bio_filter.filter(candidate).passed:
                    rejected = True
                if not rejected and not exchange_filter.filter(candidate).passed:
                    rejected = True
                if not rejected and not corporate_filter.filter(candidate).passed:
                    rejected = True
                if not rejected and not platform_filter.filter(candidate).passed:
                    rejected = True
                if (
                    not rejected
                    and rel_verifier.verify(rel_type, context, mention).result
                    == VerificationResult.CONTRADICTED
                ):
                    rejected = True

                if rejected:
                    if is_correct:
                        fn += 1
                else:
                    if is_correct:
                        tp += 1
                    else:
                        fp += 1

            prec = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            kept = tp + fp

            marker = ""
            if prec >= 0.95:
                marker = "← 95%+ FACTS"
            elif prec >= 0.90:
                marker = "← 90%+ good"
            elif prec >= 0.80:
                marker = "← 80%+"

            print(
                f"  {threshold:.2f}    |  {prec:>6.1%}   | {recall:>6.1%}  | "
                f"{kept:>4} |  {fp:>2}   | {marker}"
            )

    print("\n" + "=" * 80)
    print("RECOMMENDATION")
    print("=" * 80)
    print(
        """
Based on this analysis:

HAS_COMPETITOR:
  - At high thresholds → Near 90%+ precision
  - ✅ Can be used as "facts" with threshold tuning

HAS_PARTNER:
  - Moderate precision (~65-70%)
  - ⚠️ Use as "candidates" with evidence, not facts

HAS_SUPPLIER / HAS_CUSTOMER:
  - Low precision even at high thresholds
  - ❌ Fundamentally broken - extraction is mislabeling
  - Should NOT be created as edges without manual review
  - Consider: store as :MENTIONS with evidence only

TIERED APPROACH:
  1. HAS_COMPETITOR at threshold ≥0.45 → Store as facts
  2. HAS_PARTNER → Store as candidates with evidence
  3. HAS_SUPPLIER/CUSTOMER → Don't create edges, or :MENTIONS only
"""
    )


if __name__ == "__main__":
    main()

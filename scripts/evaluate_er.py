#!/usr/bin/env python3
"""
Evaluate entity resolution quality using labeled ground truth.

Compares the new Wide+Deep scorer against the baseline to measure improvement.

Usage:
    # Character-only evaluation (fast, no API calls)
    python scripts/evaluate_er.py

    # Full Wide+Deep with semantic scoring (requires OpenAI API)
    python scripts/evaluate_er.py --semantic
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from public_company_graph.entity_resolution.character import (
    CharacterMatcher,
)
from public_company_graph.entity_resolution.combined_scorer import (
    CombinedScorer,
    compute_combined_score,
)

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@dataclass
class EvaluationResult:
    """Result for a single ground truth record."""

    source_ticker: str
    target_ticker: str
    target_name: str
    raw_mention: str
    relationship_type: str
    ai_label: str  # correct, incorrect, ambiguous
    ai_confidence: float
    character_score: float
    semantic_score: float
    combined_score: float
    our_confidence_tier: str
    would_reject: bool  # Would we reject this based on low confidence?
    correct_rejection: bool  # Rejecting an incorrect match = good


def load_ground_truth(filepath: str) -> list[dict]:
    """Load the labeled ground truth CSV."""
    records = []
    with open(filepath, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Parse ai_confidence
            try:
                ai_conf = float(row.get("ai_confidence", 0))
            except (ValueError, TypeError):
                ai_conf = 0.0
            row["ai_confidence_float"] = ai_conf
            records.append(row)

    logger.info(f"Loaded {len(records)} ground truth records")
    return records


def evaluate_with_scorer(
    records: list[dict],
    use_semantic: bool = False,
    rejection_threshold: float = 0.50,
) -> list[EvaluationResult]:
    """
    Evaluate each ground truth record with our scorer.

    Args:
        records: Ground truth records
        use_semantic: Whether to use semantic scoring (requires embeddings)
        rejection_threshold: Score below which we'd reject a match

    Returns:
        List of evaluation results
    """
    # Initialize scorer
    get_embedding_fn = None
    if use_semantic:
        try:
            from public_company_graph.embeddings.client import get_embedding

            get_embedding_fn = get_embedding
            logger.info("Semantic scoring enabled")
        except ImportError:
            logger.warning("Could not import embedding client, using character-only")

    scorer = CombinedScorer(
        get_embedding_fn=get_embedding_fn,
        enable_semantic=use_semantic and get_embedding_fn is not None,
    )

    character_matcher = CharacterMatcher()
    results = []

    for i, record in enumerate(records):
        raw_mention = record.get("raw_mention", "")
        target_name = record.get("target_name", "")
        target_ticker = record.get("target_ticker", "")
        context = record.get("context", "")
        ai_label = record.get("ai_label", "")
        ai_confidence = record.get("ai_confidence_float", 0.0)
        relationship_type = record.get("relationship_type", "")

        # Score with our system
        if use_semantic and get_embedding_fn:
            # Full combined score
            combined = scorer.score(
                mention=raw_mention,
                context=context,
                candidate_ticker=target_ticker,
                candidate_name=target_name,
                candidate_embedding=None,  # Would need to load from DB
                relationship_type=relationship_type,
            )
            character_score = combined.character_score
            semantic_score = combined.semantic_score
            combined_score = combined.final_score
            confidence_tier = combined.confidence_tier.value
        else:
            # Character-only scoring
            char_result = character_matcher.score(raw_mention, target_name)
            character_score = char_result.score
            semantic_score = 0.0

            # Compute combined score (character only)
            combined = compute_combined_score(
                mention=raw_mention,
                candidate_ticker=target_ticker,
                candidate_name=target_name,
                character_score=character_score,
                semantic_score=0.0,
                character_weight=1.0,
                semantic_weight=0.0,
            )
            combined_score = combined.final_score
            confidence_tier = combined.confidence_tier.value

        # Would we reject this match?
        would_reject = combined_score < rejection_threshold

        # Is rejection correct? (rejecting an incorrect match is good)
        correct_rejection = would_reject and ai_label == "incorrect"

        result = EvaluationResult(
            source_ticker=record.get("source_ticker", ""),
            target_ticker=target_ticker,
            target_name=target_name,
            raw_mention=raw_mention,
            relationship_type=relationship_type,
            ai_label=ai_label,
            ai_confidence=ai_confidence,
            character_score=character_score,
            semantic_score=semantic_score,
            combined_score=combined_score,
            our_confidence_tier=confidence_tier,
            would_reject=would_reject,
            correct_rejection=correct_rejection,
        )
        results.append(result)

        if (i + 1) % 20 == 0:
            logger.info(f"Evaluated {i + 1}/{len(records)} records")

    return results


def analyze_results(results: list[EvaluationResult]) -> dict:
    """Analyze evaluation results and compute metrics."""
    # Group by AI label
    by_label = defaultdict(list)
    for r in results:
        by_label[r.ai_label].append(r)

    # Score statistics by label
    stats = {}
    for label in ["correct", "incorrect", "ambiguous"]:
        group = by_label[label]
        if group:
            scores = [r.combined_score for r in group]
            char_scores = [r.character_score for r in group]
            stats[label] = {
                "count": len(group),
                "avg_combined": sum(scores) / len(scores),
                "avg_character": sum(char_scores) / len(char_scores),
                "min_combined": min(scores),
                "max_combined": max(scores),
            }

    # Rejection analysis
    total = len(results)
    would_reject = sum(1 for r in results if r.would_reject)
    correct_rejections = sum(1 for r in results if r.correct_rejection)
    incorrect_count = len(by_label["incorrect"])

    # Tier distribution
    tier_counts = Counter(r.our_confidence_tier for r in results)

    return {
        "total": total,
        "by_label": stats,
        "rejection": {
            "would_reject": would_reject,
            "correct_rejections": correct_rejections,
            "total_incorrect": incorrect_count,
            "recall_on_incorrect": (correct_rejections / incorrect_count if incorrect_count else 0),
        },
        "tier_distribution": dict(tier_counts),
    }


def print_report(analysis: dict, results: list[EvaluationResult]) -> None:
    """Print a formatted report."""
    print("\n" + "=" * 70)
    print("ENTITY RESOLUTION EVALUATION REPORT")
    print("=" * 70)

    # Baseline
    by_label = analysis["by_label"]
    total = analysis["total"]
    correct = by_label.get("correct", {}).get("count", 0)
    incorrect = by_label.get("incorrect", {}).get("count", 0)

    print("\n📊 BASELINE (Current System)")
    print(f"   Total samples: {total}")
    print(f"   Correct:   {correct} ({100 * correct / total:.1f}%)")
    print(f"   Incorrect: {incorrect} ({100 * incorrect / total:.1f}%)")
    baseline_precision = correct / (correct + incorrect) if (correct + incorrect) else 0
    print(f"   Precision: {baseline_precision:.1%}")

    # Score distributions
    print("\n📈 SCORE DISTRIBUTIONS (by AI label)")
    print(f"   {'Label':<12} {'Count':>6} {'Avg Score':>10} {'Min':>8} {'Max':>8}")
    print(f"   {'-' * 48}")
    for label in ["correct", "incorrect", "ambiguous"]:
        if label in by_label:
            s = by_label[label]
            print(
                f"   {label:<12} {s['count']:>6} {s['avg_combined']:>10.3f} "
                f"{s['min_combined']:>8.3f} {s['max_combined']:>8.3f}"
            )

    # Separation analysis
    if "correct" in by_label and "incorrect" in by_label:
        correct_avg = by_label["correct"]["avg_combined"]
        incorrect_avg = by_label["incorrect"]["avg_combined"]
        separation = correct_avg - incorrect_avg

        print("\n🎯 SEPARATION ANALYSIS")
        print(f"   Avg score (correct):   {correct_avg:.3f}")
        print(f"   Avg score (incorrect): {incorrect_avg:.3f}")
        print(f"   Separation gap:        {separation:.3f}")

        if separation > 0.15:
            print("   ✓ Good separation - scorer can distinguish correct from incorrect")
        elif separation > 0.05:
            print("   ~ Moderate separation - some discrimination ability")
        else:
            print("   ✗ Poor separation - scorer not helping")

    # Rejection analysis
    rej = analysis["rejection"]
    print("\n🚫 REJECTION ANALYSIS (threshold = 0.50)")
    print(f"   Would reject: {rej['would_reject']}/{total} matches")
    print(
        f"   Correct rejections (caught errors): {rej['correct_rejections']}/{rej['total_incorrect']}"
    )
    print(f"   Recall on incorrect: {rej['recall_on_incorrect']:.1%}")

    # Confidence tier distribution
    tiers = analysis["tier_distribution"]
    print("\n📊 CONFIDENCE TIER DISTRIBUTION")
    for tier in ["high", "medium", "low"]:
        count = tiers.get(tier, 0)
        print(f"   {tier.upper():<8}: {count:>4} ({100 * count / total:.1f}%)")

    # Show some interesting cases
    print("\n🔍 NOTABLE CASES")

    # High-scoring incorrect (false positives we'd miss)
    false_positives = [r for r in results if r.ai_label == "incorrect" and r.combined_score >= 0.70]
    if false_positives:
        print("\n   ⚠️  High-confidence INCORRECT matches (false positives we'd miss):")
        for r in sorted(false_positives, key=lambda x: -x.combined_score)[:5]:
            print(
                f"      {r.raw_mention[:25]:<25} → {r.target_ticker:<6} "
                f"score={r.combined_score:.2f} ({r.relationship_type})"
            )

    # Low-scoring correct (true positives we'd wrongly reject)
    false_negatives = [r for r in results if r.ai_label == "correct" and r.combined_score < 0.50]
    if false_negatives:
        print("\n   ⚠️  Low-confidence CORRECT matches (would wrongly reject):")
        for r in sorted(false_negatives, key=lambda x: x.combined_score)[:5]:
            print(
                f"      {r.raw_mention[:25]:<25} → {r.target_ticker:<6} "
                f"score={r.combined_score:.2f} ({r.relationship_type})"
            )

    # Good catches - low-scoring incorrect
    good_catches = [r for r in results if r.ai_label == "incorrect" and r.combined_score < 0.50]
    if good_catches:
        print("\n   ✓  Good catches (correctly flagged incorrect):")
        for r in sorted(good_catches, key=lambda x: x.combined_score)[:5]:
            print(
                f"      {r.raw_mention[:25]:<25} → {r.target_ticker:<6} "
                f"score={r.combined_score:.2f} ({r.relationship_type})"
            )

    # Improvement potential
    print("\n📈 IMPROVEMENT POTENTIAL")
    if rej["recall_on_incorrect"] > 0:
        new_incorrect = rej["total_incorrect"] - rej["correct_rejections"]
        new_precision = correct / (correct + new_incorrect) if (correct + new_incorrect) else 0
        improvement = new_precision - baseline_precision
        print("   If we reject low-confidence matches:")
        print(f"   - New precision: {new_precision:.1%} (was {baseline_precision:.1%})")
        print(f"   - Improvement: +{improvement:.1%}")
    else:
        print("   No improvement from rejection threshold")

    print("\n" + "=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Evaluate entity resolution against ground truth")
    parser.add_argument(
        "--input",
        default="data/er_ground_truth_labeled.csv",
        help="Path to labeled ground truth CSV",
    )
    parser.add_argument(
        "--semantic",
        action="store_true",
        help="Enable semantic scoring (requires OpenAI API)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.50,
        help="Rejection threshold for low-confidence matches",
    )
    args = parser.parse_args()

    # Load ground truth
    records = load_ground_truth(args.input)

    # Evaluate
    logger.info(
        f"Evaluating with {'semantic + character' if args.semantic else 'character-only'} scoring"
    )
    results = evaluate_with_scorer(
        records,
        use_semantic=args.semantic,
        rejection_threshold=args.threshold,
    )

    # Analyze
    analysis = analyze_results(results)

    # Report
    print_report(analysis, results)


if __name__ == "__main__":
    main()

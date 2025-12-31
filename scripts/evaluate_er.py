#!/usr/bin/env python3
"""
Evaluate entity resolution accuracy against ground truth.

Uses a labeled ground truth CSV to compute:
- Precision: % of resolved relationships that are correct
- Recall: % of actual relationships that were resolved
- F1 Score: Harmonic mean of precision and recall

Also analyzes errors by confidence tier to identify improvement opportunities.

Usage:
    python scripts/evaluate_er.py --ground-truth data/er_ground_truth.csv
"""

import argparse
import csv
import logging
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


@dataclass
class EvaluationMetrics:
    """Entity resolution evaluation metrics."""

    total_samples: int
    correct: int
    incorrect: int
    ambiguous: int
    unlabeled: int

    # Core metrics
    precision: float  # correct / (correct + incorrect)
    recall: float  # correct / (correct + ambiguous)  -- ambiguous could be correct
    f1: float  # 2 * (precision * recall) / (precision + recall)

    # By confidence tier
    high_correct: int
    high_incorrect: int
    medium_correct: int
    medium_incorrect: int
    low_correct: int
    low_incorrect: int

    # By relationship type
    by_relationship: dict[str, dict]


def load_ground_truth(path: Path) -> list[dict]:
    """Load ground truth CSV file."""
    records = []
    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Normalize label
            label = row.get("label", "").strip().lower()
            if label in ("correct", "yes", "1", "true", "y"):
                label = "correct"
            elif label in ("incorrect", "no", "0", "false", "n", "wrong"):
                label = "incorrect"
            elif label in ("ambiguous", "unsure", "?", "maybe"):
                label = "ambiguous"
            else:
                label = ""  # unlabeled

            row["label_normalized"] = label
            records.append(row)

    return records


def evaluate(records: list[dict]) -> EvaluationMetrics:
    """Compute evaluation metrics from labeled records."""
    # Count by label
    correct = sum(1 for r in records if r["label_normalized"] == "correct")
    incorrect = sum(1 for r in records if r["label_normalized"] == "incorrect")
    ambiguous = sum(1 for r in records if r["label_normalized"] == "ambiguous")
    unlabeled = sum(1 for r in records if r["label_normalized"] == "")

    # Core metrics
    # Precision: Of resolved matches, how many are correct?
    precision_denom = correct + incorrect
    precision = correct / precision_denom if precision_denom > 0 else 0.0

    # Recall: Of all true matches (correct + ambiguous that could be correct),
    # how many did we get correct?
    # Note: We treat ambiguous as potentially correct for recall purposes
    recall_denom = correct + ambiguous + incorrect
    recall = correct / recall_denom if recall_denom > 0 else 0.0

    # F1
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    # By confidence tier
    high_correct = sum(
        1
        for r in records
        if r["label_normalized"] == "correct" and r.get("confidence_tier") == "high"
    )
    high_incorrect = sum(
        1
        for r in records
        if r["label_normalized"] == "incorrect" and r.get("confidence_tier") == "high"
    )
    medium_correct = sum(
        1
        for r in records
        if r["label_normalized"] == "correct" and r.get("confidence_tier") == "medium"
    )
    medium_incorrect = sum(
        1
        for r in records
        if r["label_normalized"] == "incorrect" and r.get("confidence_tier") == "medium"
    )
    low_correct = sum(
        1
        for r in records
        if r["label_normalized"] == "correct" and r.get("confidence_tier") == "low"
    )
    low_incorrect = sum(
        1
        for r in records
        if r["label_normalized"] == "incorrect" and r.get("confidence_tier") == "low"
    )

    # By relationship type
    by_relationship: dict[str, dict] = {}
    rel_types = {r.get("relationship_type", "") for r in records}
    for rel_type in rel_types:
        if not rel_type:
            continue
        rel_records = [r for r in records if r.get("relationship_type") == rel_type]
        rel_correct = sum(1 for r in rel_records if r["label_normalized"] == "correct")
        rel_incorrect = sum(1 for r in rel_records if r["label_normalized"] == "incorrect")
        rel_precision = (
            rel_correct / (rel_correct + rel_incorrect)
            if (rel_correct + rel_incorrect) > 0
            else 0.0
        )
        by_relationship[rel_type] = {
            "total": len(rel_records),
            "correct": rel_correct,
            "incorrect": rel_incorrect,
            "precision": rel_precision,
        }

    return EvaluationMetrics(
        total_samples=len(records),
        correct=correct,
        incorrect=incorrect,
        ambiguous=ambiguous,
        unlabeled=unlabeled,
        precision=precision,
        recall=recall,
        f1=f1,
        high_correct=high_correct,
        high_incorrect=high_incorrect,
        medium_correct=medium_correct,
        medium_incorrect=medium_incorrect,
        low_correct=low_correct,
        low_incorrect=low_incorrect,
        by_relationship=by_relationship,
    )


def analyze_errors(records: list[dict]) -> None:
    """Analyze error patterns in incorrect matches."""
    incorrect = [r for r in records if r["label_normalized"] == "incorrect"]

    if not incorrect:
        logger.info("\n✓ No incorrect matches found!")
        return

    logger.info(f"\n=== Error Analysis ({len(incorrect)} incorrect matches) ===")

    # Analyze by mention length
    short_mention_errors = [r for r in incorrect if len(r.get("raw_mention", "")) <= 4]
    logger.info(
        f"\nShort mention errors (≤4 chars): {len(short_mention_errors)} "
        f"({100 * len(short_mention_errors) / len(incorrect):.1f}%)"
    )
    if short_mention_errors[:5]:
        logger.info("  Examples:")
        for r in short_mention_errors[:5]:
            logger.info(f"    '{r.get('raw_mention')}' → {r.get('target_ticker')}")

    # Analyze by relationship type
    error_by_rel = Counter(r.get("relationship_type", "") for r in incorrect)
    logger.info("\nErrors by relationship type:")
    for rel_type, count in error_by_rel.most_common():
        logger.info(f"  {rel_type}: {count}")

    # Show specific error examples
    logger.info("\nError examples:")
    for i, r in enumerate(incorrect[:10], 1):
        logger.info(
            f"  {i}. '{r.get('raw_mention')}' → {r.get('target_ticker')} ({r.get('target_name')})"
        )
        logger.info(f"     Source: {r.get('source_ticker')}, Confidence: {r.get('confidence')}")
        if r.get("notes"):
            logger.info(f"     Notes: {r.get('notes')}")


def print_report(metrics: EvaluationMetrics) -> None:
    """Print evaluation report."""
    print("\n" + "=" * 60)
    print("ENTITY RESOLUTION EVALUATION REPORT")
    print("=" * 60)

    print("\n📊 Dataset Summary")
    print(f"   Total samples: {metrics.total_samples}")
    print(f"   Labeled: {metrics.correct + metrics.incorrect + metrics.ambiguous}")
    print(f"   Unlabeled: {metrics.unlabeled}")

    print("\n📈 Core Metrics")
    print(
        f"   Precision: {metrics.precision:.1%} ({metrics.correct}/{metrics.correct + metrics.incorrect})"
    )
    print(f"   Recall:    {metrics.recall:.1%}")
    print(f"   F1 Score:  {metrics.f1:.1%}")

    print("\n📊 By Confidence Tier")
    print(f"   HIGH:   {metrics.high_correct} correct, {metrics.high_incorrect} incorrect")
    if metrics.high_correct + metrics.high_incorrect > 0:
        high_prec = metrics.high_correct / (metrics.high_correct + metrics.high_incorrect)
        print(f"           Precision: {high_prec:.1%}")
    print(f"   MEDIUM: {metrics.medium_correct} correct, {metrics.medium_incorrect} incorrect")
    if metrics.medium_correct + metrics.medium_incorrect > 0:
        med_prec = metrics.medium_correct / (metrics.medium_correct + metrics.medium_incorrect)
        print(f"           Precision: {med_prec:.1%}")
    print(f"   LOW:    {metrics.low_correct} correct, {metrics.low_incorrect} incorrect")
    if metrics.low_correct + metrics.low_incorrect > 0:
        low_prec = metrics.low_correct / (metrics.low_correct + metrics.low_incorrect)
        print(f"           Precision: {low_prec:.1%}")

    print("\n📊 By Relationship Type")
    for rel_type, data in sorted(metrics.by_relationship.items()):
        print(
            f"   {rel_type}: {data['precision']:.1%} precision ({data['correct']}/{data['total']})"
        )

    print("\n" + "=" * 60)

    # Interpretation
    if metrics.precision >= 0.90:
        print("✅ Excellent precision! Entity resolution is working well.")
    elif metrics.precision >= 0.75:
        print("🟡 Good precision, but room for improvement.")
    else:
        print("🔴 Low precision - significant false positives detected.")

    if metrics.high_incorrect > 0:
        print(f"⚠️  {metrics.high_incorrect} high-confidence errors - review blocklists")

    print()


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate entity resolution accuracy against ground truth"
    )
    parser.add_argument(
        "--ground-truth",
        "-g",
        type=Path,
        required=True,
        help="Path to ground truth CSV",
    )
    parser.add_argument(
        "--analyze-errors",
        action="store_true",
        help="Show detailed error analysis",
    )
    args = parser.parse_args()

    if not args.ground_truth.exists():
        logger.error(f"Ground truth file not found: {args.ground_truth}")
        return

    logger.info(f"Loading ground truth from {args.ground_truth}")
    records = load_ground_truth(args.ground_truth)

    if not records:
        logger.error("No records found in ground truth file")
        return

    # Check for unlabeled records
    unlabeled = sum(1 for r in records if r["label_normalized"] == "")
    if unlabeled == len(records):
        logger.warning(
            "\n⚠️  All records are unlabeled! Please fill in the 'label' column with:\n"
            "    - 'correct' for correct matches\n"
            "    - 'incorrect' for wrong matches\n"
            "    - 'ambiguous' for uncertain cases\n"
        )
        return

    if unlabeled > 0:
        logger.warning(
            f"\n⚠️  {unlabeled} records are unlabeled and will be excluded from metrics\n"
        )

    metrics = evaluate(records)
    print_report(metrics)

    if args.analyze_errors:
        analyze_errors(records)


if __name__ == "__main__":
    main()

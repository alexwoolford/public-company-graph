#!/usr/bin/env python3
"""
Evaluate entity resolution on a specific split.

Usage:
  python scripts/er_evaluate_split.py --split train       # Develop/analyze
  python scripts/er_evaluate_split.py --split validation  # Check for overfitting
  python scripts/er_evaluate_split.py --split test        # FINAL evaluation only!
"""

import argparse
import csv
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from public_company_graph.entity_resolution.layered_validator import (
    LayeredEntityValidator,
)

DATA_DIR = Path(__file__).parent.parent / "data"


def load_split(split: str) -> list[dict]:
    """Load a specific split file."""
    filepath = DATA_DIR / f"er_{split}.csv"
    if not filepath.exists():
        raise FileNotFoundError(f"{filepath} not found. Run: python scripts/er_train_test_split.py")

    with open(filepath, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def evaluate_baseline(records: list[dict]) -> dict:
    """Baseline: trust all relationships."""
    correct = sum(1 for r in records if r["ai_label"] == "correct")
    total = len(records)

    return {
        "name": "Baseline (no validation)",
        "correct": correct,
        "total": total,
        "precision": correct / total if total > 0 else 0,
        "rejected": 0,
    }


def evaluate_layered(records: list[dict], embedding_threshold: float) -> dict:
    """Evaluate layered validator."""
    validator = LayeredEntityValidator(
        embedding_threshold=embedding_threshold,
        skip_embedding=False,
    )

    kept_correct = 0
    kept_incorrect = 0
    rejected_correct = 0
    rejected_incorrect = 0

    for r in records:
        result = validator.validate(
            context=r.get("context", ""),
            mention=r.get("raw_mention", ""),
            ticker=r["target_ticker"],
            company_name=r.get("target_name", ""),
            relationship_type=r["relationship_type"],
        )

        is_correct = r["ai_label"] == "correct"

        if result.accepted:
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

    # Recall: what fraction of correct records did we keep?
    total_correct = kept_correct + rejected_correct
    recall = kept_correct / total_correct if total_correct > 0 else 0

    return {
        "name": "Layered Validator",
        "kept_correct": kept_correct,
        "kept_incorrect": kept_incorrect,
        "rejected_correct": rejected_correct,
        "rejected_incorrect": rejected_incorrect,
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate ER on a split")
    parser.add_argument(
        "--split",
        choices=["train", "validation", "test"],
        required=True,
        help="Which split to evaluate on",
    )
    parser.add_argument(
        "--embedding-threshold",
        type=float,
        default=0.30,
        help="Embedding similarity threshold (default: 0.30)",
    )
    args = parser.parse_args()

    # Warning for test set
    if args.split == "test":
        print("⚠️  WARNING: Evaluating on TEST set.")
        print("   Only do this for FINAL evaluation!")
        print("   Do NOT tune parameters based on these results.")
        response = input("   Continue? (yes/no): ")
        if response.lower() != "yes":
            print("Aborted.")
            return
        print()

    records = load_split(args.split)
    print(f"Loaded {len(records)} records from er_{args.split}.csv")
    print()

    # Baseline
    baseline = evaluate_baseline(records)
    print("BASELINE (no validation)")
    print(f"  Precision: {baseline['precision']:.1%} ({baseline['correct']}/{baseline['total']})")
    print()

    # Layered validator
    layered = evaluate_layered(records, args.embedding_threshold)

    print(f"LAYERED VALIDATOR (threshold={args.embedding_threshold})")
    print(f"  Kept:     {layered['kept_correct']} correct, {layered['kept_incorrect']} incorrect")
    print(
        f"  Rejected: {layered['rejected_correct']} correct (FN), {layered['rejected_incorrect']} incorrect (TN)"
    )
    print(f"  Precision: {layered['precision']:.1%}")
    print(f"  Recall:    {layered['recall']:.1%}")
    print(f"  F1:        {layered['f1']:.1%}")
    print()

    # Improvement
    improvement = layered["precision"] - baseline["precision"]
    print(f"IMPROVEMENT: {improvement:+.1%} precision vs baseline")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Analyze errors on a specific split to guide improvements.

ONLY run this on train or validation sets during development.
Use this to understand patterns and develop new logic.

Usage:
  python scripts/er_analyze_errors.py --split train
  python scripts/er_analyze_errors.py --split train --type false-positives
  python scripts/er_analyze_errors.py --split train --type false-negatives
"""

import argparse
import csv
from collections import Counter
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


def analyze_errors(records: list[dict], error_type: str | None = None):
    """Analyze false positives and false negatives."""
    validator = LayeredEntityValidator(
        embedding_threshold=0.30,
        skip_embedding=False,
    )

    false_positives = []  # Kept but incorrect
    false_negatives = []  # Rejected but correct

    for r in records:
        result = validator.validate(
            context=r.get("context", ""),
            mention=r.get("raw_mention", ""),
            ticker=r["target_ticker"],
            company_name=r.get("target_name", ""),
            relationship_type=r["relationship_type"],
        )

        is_correct = r["ai_label"] == "correct"

        if result.accepted and not is_correct:
            false_positives.append(
                {
                    "record": r,
                    "result": result,
                }
            )
        elif not result.accepted and is_correct:
            false_negatives.append(
                {
                    "record": r,
                    "result": result,
                }
            )

    if error_type == "false-positives" or error_type is None:
        print("=" * 70)
        print(f"FALSE POSITIVES ({len(false_positives)}) - Kept but should be rejected")
        print("=" * 70)
        print("These are the ones hurting precision. Look for patterns to filter.")
        print()

        # Group by AI reasoning
        reasons = Counter()
        for fp in false_positives:
            reason = fp["record"].get("ai_business_logic", "unknown")[:50]
            reasons[reason] += 1

        print("Top reasons (from AI):")
        for reason, count in reasons.most_common(10):
            print(f"  {count}x: {reason}...")
        print()

        # Show examples
        for i, fp in enumerate(false_positives[:5], 1):
            r = fp["record"]
            result = fp["result"]
            print(
                f"--- FP {i}: {r['source_ticker']} -> {r['target_ticker']} ({r['relationship_type']}) ---"
            )
            print(f"Mention: {r.get('raw_mention', 'N/A')}")
            print(f"Context: {r.get('context', 'N/A')[:200]}...")
            print(f"AI reason: {r.get('ai_business_logic', 'N/A')}")
            print(
                f"Embedding score: {result.embedding_similarity:.3f}"
                if result.embedding_similarity
                else "N/A"
            )
            print()

    if error_type == "false-negatives" or error_type is None:
        print("=" * 70)
        print(f"FALSE NEGATIVES ({len(false_negatives)}) - Rejected but should be kept")
        print("=" * 70)
        print("These are hurting recall. Are filters too aggressive?")
        print()

        # Group by rejection reason
        rejection_reasons = Counter()
        for fn in false_negatives:
            result = fn["result"]
            if not result.embedding_passed:
                rejection_reasons["embedding_too_low"] += 1
            elif not result.biographical_passed:
                rejection_reasons["biographical_filter"] += 1
            elif not result.relationship_passed:
                rejection_reasons["relationship_verifier"] += 1
            else:
                rejection_reasons["unknown"] += 1

        print("Rejection reasons:")
        for reason, count in rejection_reasons.items():
            print(f"  {count}x: {reason}")
        print()

        # Show examples
        for i, fn in enumerate(false_negatives[:5], 1):
            r = fn["record"]
            result = fn["result"]
            print(
                f"--- FN {i}: {r['source_ticker']} -> {r['target_ticker']} ({r['relationship_type']}) ---"
            )
            print(f"Mention: {r.get('raw_mention', 'N/A')}")
            print(f"Context: {r.get('context', 'N/A')[:200]}...")
            print(
                f"Embedding: {result.embedding_similarity:.3f} (passed={result.embedding_passed})"
                if result.embedding_similarity
                else "N/A"
            )
            print(f"Biographical: passed={result.biographical_passed}")
            print(f"Relationship: passed={result.relationship_passed}")
            print()

    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"False Positives: {len(false_positives)} (hurting precision)")
    print(f"False Negatives: {len(false_negatives)} (hurting recall)")
    print()
    print("Next steps:")
    print("  - For FPs: Find patterns that can be filtered (new rules)")
    print("  - For FNs: Check if filters are too aggressive (relax thresholds)")
    print("  - After changes, validate on: python scripts/er_evaluate_split.py --split validation")


def main():
    parser = argparse.ArgumentParser(description="Analyze ER errors")
    parser.add_argument(
        "--split",
        choices=["train", "validation"],
        default="train",
        help="Which split to analyze (default: train)",
    )
    parser.add_argument(
        "--type",
        choices=["false-positives", "false-negatives"],
        help="Specific error type to analyze",
    )
    args = parser.parse_args()

    if args.split == "test":
        print("⚠️  ERROR: Do not analyze test set errors!")
        print("   Use train or validation only.")
        return

    records = load_split(args.split)
    print(f"Analyzing {len(records)} records from er_{args.split}.csv")
    print()

    analyze_errors(records, args.type)


if __name__ == "__main__":
    main()

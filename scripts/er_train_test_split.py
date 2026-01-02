#!/usr/bin/env python3
"""
Entity Resolution Train/Validation/Test Split.

Creates deterministic splits of the AI-labeled data for proper evaluation:
- Train (60%): Use for developing patterns and logic
- Validation (20%): Use to detect overfitting during development
- Test (20%): Final evaluation only - DO NOT tune based on this

GOLDEN RULE: Never look at test set results until you're done iterating.
"""

import csv
import random
from pathlib import Path

# Deterministic seed for reproducibility
RANDOM_SEED = 42

# File paths
DATA_DIR = Path(__file__).parent.parent / "data"
SOURCE_FILE = DATA_DIR / "er_ai_audit.csv"
TRAIN_FILE = DATA_DIR / "er_train.csv"
VALIDATION_FILE = DATA_DIR / "er_validation.csv"
TEST_FILE = DATA_DIR / "er_test.csv"


def load_labeled_records() -> list[dict]:
    """Load all labeled (non-ambiguous) records."""
    with open(SOURCE_FILE, encoding="utf-8") as f:
        records = list(csv.DictReader(f))

    # Filter to labeled records only (correct or incorrect)
    labeled = [r for r in records if r.get("ai_label") in ("correct", "incorrect")]
    return labeled


def split_data(records: list[dict], train_ratio=0.6, val_ratio=0.2):
    """Split records into train/validation/test sets."""
    random.seed(RANDOM_SEED)
    shuffled = records.copy()
    random.shuffle(shuffled)

    n = len(shuffled)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    train = shuffled[:train_end]
    validation = shuffled[train_end:val_end]
    test = shuffled[val_end:]

    return train, validation, test


def save_split(records: list[dict], filepath: Path):
    """Save records to CSV."""
    if not records:
        return

    fieldnames = list(records[0].keys())
    with open(filepath, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)


def print_stats(name: str, records: list[dict]):
    """Print statistics for a split."""
    total = len(records)
    correct = sum(1 for r in records if r["ai_label"] == "correct")
    incorrect = total - correct
    precision = correct / total if total > 0 else 0

    print(
        f"  {name:12} {total:>4} records  ({correct} correct, {incorrect} incorrect)  precision={precision:.1%}"
    )


def main():
    print("=" * 60)
    print("ENTITY RESOLUTION TRAIN/VALIDATION/TEST SPLIT")
    print("=" * 60)

    # Load data
    records = load_labeled_records()
    print(f"\nLoaded {len(records)} labeled records from {SOURCE_FILE.name}")

    # Split
    train, validation, test = split_data(records)

    print(f"\nSplit (seed={RANDOM_SEED}):")
    print_stats("Train", train)
    print_stats("Validation", validation)
    print_stats("Test", test)

    # Save
    save_split(train, TRAIN_FILE)
    save_split(validation, VALIDATION_FILE)
    save_split(test, TEST_FILE)

    print("\nSaved to:")
    print(f"  {TRAIN_FILE}")
    print(f"  {VALIDATION_FILE}")
    print(f"  {TEST_FILE}")

    print("\n" + "=" * 60)
    print("WORKFLOW")
    print("=" * 60)
    print("""
1. DEVELOP on Train set:
   - Analyze errors: python scripts/er_analyze_errors.py --split train
   - Implement fixes based on train set patterns

2. CHECK on Validation set:
   - After each change: python scripts/er_evaluate_split.py --split validation
   - If validation improves → good change
   - If validation degrades → likely overfitting, revert

3. FINAL evaluation on Test set (ONCE, at the end):
   - python scripts/er_evaluate_split.py --split test
   - This is your true generalization performance

⚠️  GOLDEN RULE: Do NOT tune based on test set results!
""")


if __name__ == "__main__":
    main()

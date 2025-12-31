#!/usr/bin/env python3
"""
Create ground truth dataset for entity resolution evaluation.

Exports a sample of business relationships with their full context
for manual labeling. This creates the benchmark to measure improvement.

Output: CSV with columns:
- source_ticker: Company that made the mention
- target_ticker: Resolved company
- target_name: Resolved company name
- relationship_type: HAS_COMPETITOR, HAS_SUPPLIER, etc.
- raw_mention: Original text that was matched
- context: Surrounding text from 10-K
- confidence: Current confidence score
- label: (empty - to be filled manually: correct/incorrect/ambiguous)
- notes: (empty - for annotator comments)
"""

import argparse
import csv
import logging
import random
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def export_ground_truth_sample(
    driver,
    output_path: Path,
    sample_size: int = 100,
    stratify: bool = True,
) -> None:
    """
    Export a stratified sample of business relationships for labeling.

    Stratification ensures we get a mix of:
    - High/medium/low confidence matches
    - Different relationship types
    - Short vs long mention names
    """
    with driver.session(database="domain") as session:
        # Query relationships with metadata
        query = """
        MATCH (source:Company)-[r]->(target:Company)
        WHERE type(r) IN ['HAS_COMPETITOR', 'HAS_CUSTOMER', 'HAS_SUPPLIER', 'HAS_PARTNER']
        RETURN
            source.ticker AS source_ticker,
            source.name AS source_name,
            target.ticker AS target_ticker,
            target.name AS target_name,
            type(r) AS relationship_type,
            r.raw_mention AS raw_mention,
            r.context AS context,
            r.confidence AS confidence,
            r.confidence_tier AS confidence_tier
        ORDER BY rand()
        LIMIT $limit
        """

        result = session.run(query, limit=sample_size * 3)  # Get extra for stratification
        all_records = list(result)

        if not all_records:
            logger.error("No business relationships found in database!")
            return

        logger.info(f"Retrieved {len(all_records)} relationships")

        # Stratify by confidence tier if requested
        if stratify and len(all_records) > sample_size:
            high_conf = [r for r in all_records if (r["confidence"] or 0) >= 0.8]
            med_conf = [r for r in all_records if 0.5 <= (r["confidence"] or 0) < 0.8]
            low_conf = [r for r in all_records if (r["confidence"] or 0) < 0.5]

            # Target distribution: 40% high, 40% medium, 20% low
            n_high = min(len(high_conf), int(sample_size * 0.4))
            n_med = min(len(med_conf), int(sample_size * 0.4))
            n_low = min(len(low_conf), int(sample_size * 0.2))

            # Fill remainder from largest bucket
            remaining = sample_size - n_high - n_med - n_low
            if remaining > 0:
                if len(high_conf) > n_high:
                    n_high += remaining
                elif len(med_conf) > n_med:
                    n_med += remaining

            random.shuffle(high_conf)
            random.shuffle(med_conf)
            random.shuffle(low_conf)

            selected = high_conf[:n_high] + med_conf[:n_med] + low_conf[:n_low]
            random.shuffle(selected)
        else:
            selected = all_records[:sample_size]

        logger.info(f"Selected {len(selected)} relationships for ground truth")

        # Write CSV
        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "id",
                    "source_ticker",
                    "source_name",
                    "target_ticker",
                    "target_name",
                    "relationship_type",
                    "raw_mention",
                    "context",
                    "confidence",
                    "confidence_tier",
                    "label",  # To be filled: correct/incorrect/ambiguous
                    "notes",  # For annotator comments
                ]
            )

            for i, record in enumerate(selected, 1):
                # Truncate context for readability (keep first 500 chars)
                context = record["context"] or ""
                if len(context) > 500:
                    context = context[:500] + "..."

                writer.writerow(
                    [
                        i,
                        record["source_ticker"],
                        record["source_name"],
                        record["target_ticker"],
                        record["target_name"],
                        record["relationship_type"],
                        record["raw_mention"],
                        context,
                        f"{record['confidence']:.3f}" if record["confidence"] else "",
                        record["confidence_tier"],
                        "",  # label - to be filled
                        "",  # notes - to be filled
                    ]
                )

        logger.info(f"✓ Wrote ground truth dataset to {output_path}")
        logger.info(f"  Total samples: {len(selected)}")

        # Print confidence distribution
        conf_dist = {}
        for r in selected:
            tier = r["confidence_tier"] or "unknown"
            conf_dist[tier] = conf_dist.get(tier, 0) + 1
        logger.info(f"  Confidence distribution: {conf_dist}")

        # Print relationship type distribution
        rel_dist = {}
        for r in selected:
            rel_type = r["relationship_type"]
            rel_dist[rel_type] = rel_dist.get(rel_type, 0) + 1
        logger.info(f"  Relationship distribution: {rel_dist}")


def main():
    parser = argparse.ArgumentParser(
        description="Create ground truth dataset for entity resolution evaluation"
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=Path("data/er_ground_truth.csv"),
        help="Output CSV path (default: data/er_ground_truth.csv)",
    )
    parser.add_argument(
        "--sample-size",
        "-n",
        type=int,
        default=100,
        help="Number of samples to export (default: 100)",
    )
    parser.add_argument(
        "--no-stratify",
        action="store_true",
        help="Disable stratification by confidence tier",
    )
    args = parser.parse_args()

    # Ensure output directory exists
    args.output.parent.mkdir(parents=True, exist_ok=True)

    # Connect to Neo4j
    from public_company_graph.neo4j.connection import get_neo4j_driver

    driver = get_neo4j_driver()

    try:
        export_ground_truth_sample(
            driver,
            args.output,
            sample_size=args.sample_size,
            stratify=not args.no_stratify,
        )
    finally:
        driver.close()

    print("\n📋 Next steps:")
    print(f"   1. Open {args.output} in a spreadsheet")
    print("   2. Review each row and fill in 'label' column:")
    print("      - 'correct': The resolved company is correct")
    print("      - 'incorrect': The resolved company is wrong")
    print("      - 'ambiguous': Cannot determine / multiple valid interpretations")
    print("   3. Add notes explaining incorrect matches")
    print(f"   4. Save and run: python scripts/evaluate_er.py --ground-truth {args.output}")


if __name__ == "__main__":
    main()

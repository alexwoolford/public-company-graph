#!/usr/bin/env python3
"""
Entity Resolution Ground Truth Tool.

A unified CLI for creating, labeling, and evaluating entity resolution
ground truth datasets. This tool helps measure and improve the accuracy
of company mention extraction from SEC 10-K filings.

WORKFLOW:
    1. Sample relationships from the graph
    2. Label samples with AI assistance (parallel)
    3. Evaluate filter/verifier performance against labeled data

USAGE:
    # Step 1: Extract 200 random samples from Neo4j
    python scripts/er_ground_truth.py sample --count 200

    # Step 2: Label samples with AI (parallel, ~10x faster)
    python scripts/er_ground_truth.py label --concurrency 10

    # Step 3: Evaluate current filters against labeled data
    python scripts/er_ground_truth.py evaluate

    # View dataset statistics
    python scripts/er_ground_truth.py stats

DATA FILE:
    All operations use: data/er_ground_truth.csv
    - Unlabeled samples have empty ai_label column
    - Labeled samples have ai_label = correct|incorrect|ambiguous
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import logging
import random
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Default data file path
DEFAULT_DATA_FILE = Path("data/er_ground_truth.csv")

# CSV columns
BASE_COLUMNS = [
    "id",
    "source_ticker",
    "source_name",
    "source_cik",
    "target_ticker",
    "target_name",
    "target_cik",
    "relationship_type",
    "raw_mention",
    "context",
    "confidence",
    "confidence_tier",
]

AI_COLUMNS = [
    "ai_label",
    "ai_confidence",
    "ai_reasoning",
    "ai_business_logic",
    "ai_alternative",
]

ALL_COLUMNS = BASE_COLUMNS + AI_COLUMNS


# =============================================================================
# SAMPLE SUBCOMMAND - Extract samples from Neo4j
# =============================================================================


def cmd_sample(args: argparse.Namespace) -> None:
    """Extract random relationship samples from Neo4j."""
    from public_company_graph.config import get_settings
    from public_company_graph.neo4j.connection import get_neo4j_driver

    data_file: Path = args.data_file
    count: int = args.count
    relationship_types: list[str] = args.types
    append: bool = args.append

    logger.info(f"Sampling {count} relationships from Neo4j...")
    logger.info(f"  Types: {relationship_types}")

    settings = get_settings()
    driver = get_neo4j_driver()

    # Query for random relationships
    query = """
    MATCH (source:Company)-[r]->(target:Company)
    WHERE type(r) IN $types
    WITH source, target, r, rand() AS random
    ORDER BY random
    LIMIT $count
    RETURN
        source.ticker AS source_ticker,
        source.name AS source_name,
        source.cik AS source_cik,
        target.ticker AS target_ticker,
        target.name AS target_name,
        target.cik AS target_cik,
        type(r) AS relationship_type,
        r.raw_mention AS raw_mention,
        r.context AS context,
        r.confidence AS confidence,
        r.confidence_tier AS confidence_tier
    """

    records = []
    with driver.session(database=settings.neo4j_database) as session:
        result = session.run(query, types=relationship_types, count=count)
        for record in result:
            records.append(dict(record))

    driver.close()

    if not records:
        logger.warning("No relationships found. Is the graph populated?")
        return

    # Shuffle for good measure
    random.shuffle(records)

    # Load existing records if appending
    existing_records = []
    existing_keys = set()
    next_id = 1

    if append and data_file.exists():
        with open(data_file, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                existing_records.append(row)
                key = f"{row['source_ticker']}→{row['target_ticker']}→{row['relationship_type']}"
                existing_keys.add(key)
                try:
                    next_id = max(next_id, int(row.get("id", 0)) + 1)
                except (ValueError, TypeError):
                    pass
        logger.info(f"  Appending to {len(existing_records)} existing records")

    # Add IDs and filter duplicates
    new_records = []
    for record in records:
        key = f"{record['source_ticker']}→{record['target_ticker']}→{record['relationship_type']}"
        if key not in existing_keys:
            record["id"] = next_id
            next_id += 1
            # Initialize AI columns as empty
            for col in AI_COLUMNS:
                record[col] = ""
            new_records.append(record)
            existing_keys.add(key)

    if not new_records:
        logger.warning("All sampled records already exist in dataset")
        return

    # Combine and write
    all_records = existing_records + new_records

    data_file.parent.mkdir(parents=True, exist_ok=True)
    with open(data_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=ALL_COLUMNS)
        writer.writeheader()
        for record in all_records:
            # Ensure all columns exist
            row = {col: record.get(col, "") for col in ALL_COLUMNS}
            writer.writerow(row)

    logger.info(f"✓ Added {len(new_records)} new samples")
    logger.info(f"  Total dataset size: {len(all_records)} records")
    logger.info(f"  Saved to: {data_file}")


# =============================================================================
# LABEL SUBCOMMAND - AI-assisted labeling (parallel)
# =============================================================================


@dataclass
class CompanyInfo:
    """Company information for AI context."""

    ticker: str
    name: str
    description: str | None
    sector: str | None
    industry: str | None


@dataclass
class LabelingResult:
    """Result of AI labeling."""

    label: str
    confidence: float
    reasoning: str
    business_logic_check: str
    alternative_interpretation: str | None


@dataclass
class ProgressTracker:
    """Track progress across parallel tasks."""

    total: int
    completed: int = 0
    correct: int = 0
    incorrect: int = 0
    ambiguous: int = 0
    errors: int = 0
    start_time: float = field(default_factory=time.time)
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    async def record(self, label: str, is_error: bool = False) -> None:
        async with self._lock:
            self.completed += 1
            if is_error:
                self.errors += 1
            elif label == "correct":
                self.correct += 1
            elif label == "incorrect":
                self.incorrect += 1
            else:
                self.ambiguous += 1

    def summary(self) -> str:
        elapsed = time.time() - self.start_time
        rate = self.completed / elapsed if elapsed > 0 else 0
        remaining = (self.total - self.completed) / rate if rate > 0 else 0
        return (
            f"[{self.completed}/{self.total}] "
            f"✓{self.correct} ✗{self.incorrect} ?{self.ambiguous} "
            f"({rate:.1f}/s, ~{remaining:.0f}s left)"
        )


# AI labeling configuration
RESPONSES_API_TOOL = {
    "type": "function",
    "name": "submit_entity_resolution_label",
    "description": "Submit your analysis and label for this entity resolution case",
    "parameters": {
        "type": "object",
        "properties": {
            "label": {
                "type": "string",
                "enum": ["correct", "incorrect", "ambiguous"],
                "description": "Your verdict",
            },
            "confidence": {"type": "number", "description": "Confidence 0.0 to 1.0"},
            "reasoning": {"type": "string", "description": "Your detailed reasoning"},
            "business_logic_check": {
                "type": "string",
                "description": "Does this relationship make business sense?",
            },
            "alternative_interpretation": {
                "type": "string",
                "description": "If incorrect, what might it refer to?",
            },
        },
        "required": ["label", "confidence", "reasoning", "business_logic_check"],
    },
}

LABELING_FUNCTION = {
    "type": "function",
    "function": {**RESPONSES_API_TOOL, "name": RESPONSES_API_TOOL["name"]},
}

SYSTEM_PROMPT = """You are an expert analyst verifying entity resolution in SEC 10-K filings.

Determine whether a company mention was correctly matched to the right public company.

Consider:
1. **Name Match**: Does the mention clearly refer to the target company?
2. **Business Logic**: Does this relationship make sense between these companies?
3. **Context**: Does surrounding text support this interpretation?
4. **Red Flags**: Generic words, ambiguous abbreviations, wrong relationship type

Be CONSERVATIVE: If uncertain, use "ambiguous".
Call submit_entity_resolution_label with your verdict."""


def truncate_text(text: str | None, max_length: int = 500) -> str:
    if not text:
        return "(not available)"
    return text[:max_length] + "..." if len(text) > max_length else text


def build_prompt(
    source: CompanyInfo,
    target: CompanyInfo,
    rel_type: str,
    mention: str,
    context: str,
) -> str:
    rel_simple = rel_type.replace("HAS_", "").lower()
    return f"""Analyze this entity resolution case:

## Source Company (filing the 10-K)
- Ticker: {source.ticker} | Name: {source.name}
- Sector: {source.sector or "unknown"} | Industry: {source.industry or "unknown"}
- Description: {truncate_text(source.description, 300)}

## Target Company (the resolved match)
- Ticker: {target.ticker} | Name: {target.name}
- Sector: {target.sector or "unknown"} | Industry: {target.industry or "unknown"}
- Description: {truncate_text(target.description, 300)}

## Match to Verify
- Relationship: {rel_type} (source mentions target as a {rel_simple})
- Raw Mention: "{mention}"
- Context: "{truncate_text(context, 500)}"

Is this CORRECT, INCORRECT, or AMBIGUOUS?"""


def get_company_info(session: Any, ticker: str) -> CompanyInfo | None:
    result = session.run(
        """
        MATCH (c:Company {ticker: $ticker})
        RETURN c.ticker AS ticker, c.name AS name, c.description AS description,
               c.sector AS sector, c.industry AS industry
        """,
        ticker=ticker,
    )
    record = result.single()
    if record:
        return CompanyInfo(**dict(record))
    return None


async def label_with_responses_api(
    client: Any, prompt: str, model: str, reasoning_effort: str
) -> LabelingResult:
    response = await client.responses.create(
        model=model,
        input=f"{SYSTEM_PROMPT}\n\n{prompt}",
        tools=[RESPONSES_API_TOOL],
        tool_choice="required",
        reasoning={"effort": reasoning_effort},
        timeout=120.0,
    )
    for item in response.output:
        if item.type == "function_call" and item.name == "submit_entity_resolution_label":
            args = json.loads(item.arguments)
            return LabelingResult(
                label=args.get("label", "ambiguous"),
                confidence=float(args.get("confidence", 0.5)),
                reasoning=args.get("reasoning", ""),
                business_logic_check=args.get("business_logic_check", ""),
                alternative_interpretation=args.get("alternative_interpretation"),
            )
    return LabelingResult("ambiguous", 0.0, "No function call", "", None)


async def label_with_chat_completions(client: Any, prompt: str, model: str) -> LabelingResult:
    params: dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        "tools": [LABELING_FUNCTION],
        "tool_choice": {"type": "function", "function": {"name": "submit_entity_resolution_label"}},
    }
    if not model.startswith(("o1", "o3", "o4")):
        params["temperature"] = 0

    response = await client.chat.completions.create(**params, timeout=120.0)
    message = response.choices[0].message
    if message.tool_calls:
        args = json.loads(message.tool_calls[0].function.arguments)
        return LabelingResult(
            label=args.get("label", "ambiguous"),
            confidence=float(args.get("confidence", 0.5)),
            reasoning=args.get("reasoning", ""),
            business_logic_check=args.get("business_logic_check", ""),
            alternative_interpretation=args.get("alternative_interpretation"),
        )
    return LabelingResult("ambiguous", 0.0, "No tool call", "", None)


async def label_single_record(
    client: Any,
    record: dict[str, Any],
    original_index: int,
    source_info: CompanyInfo,
    target_info: CompanyInfo,
    model: str,
    reasoning_effort: str,
    semaphore: asyncio.Semaphore,
    progress: ProgressTracker,
) -> tuple[int, dict[str, Any]]:
    """Label a single record, returning (original_index, updated_record) for order preservation."""
    async with semaphore:
        try:
            prompt = build_prompt(
                source=source_info,
                target=target_info,
                rel_type=record.get("relationship_type", ""),
                mention=record.get("raw_mention", ""),
                context=record.get("context", ""),
            )

            if model == "gpt-5.2-pro":
                result = await label_with_responses_api(client, prompt, model, reasoning_effort)
            else:
                result = await label_with_chat_completions(client, prompt, model)

            record["ai_label"] = result.label
            record["ai_confidence"] = f"{result.confidence:.2f}"
            record["ai_reasoning"] = result.reasoning[:1000]
            record["ai_business_logic"] = result.business_logic_check[:500]
            record["ai_alternative"] = result.alternative_interpretation or ""

            await progress.record(result.label)
            emoji = {"correct": "✓", "incorrect": "✗", "ambiguous": "?"}
            logger.info(
                f"{emoji.get(result.label, '?')} {record['source_ticker']}→{record['target_ticker']}: "
                f"{result.label} | {progress.summary()}"
            )

        except Exception as e:
            logger.error(f"Error: {record['source_ticker']}→{record['target_ticker']}: {e}")
            record["ai_label"] = "ambiguous"
            record["ai_confidence"] = "0.0"
            record["ai_reasoning"] = f"Error: {e}"
            record["ai_business_logic"] = ""
            record["ai_alternative"] = ""
            await progress.record("ambiguous", is_error=True)

        return original_index, record


async def run_labeling(args: argparse.Namespace) -> None:
    """Run parallel AI labeling."""
    from openai import AsyncOpenAI

    from public_company_graph.config import get_settings
    from public_company_graph.neo4j.connection import get_neo4j_driver

    data_file: Path = args.data_file
    model: str = args.model
    reasoning_effort: str = args.reasoning_effort
    concurrency: int = args.concurrency

    if not data_file.exists():
        logger.error(f"Data file not found: {data_file}")
        logger.info("Run: python scripts/er_ground_truth.py sample --count 100")
        return

    # Load records
    with open(data_file, encoding="utf-8") as f:
        all_records = list(csv.DictReader(f))

    # Filter to unlabeled records
    records_to_label = [(i, r) for i, r in enumerate(all_records) if not r.get("ai_label")]

    if not records_to_label:
        logger.info("✓ All records already labeled!")
        return

    logger.info(f"🚀 Labeling {len(records_to_label)} records with {model}")
    logger.info(f"   Concurrency: {concurrency} | Reasoning: {reasoning_effort}")

    # Prefetch company info
    settings = get_settings()
    driver = get_neo4j_driver()
    company_cache: dict[str, CompanyInfo | None] = {}

    logger.info("📥 Fetching company information...")
    with driver.session(database=settings.neo4j_database) as session:
        tickers = set()
        for _, r in records_to_label:
            tickers.add(r.get("source_ticker", ""))
            tickers.add(r.get("target_ticker", ""))
        for ticker in tickers:
            if ticker:
                company_cache[ticker] = get_company_info(session, ticker)
    driver.close()

    # Prepare async tasks
    client = AsyncOpenAI()
    semaphore = asyncio.Semaphore(concurrency)
    progress = ProgressTracker(total=len(records_to_label))

    tasks = []
    for original_idx, record in records_to_label:
        source_info = company_cache.get(record.get("source_ticker", ""))
        target_info = company_cache.get(record.get("target_ticker", ""))

        if not source_info or not target_info:
            record["ai_label"] = "ambiguous"
            record["ai_confidence"] = "0.0"
            record["ai_reasoning"] = "Company info not found"
            record["ai_business_logic"] = ""
            record["ai_alternative"] = ""
            all_records[original_idx] = record
            continue

        tasks.append(
            label_single_record(
                client=client,
                record=record.copy(),
                original_index=original_idx,
                source_info=source_info,
                target_info=target_info,
                model=model,
                reasoning_effort=reasoning_effort,
                semaphore=semaphore,
                progress=progress,
            )
        )

    # Run in parallel
    logger.info(f"🔄 Starting {len(tasks)} parallel tasks...")
    start_time = time.time()
    results = await asyncio.gather(*tasks, return_exceptions=True)
    elapsed = time.time() - start_time

    # Update records in original order
    for result in results:
        if isinstance(result, tuple):
            idx, updated_record = result
            all_records[idx] = updated_record

    # Write back (preserving order)
    with open(data_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=ALL_COLUMNS)
        writer.writeheader()
        for record in all_records:
            row = {col: record.get(col, "") for col in ALL_COLUMNS}
            writer.writerow(row)

    logger.info(f"⏱️  Completed in {elapsed:.1f}s ({len(tasks) / elapsed:.1f} records/sec)")
    logger.info(f"✓ Saved to: {data_file}")


def cmd_label(args: argparse.Namespace) -> None:
    """AI-label unlabeled samples (parallel)."""
    asyncio.run(run_labeling(args))


# =============================================================================
# EVALUATE SUBCOMMAND - Test filters/verifiers
# =============================================================================


def cmd_evaluate(args: argparse.Namespace) -> None:
    """Evaluate filters and verifiers against labeled data."""
    from public_company_graph.entity_resolution.candidates import Candidate
    from public_company_graph.entity_resolution.filters import (
        BiographicalContextFilter,
        ExchangeReferenceFilter,
    )
    from public_company_graph.entity_resolution.relationship_verifier import (
        RelationshipVerifier,
        VerificationResult,
    )

    data_file: Path = args.data_file

    if not data_file.exists():
        logger.error(f"Data file not found: {data_file}")
        return

    # Load labeled records
    with open(data_file, encoding="utf-8") as f:
        records = [r for r in csv.DictReader(f) if r.get("ai_label")]

    if not records:
        logger.error("No labeled records found. Run 'label' first.")
        return

    # Initialize filters
    bio_filter = BiographicalContextFilter()
    exchange_filter = ExchangeReferenceFilter()
    rel_verifier = RelationshipVerifier()

    # Categorize
    correct = [r for r in records if r["ai_label"] == "correct"]
    incorrect = [r for r in records if r["ai_label"] == "incorrect"]
    ambiguous = [r for r in records if r["ai_label"] == "ambiguous"]

    print("=" * 70)
    print("ENTITY RESOLUTION EVALUATION")
    print("=" * 70)
    print(f"\nDataset: {len(records)} labeled records")
    print(f"  Correct:   {len(correct)}")
    print(f"  Incorrect: {len(incorrect)}")
    print(f"  Ambiguous: {len(ambiguous)}")

    # Test each record
    results = []
    for record in records:
        candidate = Candidate(
            text=record.get("raw_mention", ""),
            sentence=record.get("context", ""),
            start_pos=0,
            end_pos=len(record.get("raw_mention", "")),
            source_pattern="ground_truth",
        )

        bio_result = bio_filter.filter(candidate)
        exchange_result = exchange_filter.filter(candidate)
        rel_result = rel_verifier.verify(
            record.get("relationship_type", ""),
            record.get("context", ""),
            record.get("raw_mention", ""),
        )

        results.append(
            {
                **record,
                "bio_filtered": not bio_result.passed,
                "exchange_filtered": not exchange_result.passed,
                "rel_contradicted": rel_result.result == VerificationResult.CONTRADICTED,
                "any_filtered": (
                    not bio_result.passed
                    or not exchange_result.passed
                    or rel_result.result == VerificationResult.CONTRADICTED
                ),
            }
        )

    # Analyze
    print("\n" + "=" * 70)
    print("FILTER EFFECTIVENESS")
    print("=" * 70)

    incorrect_caught = [r for r in results if r["ai_label"] == "incorrect" and r["any_filtered"]]
    incorrect_bio = [r for r in results if r["ai_label"] == "incorrect" and r["bio_filtered"]]
    incorrect_exch = [r for r in results if r["ai_label"] == "incorrect" and r["exchange_filtered"]]
    incorrect_rel = [r for r in results if r["ai_label"] == "incorrect" and r["rel_contradicted"]]

    print(f"\n✓ INCORRECT matches caught: {len(incorrect_caught)}/{len(incorrect)}")
    print(f"  Biographical filter:     {len(incorrect_bio)}")
    print(f"  Exchange filter:         {len(incorrect_exch)}")
    print(f"  Relationship verifier:   {len(incorrect_rel)}")

    # False positives
    correct_filtered = [r for r in results if r["ai_label"] == "correct" and r["any_filtered"]]
    print(f"\n⚠️  False positives: {len(correct_filtered)}/{len(correct)}")

    # Precision improvement
    old_precision = len(correct) / (len(correct) + len(incorrect)) if (correct or incorrect) else 0
    new_correct = len(correct) - len(correct_filtered)
    new_incorrect = len(incorrect) - len(incorrect_caught)
    new_precision = (
        new_correct / (new_correct + new_incorrect) if (new_correct + new_incorrect) else 0
    )

    print("\n" + "=" * 70)
    print("PRECISION IMPROVEMENT")
    print("=" * 70)
    print(f"\n  Before: {old_precision:.1%}")
    print(f"  After:  {new_precision:.1%}")
    print(f"  Change: {(new_precision - old_precision) * 100:+.1f}%")


# =============================================================================
# STATS SUBCOMMAND - Show dataset statistics
# =============================================================================


def cmd_stats(args: argparse.Namespace) -> None:
    """Show dataset statistics."""
    data_file: Path = args.data_file

    if not data_file.exists():
        logger.error(f"Data file not found: {data_file}")
        logger.info("Run: python scripts/er_ground_truth.py sample --count 100")
        return

    with open(data_file, encoding="utf-8") as f:
        records = list(csv.DictReader(f))

    labeled = [r for r in records if r.get("ai_label")]
    unlabeled = [r for r in records if not r.get("ai_label")]

    correct = [r for r in labeled if r["ai_label"] == "correct"]
    incorrect = [r for r in labeled if r["ai_label"] == "incorrect"]
    ambiguous = [r for r in labeled if r["ai_label"] == "ambiguous"]

    print("=" * 50)
    print("GROUND TRUTH DATASET STATISTICS")
    print("=" * 50)
    print(f"\nFile: {data_file}")
    print(f"Total records: {len(records)}")
    print(f"  Labeled:   {len(labeled)}")
    print(f"  Unlabeled: {len(unlabeled)}")

    if labeled:
        print("\nLabel distribution:")
        print(f"  Correct:   {len(correct):>4} ({100 * len(correct) / len(labeled):.1f}%)")
        print(f"  Incorrect: {len(incorrect):>4} ({100 * len(incorrect) / len(labeled):.1f}%)")
        print(f"  Ambiguous: {len(ambiguous):>4} ({100 * len(ambiguous) / len(labeled):.1f}%)")

        if correct or incorrect:
            precision = len(correct) / (len(correct) + len(incorrect))
            print(f"\nEstimated precision: {precision:.1%}")

    # Relationship type breakdown
    by_type: dict[str, int] = {}
    for r in records:
        rel_type = r.get("relationship_type", "UNKNOWN")
        by_type[rel_type] = by_type.get(rel_type, 0) + 1

    print("\nBy relationship type:")
    for rel_type, count in sorted(by_type.items()):
        print(f"  {rel_type}: {count}")


# =============================================================================
# MAIN CLI
# =============================================================================


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Entity Resolution Ground Truth Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
WORKFLOW:
  1. sample   - Extract random samples from Neo4j
  2. label    - AI-label samples (parallel, 10x faster)
  3. evaluate - Test filters against labeled data
  4. stats    - Show dataset statistics

EXAMPLES:
  # Create initial dataset
  python scripts/er_ground_truth.py sample --count 200

  # Label with AI
  python scripts/er_ground_truth.py label

  # Evaluate current system
  python scripts/er_ground_truth.py evaluate
        """,
    )
    parser.add_argument(
        "--data-file",
        type=Path,
        default=DEFAULT_DATA_FILE,
        help=f"Ground truth CSV file (default: {DEFAULT_DATA_FILE})",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    # SAMPLE subcommand
    sample_parser = subparsers.add_parser("sample", help="Extract samples from Neo4j")
    sample_parser.add_argument(
        "--count", "-n", type=int, default=100, help="Number of samples to extract"
    )
    sample_parser.add_argument(
        "--types",
        nargs="+",
        default=["HAS_COMPETITOR", "HAS_CUSTOMER", "HAS_SUPPLIER", "HAS_PARTNER"],
        help="Relationship types to sample",
    )
    sample_parser.add_argument(
        "--append", "-a", action="store_true", help="Append to existing file"
    )
    sample_parser.set_defaults(func=cmd_sample)

    # LABEL subcommand
    label_parser = subparsers.add_parser("label", help="AI-label samples (parallel)")
    label_parser.add_argument(
        "--model",
        "-m",
        default="gpt-5.2-pro",
        choices=["gpt-5.2-pro", "gpt-5.2", "gpt-4o", "o3", "o4-mini"],
        help="Model to use",
    )
    label_parser.add_argument(
        "--reasoning-effort",
        "-r",
        default="high",
        choices=["low", "medium", "high", "xhigh"],
        help="Reasoning effort (gpt-5.2-pro only)",
    )
    label_parser.add_argument(
        "--concurrency", "-c", type=int, default=10, help="Max parallel API calls"
    )
    label_parser.set_defaults(func=cmd_label)

    # EVALUATE subcommand
    eval_parser = subparsers.add_parser("evaluate", help="Evaluate filters")
    eval_parser.set_defaults(func=cmd_evaluate)

    # STATS subcommand
    stats_parser = subparsers.add_parser("stats", help="Show dataset statistics")
    stats_parser.set_defaults(func=cmd_stats)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

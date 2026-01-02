#!/usr/bin/env python3
"""
AI-assisted labeling for entity resolution ground truth.

Uses OpenAI's best models for careful, methodical analysis of each
relationship to determine if the entity resolution was correct.

**PARALLEL IMPLEMENTATION**: Makes concurrent API calls for 10x speedup.

Model recommendations (per OpenAI, December 2025):
- gpt-5.2-pro: Best for accuracy - uses Responses API with reasoning_effort (default)
- gpt-5.2: Good balance - uses Chat Completions
- o3: Reasoning model that "thinks" carefully before responding
- o4-mini: Faster reasoning model for quicker results

Key features:
1. Parallel processing: 10 concurrent API calls (configurable)
2. Rich context: Pulls company descriptions, industries, and 10-K context
3. Function calling: Structured output via tools
4. Reasoning effort: Configurable compute for harder decisions (low/medium/high/xhigh)
5. Resume support: Skip already-processed records
6. Progress tracking: Real-time stats with estimated time remaining

Usage:
    # Default: 10 concurrent requests with gpt-5.2-pro
    python scripts/label_er_with_ai.py

    # More parallelism (stay under rate limits!)
    python scripts/label_er_with_ai.py --concurrency 20

    # Quick test with 10 samples
    python scripts/label_er_with_ai.py -n 10

Cost estimate: gpt-5.2-pro is ~$0.10-0.20 per relationship
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from openai import AsyncOpenAI

# Load environment variables from .env file
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@dataclass
class CompanyInfo:
    """Information about a company for context."""

    ticker: str
    name: str
    description: str | None
    sector: str | None
    industry: str | None
    sic_code: str | None


@dataclass
class LabelingResult:
    """Result of AI labeling."""

    label: str  # correct, incorrect, ambiguous
    confidence: float  # 0-1
    reasoning: str  # Explanation
    business_logic_check: str  # Does the relationship make sense?
    alternative_interpretation: str | None  # If incorrect, what might be right?


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
        """Record a completed task."""
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
        """Return progress summary."""
        elapsed = time.time() - self.start_time
        rate = self.completed / elapsed if elapsed > 0 else 0
        remaining = (self.total - self.completed) / rate if rate > 0 else 0
        return (
            f"[{self.completed}/{self.total}] "
            f"✓{self.correct} ✗{self.incorrect} ?{self.ambiguous} "
            f"({rate:.1f}/s, ~{remaining:.0f}s left)"
        )


# Tool definition for Responses API (gpt-5.2-pro)
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
            "confidence": {
                "type": "number",
                "description": "Your confidence (0.0 to 1.0)",
            },
            "reasoning": {
                "type": "string",
                "description": "Your detailed reasoning",
            },
            "business_logic_check": {
                "type": "string",
                "description": "Does this relationship make business sense?",
            },
            "alternative_interpretation": {
                "type": "string",
                "description": "If incorrect, what might the mention refer to?",
            },
        },
        "required": ["label", "confidence", "reasoning", "business_logic_check"],
    },
}

# Function definition for Chat Completions API
LABELING_FUNCTION = {
    "type": "function",
    "function": {
        "name": "submit_entity_resolution_label",
        "description": "Submit your analysis and label for this entity resolution case",
        "parameters": RESPONSES_API_TOOL["parameters"],
    },
}

# System prompt for the labeling task
SYSTEM_PROMPT = """You are an expert analyst verifying entity resolution in SEC 10-K filings.

Your task is to carefully determine whether a company mention in a 10-K filing was correctly matched to the right public company.

Think through each case VERY carefully. Consider:

1. **Name Match Quality**: Does the raw mention clearly refer to the target company?
   - Is this an exact match, abbreviation, or ambiguous reference?
   - Could this name refer to a different company entirely?

2. **Business Logic**: Does this relationship make sense?
   - Are these companies in related industries where this relationship is plausible?
   - Would the source company realistically have this relationship with the target?

3. **Context Analysis**: What does the surrounding text suggest?
   - Does the context support this being a reference to the target company?
   - Are there any clues that suggest a different interpretation?

4. **Red Flags**: Watch for common entity resolution errors:
   - Generic words matching ticker symbols (e.g., "AI" → C3.ai)
   - Common business terms matching company names (e.g., "Target" → Target Corp)
   - Ambiguous abbreviations

Be CONSERVATIVE: If you're not highly confident, use "ambiguous".
Be RIGOROUS: Explain your reasoning thoroughly.

After your analysis, call the submit_entity_resolution_label function with your verdict."""


def truncate_text(text: str | None, max_length: int = 500) -> str:
    """Truncate text to reasonable length."""
    if not text:
        return "(not available)"
    if len(text) <= max_length:
        return text
    return text[:max_length] + "..."


def build_user_prompt(
    source_info: CompanyInfo,
    target_info: CompanyInfo,
    relationship_type: str,
    raw_mention: str,
    context: str,
) -> str:
    """Build the user prompt with all context."""
    relationship_type_simple = relationship_type.replace("HAS_", "").lower()

    return f"""Please analyze this entity resolution case:

## Source Company (filing the 10-K)
- Ticker: {source_info.ticker}
- Name: {source_info.name}
- Sector: {source_info.sector or "(unknown)"}
- Industry: {source_info.industry or "(unknown)"}
- Description: {truncate_text(source_info.description, 400)}

## Target Company (the resolved match)
- Ticker: {target_info.ticker}
- Name: {target_info.name}
- Sector: {target_info.sector or "(unknown)"}
- Industry: {target_info.industry or "(unknown)"}
- Description: {truncate_text(target_info.description, 400)}

## The Match to Verify
- Relationship Type: {relationship_type} (source mentions target as a {relationship_type_simple})
- Raw Mention in Filing: "{raw_mention}"
- Surrounding Context: "{truncate_text(context, 600)}"

Is this match CORRECT, INCORRECT, or AMBIGUOUS? Analyze carefully, then submit your verdict."""


def get_company_info(session: Any, ticker: str) -> CompanyInfo | None:
    """Fetch company information from Neo4j (sync)."""
    result = session.run(
        """
        MATCH (c:Company {ticker: $ticker})
        RETURN c.ticker AS ticker,
               c.name AS name,
               c.description AS description,
               c.sector AS sector,
               c.industry AS industry,
               c.sic_code AS sic_code
        """,
        ticker=ticker,
    )
    record = result.single()
    if record:
        return CompanyInfo(
            ticker=record["ticker"],
            name=record["name"],
            description=record["description"],
            sector=record["sector"],
            industry=record["industry"],
            sic_code=record["sic_code"],
        )
    return None


async def label_with_responses_api(
    client: AsyncOpenAI,
    prompt: str,
    model: str,
    reasoning_effort: str,
) -> LabelingResult:
    """Use the Responses API for gpt-5.2-pro with reasoning_effort."""
    response = await client.responses.create(
        model=model,
        input=prompt,
        tools=[RESPONSES_API_TOOL],
        tool_choice="required",
        reasoning={"effort": reasoning_effort},
        timeout=120.0,
    )

    # Find the function call in the output
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

    # Fallback if no function call found
    output_text = response.output_text[:500] if response.output_text else "empty"
    return LabelingResult(
        label="ambiguous",
        confidence=0.0,
        reasoning=f"No function call in response: {output_text}",
        business_logic_check="",
        alternative_interpretation=None,
    )


async def label_with_chat_completions(
    client: AsyncOpenAI,
    user_prompt: str,
    model: str,
) -> LabelingResult:
    """Use the Chat Completions API for other models."""
    request_params: dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        "tools": [LABELING_FUNCTION],
        "tool_choice": {
            "type": "function",
            "function": {"name": "submit_entity_resolution_label"},
        },
    }

    # For non-reasoning models, use temperature=0 for determinism
    if not model.startswith(("o1", "o3", "o4")):
        request_params["temperature"] = 0

    response = await client.chat.completions.create(**request_params, timeout=120.0)

    # Extract function call arguments
    message = response.choices[0].message
    if message.tool_calls and len(message.tool_calls) > 0:
        tool_call = message.tool_calls[0]
        if tool_call.function.name == "submit_entity_resolution_label":
            args = json.loads(tool_call.function.arguments)
            return LabelingResult(
                label=args.get("label", "ambiguous"),
                confidence=float(args.get("confidence", 0.5)),
                reasoning=args.get("reasoning", ""),
                business_logic_check=args.get("business_logic_check", ""),
                alternative_interpretation=args.get("alternative_interpretation"),
            )

    # Fallback
    if message.content:
        return LabelingResult(
            label="ambiguous",
            confidence=0.0,
            reasoning=f"No structured response: {message.content[:500]}",
            business_logic_check="",
            alternative_interpretation=None,
        )

    return LabelingResult(
        label="ambiguous",
        confidence=0.0,
        reasoning="Empty response from model",
        business_logic_check="",
        alternative_interpretation=None,
    )


async def label_single_record(
    client: AsyncOpenAI,
    record: dict[str, Any],
    source_info: CompanyInfo,
    target_info: CompanyInfo,
    model: str,
    reasoning_effort: str,
    semaphore: asyncio.Semaphore,
    progress: ProgressTracker,
) -> dict[str, Any]:
    """Label a single record (with concurrency control)."""
    async with semaphore:
        source_ticker = record.get("source_ticker", "")
        target_ticker = record.get("target_ticker", "")

        try:
            user_prompt = build_user_prompt(
                source_info=source_info,
                target_info=target_info,
                relationship_type=record.get("relationship_type", ""),
                raw_mention=record.get("raw_mention", ""),
                context=record.get("context", ""),
            )

            # Choose API based on model
            if model == "gpt-5.2-pro":
                full_prompt = f"{SYSTEM_PROMPT}\n\n{user_prompt}"
                result = await label_with_responses_api(
                    client, full_prompt, model, reasoning_effort
                )
            else:
                result = await label_with_chat_completions(client, user_prompt, model)

            # Add results to record
            record["ai_label"] = result.label
            record["ai_confidence"] = f"{result.confidence:.2f}"
            record["ai_reasoning"] = result.reasoning[:1000]
            record["ai_business_logic"] = result.business_logic_check[:500]
            record["ai_alternative"] = result.alternative_interpretation or ""

            await progress.record(result.label)

            # Log with emoji
            emoji = {"correct": "✓", "incorrect": "✗", "ambiguous": "?"}
            logger.info(
                f"{emoji.get(result.label, '?')} {source_ticker}→{target_ticker}: "
                f"{result.label} ({result.confidence:.0%}) | {progress.summary()}"
            )

        except Exception as e:
            logger.error(f"Error processing {source_ticker}→{target_ticker}: {e}")
            record["ai_label"] = "ambiguous"
            record["ai_confidence"] = "0.0"
            record["ai_reasoning"] = f"Error: {e}"
            record["ai_business_logic"] = ""
            record["ai_alternative"] = ""
            await progress.record("ambiguous", is_error=True)

        return record


async def process_ground_truth(
    input_path: Path,
    output_path: Path,
    model: str = "gpt-5.2-pro",
    reasoning_effort: str = "high",
    limit: int | None = None,
    concurrency: int = 10,
    resume: bool = False,
) -> None:
    """
    Process ground truth CSV and add AI labels (parallel).

    Args:
        input_path: Input CSV with ground truth records
        output_path: Output CSV with AI labels added
        model: OpenAI model to use
        reasoning_effort: For gpt-5.2-pro, controls compute (low/medium/high/xhigh)
        limit: Max records to process (for testing)
        concurrency: Max concurrent API calls
        resume: Skip already-processed records
    """
    # Connect to Neo4j for company lookups (sync)
    from public_company_graph.neo4j.connection import get_neo4j_driver

    driver = get_neo4j_driver()
    client = AsyncOpenAI()

    # Read input CSV
    with open(input_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        records = list(reader)

    if limit:
        records = records[:limit]

    # Check for existing results to resume from
    already_processed: set[str] = set()
    existing_results: list[dict[str, Any]] = []
    if resume and output_path.exists():
        with open(output_path, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                key = (
                    f"{row.get('source_ticker', '')}→"
                    f"{row.get('target_ticker', '')}→"
                    f"{row.get('relationship_type', '')}"
                )
                already_processed.add(key)
                existing_results.append(row)
        if already_processed:
            logger.info(f"📋 Resuming: Skipping {len(already_processed)} already-processed records")

    # Filter to records that need processing
    records_to_process = []
    for record in records:
        key = (
            f"{record.get('source_ticker', '')}→"
            f"{record.get('target_ticker', '')}→"
            f"{record.get('relationship_type', '')}"
        )
        if key not in already_processed:
            records_to_process.append(record)

    if not records_to_process:
        logger.info("✓ All records already processed!")
        return

    logger.info(f"🚀 Processing {len(records_to_process)} records with {model}")
    logger.info(f"   Concurrency: {concurrency} parallel requests")
    if model == "gpt-5.2-pro":
        logger.info(f"   Reasoning effort: {reasoning_effort}")

    # Prefetch company info (sync, before async work)
    logger.info("📥 Fetching company information from Neo4j...")
    company_cache: dict[str, CompanyInfo | None] = {}
    with driver.session(database="domain") as session:
        tickers_needed = set()
        for record in records_to_process:
            tickers_needed.add(record.get("source_ticker", ""))
            tickers_needed.add(record.get("target_ticker", ""))

        for ticker in tickers_needed:
            if ticker and ticker not in company_cache:
                company_cache[ticker] = get_company_info(session, ticker)

    logger.info(f"   Cached {len(company_cache)} companies")

    # Prepare tasks
    semaphore = asyncio.Semaphore(concurrency)
    progress = ProgressTracker(total=len(records_to_process))

    tasks = []
    for record in records_to_process:
        source_ticker = record.get("source_ticker", "")
        target_ticker = record.get("target_ticker", "")
        source_info = company_cache.get(source_ticker)
        target_info = company_cache.get(target_ticker)

        if not source_info or not target_info:
            # Handle missing company info
            record["ai_label"] = "ambiguous"
            record["ai_confidence"] = "0.0"
            record["ai_reasoning"] = "Could not find company information"
            record["ai_business_logic"] = ""
            record["ai_alternative"] = ""
            existing_results.append(record)
            continue

        task = label_single_record(
            client=client,
            record=record.copy(),  # Copy to avoid mutation issues
            source_info=source_info,
            target_info=target_info,
            model=model,
            reasoning_effort=reasoning_effort,
            semaphore=semaphore,
            progress=progress,
        )
        tasks.append(task)

    # Run all tasks in parallel
    logger.info(f"🔄 Starting {len(tasks)} parallel labeling tasks...")
    start_time = time.time()

    results = await asyncio.gather(*tasks, return_exceptions=True)

    elapsed = time.time() - start_time
    logger.info(f"⏱️  Completed in {elapsed:.1f}s ({len(tasks) / elapsed:.1f} records/sec)")

    # Collect successful results
    for result in results:
        if isinstance(result, Exception):
            logger.error(f"Task failed with exception: {result}")
        elif isinstance(result, dict):
            existing_results.append(result)

    # Write all results to output file
    if existing_results:
        fieldnames = list(existing_results[0].keys())
        # Ensure AI columns are present
        for col in [
            "ai_label",
            "ai_confidence",
            "ai_reasoning",
            "ai_business_logic",
            "ai_alternative",
        ]:
            if col not in fieldnames:
                fieldnames.append(col)

        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for record in existing_results:
                writer.writerow(record)

        logger.info(f"\n✓ Wrote {len(existing_results)} records to {output_path}")

        # Summary
        correct = sum(1 for r in existing_results if r.get("ai_label") == "correct")
        incorrect = sum(1 for r in existing_results if r.get("ai_label") == "incorrect")
        ambiguous = sum(1 for r in existing_results if r.get("ai_label") == "ambiguous")
        total = correct + incorrect + ambiguous

        if total > 0:
            logger.info("\n📊 Summary:")
            logger.info(f"   Correct:   {correct} ({100 * correct / total:.1f}%)")
            logger.info(f"   Incorrect: {incorrect} ({100 * incorrect / total:.1f}%)")
            logger.info(f"   Ambiguous: {ambiguous} ({100 * ambiguous / total:.1f}%)")

            if incorrect > 0 and correct + incorrect > 0:
                precision = correct / (correct + incorrect)
                logger.info(f"   Estimated Precision: {precision:.1%}")

    driver.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="AI-assisted labeling for entity resolution ground truth (parallel)"
    )
    parser.add_argument(
        "--input",
        "-i",
        type=Path,
        default=Path("data/er_ground_truth.csv"),
        help="Input ground truth CSV",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=Path("data/er_ground_truth_labeled.csv"),
        help="Output labeled CSV",
    )
    parser.add_argument(
        "--model",
        "-m",
        type=str,
        default="gpt-5.2-pro",
        choices=[
            "gpt-5.2-pro",  # Best accuracy - uses Responses API
            "gpt-5.2",  # Good balance - uses Chat Completions
            "o3",  # Reasoning model
            "o4-mini",  # Fast reasoning model
            "gpt-4o",  # Reliable fallback
        ],
        help="Model to use (gpt-5.2-pro recommended for maximum accuracy)",
    )
    parser.add_argument(
        "--reasoning-effort",
        "-r",
        type=str,
        default="high",
        choices=["low", "medium", "high", "xhigh"],
        help="Reasoning effort for gpt-5.2-pro (higher = more accurate, slower)",
    )
    parser.add_argument(
        "--limit",
        "-n",
        type=int,
        default=None,
        help="Limit number of relationships to process (for testing)",
    )
    parser.add_argument(
        "--concurrency",
        "-c",
        type=int,
        default=10,
        help="Max concurrent API calls (default: 10, stay under rate limits)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from where we left off (skip already-processed records)",
    )
    args = parser.parse_args()

    if not args.input.exists():
        logger.error(f"Input file not found: {args.input}")
        logger.info("Run: python scripts/create_er_ground_truth.py first")
        return

    asyncio.run(
        process_ground_truth(
            input_path=args.input,
            output_path=args.output,
            model=args.model,
            reasoning_effort=args.reasoning_effort,
            limit=args.limit,
            concurrency=args.concurrency,
            resume=args.resume,
        )
    )


if __name__ == "__main__":
    main()

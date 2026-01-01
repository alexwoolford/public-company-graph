#!/usr/bin/env python3
"""
AI-assisted labeling for entity resolution ground truth.

Uses OpenAI's best models for careful, methodical analysis of each
relationship to determine if the entity resolution was correct.

Model recommendations (per OpenAI, December 2025):
- gpt-5.2-pro: Best for accuracy - uses Responses API with reasoning_effort (default)
- gpt-5.2: Good balance - supports function calling via Chat Completions
- o3: Reasoning model that "thinks" carefully before responding
- o4-mini: Faster reasoning model for quicker results

Key features:
1. Rich context: Pulls company descriptions, industries, and 10-K context
2. Function calling: Structured output via tools (gpt-5.2-pro compatible)
3. Reasoning effort: Configurable compute for harder decisions (low/medium/high/xhigh)
4. Conservative: When uncertain, labels as "ambiguous" rather than guessing

Usage:
    # Default: gpt-5.2-pro with high reasoning effort (most accurate)
    python scripts/label_er_with_ai.py

    # Maximum rigor with xhigh reasoning effort
    python scripts/label_er_with_ai.py --reasoning-effort xhigh

    # Faster with gpt-5.2 (still accurate, uses Chat Completions)
    python scripts/label_er_with_ai.py -m gpt-5.2

    # Quick test with 5 samples
    python scripts/label_er_with_ai.py -n 5

Cost estimate: gpt-5.2-pro is ~$0.10-0.20 per relationship (higher quality)
"""

import argparse
import csv
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

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


# Function definition for structured output via function calling
# This works with gpt-5.2-pro which doesn't support Structured Outputs
LABELING_FUNCTION = {
    "type": "function",
    "function": {
        "name": "submit_entity_resolution_label",
        "description": "Submit your analysis and label for this entity resolution case",
        "parameters": {
            "type": "object",
            "properties": {
                "label": {
                    "type": "string",
                    "enum": ["correct", "incorrect", "ambiguous"],
                    "description": "Your verdict: 'correct' if the match is right, 'incorrect' if wrong, 'ambiguous' if uncertain",
                },
                "confidence": {
                    "type": "number",
                    "minimum": 0,
                    "maximum": 1,
                    "description": "Your confidence in this label (0.0 to 1.0)",
                },
                "reasoning": {
                    "type": "string",
                    "description": "Your detailed step-by-step reasoning for this decision",
                },
                "business_logic_check": {
                    "type": "string",
                    "description": "Does this relationship make business sense? Explain why or why not.",
                },
                "alternative_interpretation": {
                    "type": "string",
                    "description": "If incorrect, what might the raw mention actually refer to? Leave empty if correct/ambiguous.",
                },
            },
            "required": [
                "label",
                "confidence",
                "reasoning",
                "business_logic_check",
            ],
        },
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


def get_company_info(session, ticker: str) -> CompanyInfo | None:
    """Fetch company information from Neo4j."""
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


def truncate_text(text: str | None, max_length: int = 500) -> str:
    """Truncate text to reasonable length."""
    if not text:
        return "(not available)"
    if len(text) <= max_length:
        return text
    return text[:max_length] + "..."


def label_relationship_with_ai(
    client: OpenAI,
    source_info: CompanyInfo,
    target_info: CompanyInfo,
    relationship_type: str,
    raw_mention: str,
    context: str,
    model: str = "gpt-5.2-pro",
    reasoning_effort: str = "high",
) -> LabelingResult:
    """
    Use AI to carefully label a single relationship.

    Uses:
    - Responses API for gpt-5.2-pro (with reasoning_effort parameter)
    - Chat Completions API for other models
    """
    user_prompt = build_user_prompt(
        source_info=source_info,
        target_info=target_info,
        relationship_type=relationship_type,
        raw_mention=raw_mention,
        context=context,
    )

    full_prompt = f"{SYSTEM_PROMPT}\n\n{user_prompt}"

    try:
        # gpt-5.2-pro uses the Responses API
        if model == "gpt-5.2-pro":
            return _label_with_responses_api(client, full_prompt, model, reasoning_effort)
        else:
            return _label_with_chat_completions(client, user_prompt, model)

    except Exception as e:
        logger.error(f"API error: {e}")
        return LabelingResult(
            label="ambiguous",
            confidence=0.0,
            reasoning=f"API error: {e}",
            business_logic_check="",
            alternative_interpretation=None,
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


def _label_with_responses_api(
    client: OpenAI, prompt: str, model: str, reasoning_effort: str
) -> LabelingResult:
    """Use the Responses API for gpt-5.2-pro with reasoning_effort."""
    response = client.responses.create(
        model=model,
        input=prompt,
        tools=[RESPONSES_API_TOOL],
        tool_choice="required",
        reasoning={"effort": reasoning_effort},
        timeout=120.0,  # 2 minute timeout to prevent hanging
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
    return LabelingResult(
        label="ambiguous",
        confidence=0.0,
        reasoning=f"No function call in response: {response.output_text[:500] if response.output_text else 'empty'}",
        business_logic_check="",
        alternative_interpretation=None,
    )


def _label_with_chat_completions(client: OpenAI, user_prompt: str, model: str) -> LabelingResult:
    """Use the Chat Completions API for other models."""
    request_params = {
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

    response = client.chat.completions.create(**request_params, timeout=120.0)

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
        logger.warning("No tool call in response")
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


def process_ground_truth(
    input_path: Path,
    output_path: Path,
    model: str = "gpt-5.2-pro",
    reasoning_effort: str = "high",
    limit: int | None = None,
    delay: float = 1.0,
    resume: bool = False,
) -> None:
    """
    Process ground truth CSV and add AI labels.

    If resume=True, skips records that already exist in the output file.
    """
    # Connect to Neo4j for company lookups
    from public_company_graph.neo4j.connection import get_neo4j_driver

    driver = get_neo4j_driver()
    client = OpenAI()  # Uses OPENAI_API_KEY env var

    # Read input CSV
    with open(input_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        records = list(reader)

    if limit:
        records = records[:limit]

    # Check for existing results to resume from
    already_processed = set()
    if resume and output_path.exists():
        with open(output_path, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Create a unique key for each record
                key = f"{row.get('source_ticker', '')}→{row.get('target_ticker', '')}→{row.get('relationship_type', '')}"
                already_processed.add(key)
        if already_processed:
            logger.info(
                f"📋 Resuming: Found {len(already_processed)} already-processed records to skip"
            )

    logger.info(f"Processing {len(records)} relationships with {model}")
    if model == "gpt-5.2-pro":
        logger.info(f"Using Responses API with reasoning_effort={reasoning_effort}")
    elif model.startswith(("o3", "o4")):
        logger.info("Using reasoning model - will think carefully before answering")

    # Prepare output file with header
    fieldnames = list(records[0].keys()) + [
        "ai_label",
        "ai_confidence",
        "ai_reasoning",
        "ai_business_logic",
        "ai_alternative",
    ]

    # If resuming and file exists, don't overwrite header
    if not (resume and output_path.exists() and already_processed):
        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

    results = []
    skipped = 0
    with driver.session(database="domain") as session:
        for i, record in enumerate(records, 1):
            source_ticker = record.get("source_ticker", "")
            target_ticker = record.get("target_ticker", "")
            relationship_type = record.get("relationship_type", "")

            # Skip if already processed (resume mode)
            record_key = f"{source_ticker}→{target_ticker}→{relationship_type}"
            if record_key in already_processed:
                skipped += 1
                continue

            logger.info(f"[{i}/{len(records)}] Analyzing: {source_ticker} → {target_ticker}")

            # Get company info
            source_info = get_company_info(session, source_ticker)
            target_info = get_company_info(session, target_ticker)

            if not source_info or not target_info:
                logger.warning("  Could not find company info, skipping")
                record["ai_label"] = "ambiguous"
                record["ai_confidence"] = "0.0"
                record["ai_reasoning"] = "Could not find company information"
                record["ai_business_logic"] = ""
                record["ai_alternative"] = ""
                results.append(record)
                continue

            # Call AI for labeling
            result = label_relationship_with_ai(
                client=client,
                source_info=source_info,
                target_info=target_info,
                relationship_type=record.get("relationship_type", ""),
                raw_mention=record.get("raw_mention", ""),
                context=record.get("context", ""),
                model=model,
                reasoning_effort=reasoning_effort,
            )

            # Add results to record
            record["ai_label"] = result.label
            record["ai_confidence"] = f"{result.confidence:.2f}"
            record["ai_reasoning"] = result.reasoning[:1000]  # Truncate for CSV
            record["ai_business_logic"] = result.business_logic_check[:500]
            record["ai_alternative"] = result.alternative_interpretation or ""

            # Log result
            emoji = {"correct": "✓", "incorrect": "✗", "ambiguous": "?"}
            logger.info(
                f"  {emoji.get(result.label, '?')} {result.label} "
                f"(confidence: {result.confidence:.2f})"
            )
            if result.label == "incorrect":
                logger.info(f"    Alternative: {result.alternative_interpretation}")

            results.append(record)

            # Write this record immediately (incremental save)
            with open(output_path, "a", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writerow(record)

            # Rate limiting
            if delay > 0 and i < len(records):
                time.sleep(delay)

    # Summary (file was written incrementally)
    if results or skipped:
        logger.info(f"\n✓ Wrote {len(results)} new records to {output_path}")
        if skipped:
            logger.info(f"   (Skipped {skipped} already-processed records)")

        # Summary
        correct = sum(1 for r in results if r["ai_label"] == "correct")
        incorrect = sum(1 for r in results if r["ai_label"] == "incorrect")
        ambiguous = sum(1 for r in results if r["ai_label"] == "ambiguous")

        logger.info("\n📊 Summary:")
        logger.info(f"   Correct:   {correct} ({100 * correct / len(results):.1f}%)")
        logger.info(f"   Incorrect: {incorrect} ({100 * incorrect / len(results):.1f}%)")
        logger.info(f"   Ambiguous: {ambiguous} ({100 * ambiguous / len(results):.1f}%)")

        if incorrect > 0:
            precision = correct / (correct + incorrect)
            logger.info(f"   Estimated Precision: {precision:.1%}")

    driver.close()


def main():
    parser = argparse.ArgumentParser(
        description="AI-assisted labeling for entity resolution ground truth"
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
            "gpt-5.2-pro",  # Best accuracy (per OpenAI) - uses Responses API
            "gpt-5.2",  # Good balance - uses Chat Completions
            "o3",  # Reasoning model - thinks carefully
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
        "--delay",
        "-d",
        type=float,
        default=0.5,
        help="Delay between API calls in seconds",
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

    process_ground_truth(
        input_path=args.input,
        output_path=args.output,
        model=args.model,
        reasoning_effort=args.reasoning_effort,
        limit=args.limit,
        delay=args.delay,
        resume=args.resume,
    )


if __name__ == "__main__":
    main()

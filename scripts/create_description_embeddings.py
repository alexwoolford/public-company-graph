#!/usr/bin/env python3
"""
Create OpenAI embeddings for company descriptions and merge back with original data.

This script:
1. Reads public_company_domains.json
2. Creates embeddings for each description using OpenAI
3. Merges embeddings back into the original data (keyed by CIK)
4. Saves the enriched data back to the file

Usage:
    python scripts/create_description_embeddings.py                    # Dry-run (plan only)
    python scripts/create_description_embeddings.py --execute          # Actually create embeddings
    python scripts/create_description_embeddings.py --resume           # Resume from existing embeddings
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

from dotenv import load_dotenv
from tqdm import tqdm

# Load environment variables
load_dotenv()

# Try to import OpenAI
try:
    from openai import OpenAI

    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    print("WARNING: openai not installed. Install with: pip install openai")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()

# Default embedding model (text-embedding-3-small is cost-effective)
EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIMENSION = 1536  # Default for text-embedding-3-small

# Rate limiting: OpenAI allows 5000 requests/minute for embeddings
# We'll be conservative: 100 requests/second max
MAX_REQUESTS_PER_SECOND = 100
MIN_INTERVAL = 1.0 / MAX_REQUESTS_PER_SECOND


def create_embedding(
    client: OpenAI, text: str, model: str = EMBEDDING_MODEL
) -> Optional[List[float]]:
    """Create an embedding for a single text."""
    if not text or not text.strip():
        return None

    try:
        response = client.embeddings.create(model=model, input=text.strip())
        return response.data[0].embedding
    except Exception as e:
        logging.warning(f"Error creating embedding: {e}")
        return None


def batch_create_embeddings(
    client: OpenAI, texts: List[str], model: str = EMBEDDING_MODEL, batch_size: int = 100
) -> List[Optional[List[float]]]:
    """
    Create embeddings for multiple texts in batches.

    OpenAI embeddings API supports batch requests, but we'll process
    sequentially with rate limiting to be safe.
    """
    embeddings = []
    last_request_time = 0

    for text in texts:
        # Rate limiting
        current_time = time.time()
        elapsed = current_time - last_request_time
        if elapsed < MIN_INTERVAL:
            time.sleep(MIN_INTERVAL - elapsed)

        embedding = create_embedding(client, text, model)
        embeddings.append(embedding)
        last_request_time = time.time()

    return embeddings


def load_data(input_file: Path) -> List[Dict]:
    """Load the JSON data file."""
    with open(input_file) as f:
        return json.load(f)


def save_data(data: List[Dict], output_file: Path):
    """Save the enriched JSON data file."""
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(data, f, indent=2)


def create_embeddings_for_descriptions(
    data: List[Dict], embeddings_file: Optional[Path] = None, resume: bool = False
) -> Dict[str, List[float]]:
    """
    Create embeddings for all descriptions in the data.

    Returns a dictionary mapping CIK -> embedding vector.
    """
    if not OPENAI_AVAILABLE:
        raise ImportError("openai not available. Install with: pip install openai")

    if not OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY not set in .env file")

    client = OpenAI(api_key=OPENAI_API_KEY)

    # Load existing embeddings if resuming
    existing_embeddings = {}
    if resume and embeddings_file and embeddings_file.exists():
        try:
            with open(embeddings_file) as f:
                existing_embeddings = json.load(f)
            logging.info(f"Loaded {len(existing_embeddings)} existing embeddings")
        except Exception as e:
            logging.warning(f"Could not load existing embeddings: {e}")

    # Collect all descriptions that need embeddings
    to_process = []
    cik_to_index = {}

    for idx, entry in enumerate(data):
        cik = entry.get("cik")
        description = entry.get("description")

        if not cik:
            continue

        # Skip if already has embedding and we're resuming
        if resume and cik in existing_embeddings:
            continue

        # Only process entries with descriptions
        if description and description.strip():
            to_process.append((cik, description, idx))
            cik_to_index[cik] = idx

    if not to_process:
        logging.info("No descriptions to process")
        return existing_embeddings

    # Create embeddings
    embeddings = existing_embeddings.copy()

    # Process with progress bar (one item at a time for accurate progress)
    last_request_time = 0
    with tqdm(total=len(to_process), desc="Creating embeddings", unit="desc") as pbar:
        for cik, description, _ in to_process:
            # Rate limiting
            current_time = time.time()
            elapsed = current_time - last_request_time
            if elapsed < MIN_INTERVAL:
                time.sleep(MIN_INTERVAL - elapsed)

            embedding = create_embedding(client, description, EMBEDDING_MODEL)
            if embedding:
                embeddings[cik] = embedding

            last_request_time = time.time()
            pbar.update(1)

            # Save intermediate results periodically (every 100 embeddings)
            if len(embeddings) % 100 == 0 and embeddings_file:
                embeddings_file.parent.mkdir(parents=True, exist_ok=True)
                with open(embeddings_file, "w") as f:
                    json.dump(embeddings, f, indent=2)

    # Final save
    if embeddings_file:
        embeddings_file.parent.mkdir(parents=True, exist_ok=True)
        with open(embeddings_file, "w") as f:
            json.dump(embeddings, f, indent=2)
        logging.info(f"Saved embeddings to {embeddings_file}")

    return embeddings


def merge_embeddings_into_data(data: List[Dict], embeddings: Dict[str, List[float]]) -> List[Dict]:
    """Merge embeddings back into the original data."""
    enriched = []
    missing_count = 0

    for entry in data:
        cik = entry.get("cik")
        if cik and cik in embeddings:
            entry["description_embedding"] = embeddings[cik]
            entry["embedding_model"] = EMBEDDING_MODEL
            entry["embedding_dimension"] = EMBEDDING_DIMENSION
        elif entry.get("description"):
            # Has description but no embedding (error case)
            missing_count += 1

        enriched.append(entry)

    if missing_count > 0:
        logging.warning(f"{missing_count} entries with descriptions but no embeddings")

    return enriched


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Create OpenAI embeddings for company descriptions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually create embeddings (default is dry-run)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from existing embeddings file",
    )
    parser.add_argument(
        "--input-file",
        type=Path,
        default=Path("data/public_company_domains.json"),
        help="Input JSON file (default: data/public_company_domains.json)",
    )
    parser.add_argument(
        "--output-file",
        type=Path,
        default=None,
        help="Output JSON file (default: overwrites input file)",
    )
    parser.add_argument(
        "--embeddings-file",
        type=Path,
        default=Path("data/description_embeddings.json"),
        help="Separate embeddings file (default: data/description_embeddings.json)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=EMBEDDING_MODEL,
        help=f"OpenAI embedding model (default: {EMBEDDING_MODEL})",
    )

    args = parser.parse_args()

    # Set up logging
    log_dir = Path("logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"create_description_embeddings_{timestamp}.log"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            # Don't log to stdout - let tqdm handle progress display
        ],
    )

    # Suppress OpenAI's verbose HTTP logging
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)

    logger = logging.getLogger(__name__)

    logger.info("=" * 80)
    logger.info("Company Description Embeddings")
    logger.info(f"Log file: {log_file}")

    if not OPENAI_AVAILABLE:
        logger.error("openai not installed. Install with: pip install openai")
        sys.exit(1)

    if not OPENAI_API_KEY:
        logger.error("OPENAI_API_KEY not set in .env file")
        sys.exit(1)

    # Determine output file
    output_file = args.output_file or args.input_file

    # Load data
    logger.info(f"Loading data from {args.input_file}...")
    if not args.input_file.exists():
        logger.error(f"Input file not found: {args.input_file}")
        sys.exit(1)

    data = load_data(args.input_file)
    logger.info(f"Loaded {len(data)} entries")

    # Count entries with descriptions
    entries_with_descriptions = sum(1 for entry in data if entry.get("description"))
    logger.info(f"Found {entries_with_descriptions} entries with descriptions")

    # Dry-run mode
    if not args.execute:
        logger.info("=" * 80)
        logger.info("DRY RUN MODE")
        logger.info("=" * 80)
        logger.info(f"This script will:")
        logger.info(f"  1. Read {args.input_file}")
        logger.info(f"  2. Create embeddings for {entries_with_descriptions} descriptions")
        logger.info(f"  3. Use OpenAI model: {args.model}")
        logger.info(f"  4. Save embeddings to: {args.embeddings_file}")
        logger.info(f"  5. Merge embeddings into: {output_file}")
        logger.info("")
        logger.info("Estimated cost (text-embedding-3-small):")
        logger.info(
            f"  ~${entries_with_descriptions * 0.00002:.2f} for {entries_with_descriptions} embeddings"
        )
        logger.info("")
        logger.info("To execute, run: python scripts/create_description_embeddings.py --execute")
        logger.info("=" * 80)
        return

    # Execute mode
    logger.info("=" * 80)
    logger.info("EXECUTE MODE")
    logger.info("=" * 80)

    # Create embeddings
    embeddings = create_embeddings_for_descriptions(
        data, embeddings_file=args.embeddings_file, resume=args.resume
    )

    logger.info(f"Created/loaded {len(embeddings)} embeddings")

    # Merge embeddings into data
    logger.info("Merging embeddings into data...")
    enriched_data = merge_embeddings_into_data(data, embeddings)

    # Save enriched data
    logger.info(f"Saving enriched data to {output_file}...")
    save_data(enriched_data, output_file)

    # Statistics
    entries_with_embeddings = sum(
        1 for entry in enriched_data if entry.get("description_embedding")
    )
    logger.info("=" * 80)
    logger.info("Complete!")
    logger.info(f"  Total entries: {len(enriched_data)}")
    logger.info(f"  Entries with descriptions: {entries_with_descriptions}")
    logger.info(f"  Entries with embeddings: {entries_with_embeddings}")
    logger.info(f"  Saved to: {output_file}")
    if args.embeddings_file:
        logger.info(f"  Embeddings cache: {args.embeddings_file}")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()

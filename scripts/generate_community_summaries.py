#!/usr/bin/env python3
"""
Generate AI summaries for Community nodes.

This script:
1. Collects all company descriptions for each community
2. Uses OpenAI to generate a concise summary of the community
3. Updates Community nodes with the summary
4. Supports resume functionality and rate limiting
"""

import os
import sys
import json
import time
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from neo4j import GraphDatabase
from openai import OpenAI
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()

NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "")
NEO4J_DATABASE = os.getenv("NEO4J_DATABASE", "neo4j")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# OpenAI configuration
# Note: GPT-5 models appear to have issues returning content, so using GPT-4o-mini
# GPT-4o-mini: Proven to work, very affordable ($0.30 total for 80 communities)
# GPT-4o: Better quality option ($5.07 total) - uncomment to use instead
MODEL = "gpt-4o-mini"  # Recommended: proven to work, very affordable
# MODEL = "gpt-4o"  # Alternative: better quality, more expensive
MAX_TOKENS = 200  # Keep summaries concise (use max_tokens for GPT-4 models)
MIN_INTERVAL = 0.5  # Rate limiting

# Maximum description length per company (to avoid token limits)
MAX_DESCRIPTION_LENGTH = 500


def get_community_descriptions(driver, community_id: int, database: str = None) -> List[str]:
    """Get all company descriptions for a community."""
    with driver.session(database=database) as session:
        result = session.run("""
            MATCH (comm:Community {communityId: $comm_id})<-[:BELONGS_TO_COMMUNITY]-(c:Company)
            WHERE c.description IS NOT NULL AND size(c.description) > 20
            RETURN c.description AS description
            ORDER BY c.ticker
        """, comm_id=community_id)
        
        descriptions = []
        for record in result:
            desc = record["description"]
            # Truncate very long descriptions
            if len(desc) > MAX_DESCRIPTION_LENGTH:
                desc = desc[:MAX_DESCRIPTION_LENGTH] + "..."
            descriptions.append(desc)
        
        return descriptions


def generate_community_summary(client: OpenAI, descriptions: List[str]) -> Optional[str]:
    """Generate a summary for a community using OpenAI."""
    if not descriptions:
        return None
    
    # Combine descriptions with separators
    combined_text = "\n\n---\n\n".join(descriptions)
    
    # Limit total input to avoid token limits (rough estimate: 1 token ≈ 4 chars)
    max_input_chars = 8000  # ~2000 tokens, leaving room for prompt and response
    if len(combined_text) > max_input_chars:
        # Take first N descriptions that fit
        truncated = ""
        for desc in descriptions:
            if len(truncated) + len(desc) + 10 > max_input_chars:
                break
            truncated += desc + "\n\n---\n\n"
        combined_text = truncated.rstrip("\n\n---\n\n")
    
    prompt = f"""You are analyzing a group of companies that have been clustered together based on similar business descriptions using machine learning.

Here are the business descriptions of companies in this group:

{combined_text}

Please provide a concise, 2-3 sentence summary that:
1. Identifies the common business focus, industry sector, or theme shared by these companies
2. Describes what makes them similar (products, services, markets, business models)
3. Avoids listing individual company names or details

Focus on synthesizing the shared characteristics into a clear, informative summary.

Summary:"""
    
    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": "You are a business analyst who creates concise summaries of company groups."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=MAX_TOKENS,
            temperature=0.2  # Lower temperature for more consistent, focused summaries
        )
        
        summary = response.choices[0].message.content.strip()
        return summary
    except Exception as e:
        print(f"   ✗ Error generating summary: {e}")
        return None


def update_community_summary(driver, community_id: int, summary: str, database: str = None):
    """Update a Community node with its summary."""
    with driver.session(database=database) as session:
        session.run("""
            MATCH (comm:Community {communityId: $comm_id})
            SET comm.summary = $summary,
                comm.summaryModel = $model,
                comm.summaryGeneratedAt = datetime()
        """, comm_id=community_id, summary=summary, model=MODEL)


def generate_all_summaries(
    driver,
    client: OpenAI,
    database: str = None,
    resume: bool = False,
    progress_file: Optional[Path] = None
) -> Dict[int, str]:
    """Generate summaries for all communities."""
    print("=" * 80)
    print("Generating Community Summaries")
    print("=" * 80)
    
    # Get all communities
    with driver.session(database=database) as session:
        result = session.run("""
            MATCH (comm:Community)
            RETURN comm.communityId AS communityId, comm.size AS size,
                   comm.summary AS existingSummary
            ORDER BY comm.size DESC
        """)
        
        communities = [(r["communityId"], r["size"], r["existingSummary"]) for r in result]
    
    print(f"\nTotal communities: {len(communities)}")
    
    # Load progress if resuming
    completed = set()
    if resume and progress_file and progress_file.exists():
        with open(progress_file, "r") as f:
            progress_data = json.load(f)
            completed = set(progress_data.get("completed", []))
        print(f"Resuming: {len(completed)} communities already have summaries")
    
    summaries = {}
    last_request_time = 0
    
    # Process communities
    with tqdm(total=len(communities), desc="Generating summaries", unit="comm") as pbar:
        for community_id, size, existing_summary in communities:
            # Skip if already has summary and we're resuming
            if resume and existing_summary:
                completed.add(community_id)
                summaries[community_id] = existing_summary
                pbar.update(1)
                continue
            
            # Rate limiting
            current_time = time.time()
            elapsed = current_time - last_request_time
            if elapsed < MIN_INTERVAL:
                time.sleep(MIN_INTERVAL - elapsed)
            
            # Get descriptions
            descriptions = get_community_descriptions(driver, community_id, database)
            
            if not descriptions:
                print(f"\n   ⚠ Community {community_id} ({size} companies): No descriptions found")
                pbar.update(1)
                continue
            
            # Generate summary
            summary = generate_community_summary(client, descriptions)
            
            if summary:
                # Update database
                update_community_summary(driver, community_id, summary, database)
                summaries[community_id] = summary
                completed.add(community_id)
                
                # Save progress
                if progress_file:
                    progress_file.parent.mkdir(parents=True, exist_ok=True)
                    with open(progress_file, "w") as f:
                        json.dump({"completed": list(completed), "summaries": summaries}, f, indent=2)
                
                pbar.set_postfix({"last": f"Comm {community_id}"})
            else:
                print(f"\n   ✗ Failed to generate summary for Community {community_id}")
            
            last_request_time = time.time()
            pbar.update(1)
    
    print(f"\n✓ Generated {len(summaries)} summaries")
    return summaries


def show_sample_summaries(driver, database: str = None, limit: int = 5):
    """Show sample community summaries."""
    print("\n" + "=" * 80)
    print("Sample Community Summaries")
    print("=" * 80)
    
    with driver.session(database=database) as session:
        result = session.run("""
            MATCH (comm:Community)
            WHERE comm.summary IS NOT NULL
            RETURN comm.communityId AS communityId, comm.size AS size,
                   comm.summary AS summary, comm.avgSimilarity AS avgSimilarity
            ORDER BY comm.size DESC
            LIMIT $limit
        """, limit=limit)
        
        for i, record in enumerate(result, 1):
            comm_id = record["communityId"]
            size = record["size"]
            summary = record["summary"]
            avg_sim = record["avgSimilarity"]
            
            print(f"\n{i}. Community {comm_id} ({size} companies, avg similarity: {avg_sim:.4f})")
            print(f"   {summary}")


def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Generate AI summaries for Community nodes"
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Execute the generation (default is dry-run)"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from previous run (skip communities that already have summaries)"
    )
    
    args = parser.parse_args()
    
    if not args.execute:
        print("=" * 80)
        print("DRY RUN - Community Summary Generation")
        print("=" * 80)
        print("\nWould:")
        print("1. Get all company descriptions for each community")
        print("2. Use OpenAI to generate concise summaries")
        print("3. Update Community nodes with summaries")
        print("\nUse --execute to actually generate summaries")
        return
    
    if not OPENAI_API_KEY:
        print("✗ Error: OPENAI_API_KEY not set in environment")
        sys.exit(1)
    
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    client = OpenAI(api_key=OPENAI_API_KEY)
    
    progress_file = Path("data/community_summaries_progress.json")
    
    try:
        summaries = generate_all_summaries(
            driver,
            client,
            NEO4J_DATABASE,
            resume=args.resume,
            progress_file=progress_file
        )
        
        show_sample_summaries(driver, NEO4J_DATABASE, limit=10)
        
        print("\n" + "=" * 80)
        print("✓ Complete!")
        print("=" * 80)
        
    finally:
        driver.close()


if __name__ == "__main__":
    main()


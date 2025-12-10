#!/usr/bin/env python3
"""
Production-quality parallel domain collection with multi-source consensus.

This script collects company domains from multiple sources concurrently,
uses weighted voting to determine the correct domain, and stops early when
confidence is high. Designed for speed, accuracy, and cost efficiency.

Architecture:
- Multiple data sources executed in parallel (yfinance, Finviz, SEC, Finnhub)
- Weighted voting system (higher weight for more reliable sources)
- Early stopping when confidence threshold is met (2+ sources agree)
- Rate limiting per source with proper concurrency control
- Caching to avoid redundant API calls
- LLM adjudication only when sources disagree

Sources (in priority order):
1. yfinance (weight: 3) - Fast, reliable, good coverage
2. Finviz (weight: 2) - Fast, good coverage
3. SEC EDGAR (weight: 2) - Authoritative but slower
4. Finnhub (weight: 1) - Incomplete but can augment
"""

import asyncio
import json
import logging
import os
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Dict, List, Optional, Set, Tuple

import requests
from dotenv import load_dotenv
from tqdm import tqdm

# Load environment variables
load_dotenv()

# Try to import optional dependencies
try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False

try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY", "").strip()

# Global logger
_logger: Optional[logging.Logger] = None

# Thread-safe cache for domain validation
_domain_cache: Dict[str, bool] = {}
_cache_lock = Lock()

# Rate limiting locks per source
# Limits are conservative to avoid getting blocked, but optimized for throughput
# With 20 concurrent workers, each making 4 parallel API calls, we need per-source limits
_rate_limits = {
    "sec": {
        "lock": Lock(), 
        "last_call": 0, 
        "min_interval": 1.0 / 10.0  # SEC EDGAR official limit: 10 req/sec
    },
    "finviz": {
        "lock": Lock(), 
        "last_call": 0, 
        "min_interval": 1.0 / 5.0  # Finviz: No official API, web scraping. 5 req/sec is safe.
    },
    "finnhub": {
        "lock": Lock(), 
        "last_call": 0, 
        "min_interval": 1.0 / 1.0  # Finnhub free tier: 60 req/min = 1 req/sec
        # Paid tier can be higher, adjust if you have paid access
    },
    "yfinance": {
        "lock": Lock(), 
        "last_call": 0, 
        "min_interval": 0.0  # yfinance: No explicit limit, library handles throttling
    },
}


@dataclass
class DomainResult:
    """Result from a single data source."""
    domain: Optional[str]
    source: str
    confidence: float  # 0.0 to 1.0
    description: Optional[str] = None  # Company description from this source
    metadata: Dict = field(default_factory=dict)


@dataclass
class CompanyResult:
    """Final result for a company."""
    cik: str
    ticker: str
    name: str
    domain: Optional[str]
    sources: List[str]
    confidence: float
    votes: int
    all_candidates: Dict[str, List[str]]  # domain -> list of sources
    description: Optional[str] = None  # Company description (best available)
    description_source: Optional[str] = None  # Source of the description
    metadata: Dict = field(default_factory=dict)


def rate_limit(source: str):
    """Enforce rate limiting for a specific source."""
    if source not in _rate_limits:
        return
    
    limit = _rate_limits[source]
    with limit["lock"]:
        current_time = time.time()
        elapsed = current_time - limit["last_call"]
        if elapsed < limit["min_interval"]:
            time.sleep(limit["min_interval"] - elapsed)
        limit["last_call"] = time.time()


def normalize_domain(url: Optional[str]) -> Optional[str]:
    """Extract and normalize domain from URL."""
    if not url:
        return None
    
    try:
        from urllib.parse import urlparse
        parsed = urlparse(url if "://" in url else f"https://{url}")
        domain = parsed.netloc or parsed.path.split("/")[0]
        domain = domain.lower().replace("www.", "")
        domain = domain.split(":")[0].split("/")[0].split("?")[0]
        
        if "." in domain and len(domain) > 3:
            return domain
    except Exception:
        pass
    
    return None


def is_infrastructure_domain(domain: str) -> bool:
    """Check if domain is infrastructure (sec.gov, xbrl.org, etc.)."""
    import re
    # Specific infrastructure domains
    infrastructure_patterns = [
        r"sec\.gov",
        r"xbrl\.org",
        r"fasb\.org",
        r"gaap\.org",
        r"\.gov$",  # All .gov domains
    ]
    
    # Check patterns
    if any(re.search(pattern, domain, re.IGNORECASE) for pattern in infrastructure_patterns):
        return True
    
    # Known infrastructure domains
    known_infrastructure = {
        "w3.org", "xbrl.sec.gov", "sec.gov", "fasb.org", "gaap.org",
        "finviz.com", "yahoo.com", "google.com",  # Don't return these as company domains
    }
    
    return domain.lower() in known_infrastructure


def get_domain_from_yfinance(ticker: str, company_name: str = "") -> DomainResult:
    """Get domain and description from yfinance (high confidence source)."""
    if not YFINANCE_AVAILABLE:
        return DomainResult(None, "yfinance", 0.0)
    
    rate_limit("yfinance")
    
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        website = info.get("website")
        
        # Extract description (prefer longBusinessSummary, fallback to description)
        description = info.get("longBusinessSummary") or info.get("description")
        if description:
            # Clean up description: remove extra whitespace, limit length
            description = " ".join(description.split())
            if len(description) > 2000:  # Truncate very long descriptions
                description = description[:2000] + "..."
        
        if website:
            domain = normalize_domain(website)
            if domain and not is_infrastructure_domain(domain):
                return DomainResult(
                    domain, 
                    "yfinance", 
                    0.9, 
                    description=description,
                    metadata={"raw_website": website}
                )
    except Exception as e:
        if _logger:
            _logger.debug(f"yfinance error for {ticker}: {e}")
    
    return DomainResult(None, "yfinance", 0.0)


def get_domain_from_finviz(session: requests.Session, ticker: str) -> DomainResult:
    """Get domain from Finviz (medium confidence source)."""
    rate_limit("finviz")
    
    try:
        url = f"https://finviz.com/quote.ashx?t={ticker}"
        headers = {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        }
        response = session.get(url, headers=headers, timeout=5)
        
        if response.status_code == 200:
            import re
            # Finviz has website in a table: <td>Website</td><td><a href="https://www.company.com">Website</a></td>
            # More specific pattern to avoid catching Yahoo Finance links
            # Look for the Website label followed by a link that's NOT yahoo.com or finance.yahoo.com
            website_pattern = r'Website["\']?\s*</td>\s*<td[^>]*>\s*<a[^>]*href=["\'](https?://(?:www\.)?([a-z0-9]([a-z0-9-]{0,61}[a-z0-9])?(\.[a-z0-9]([a-z0-9-]{0,61}[a-z0-9])?)+))'
            match = re.search(website_pattern, response.text, re.IGNORECASE)
            if match:
                domain = normalize_domain(match.group(1))
                # Filter out infrastructure and known bad domains
                if (domain and 
                    not is_infrastructure_domain(domain) and 
                    "finviz.com" not in domain and
                    "yahoo.com" not in domain and
                    "google.com" not in domain):
                    return DomainResult(domain, "finviz", 0.7)
    except Exception as e:
        if _logger:
            _logger.debug(f"Finviz error for {ticker}: {e}")
    
    return DomainResult(None, "finviz", 0.0)


def get_domain_from_sec(session: requests.Session, cik: str, ticker: str, company_name: str) -> DomainResult:
    """Get domain from SEC EDGAR (authoritative but slower)."""
    rate_limit("sec")
    
    try:
        # Fetch SEC submission
        url = f"https://data.sec.gov/submissions/CIK{cik.zfill(10)}.json"
        headers = {
            "User-Agent": "domain_status_graph script (contact: your-email@example.com)",
            "Accept": "application/json",
        }
        response = session.get(url, headers=headers, timeout=10)
        
        if response.status_code == 200:
            submission = response.json()
            
            # Check website field first (fastest, most reliable)
            website = submission.get("website")
            if website:
                domain = normalize_domain(website)
                if domain and not is_infrastructure_domain(domain):
                    return DomainResult(domain, "sec_edgar", 0.85, {"field": "website"})
            
            # Fallback: check investor website (sometimes populated when website isn't)
            investor_website = submission.get("investorWebsite")
            if investor_website:
                domain = normalize_domain(investor_website)
                if domain and not is_infrastructure_domain(domain):
                    # Prefer main domain over investor relations subdomain
                    # e.g., "investor.apple.com" -> "apple.com"
                    if domain.startswith("investor."):
                        domain = domain.replace("investor.", "")
                    return DomainResult(domain, "sec_edgar", 0.75, {"field": "investorWebsite"})
    except Exception as e:
        if _logger:
            _logger.debug(f"SEC error for {ticker} (CIK {cik}): {e}")
    
    return DomainResult(None, "sec_edgar", 0.0)


def get_domain_from_finnhub(ticker: str) -> DomainResult:
    """Get domain and description from Finnhub (low confidence, incomplete coverage)."""
    if not FINNHUB_API_KEY:
        return DomainResult(None, "finnhub", 0.0)
    
    rate_limit("finnhub")
    
    try:
        url = "https://finnhub.io/api/v1/stock/profile2"
        params = {"symbol": ticker, "token": FINNHUB_API_KEY}
        response = requests.get(url, params=params, timeout=5)
        
        if response.status_code == 200:
            data = response.json()
            weburl = data.get("weburl")
            
            # Extract description if available (Finnhub may have finnhubIndustry or description)
            description = data.get("description") or data.get("finnhubIndustry")
            if description:
                description = " ".join(str(description).split())
                if len(description) > 2000:
                    description = description[:2000] + "..."
            
            if weburl:
                domain = normalize_domain(weburl)
                if domain and not is_infrastructure_domain(domain):
                    return DomainResult(
                        domain, 
                        "finnhub", 
                        0.6,
                        description=description
                    )
    except Exception as e:
        if _logger:
            _logger.debug(f"Finnhub error for {ticker}: {e}")
    
    return DomainResult(None, "finnhub", 0.0)


def collect_domains_parallel(
    session: requests.Session,
    cik: str,
    ticker: str,
    company_name: str,
    early_stop_confidence: float = 0.75,
) -> CompanyResult:
    """
    Collect domains from all sources in parallel with early stopping.
    
    Strategy:
    1. Launch all sources concurrently
    2. As results arrive, check for consensus
    3. Stop early if 2+ high-confidence sources agree (weighted score >= threshold)
    4. Use weighted voting to determine final domain
    5. LLM adjudication only when sources disagree
    
    Args:
        session: HTTP session
        cik: Company CIK
        ticker: Stock ticker
        company_name: Company name
        early_stop_confidence: Stop early if weighted confidence >= this (default 0.75)
    
    Returns:
        CompanyResult with domain, sources, confidence, etc.
    """
    # Source weights (higher = more reliable)
    source_weights = {
        "yfinance": 3.0,
        "sec_edgar": 2.5,
        "finviz": 2.0,
        "finnhub": 1.0,
    }
    
    # Execute all sources concurrently
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {
            executor.submit(get_domain_from_yfinance, ticker, company_name): "yfinance",
            executor.submit(get_domain_from_finviz, session, ticker): "finviz",
            executor.submit(get_domain_from_sec, session, cik, ticker, company_name): "sec",
            executor.submit(get_domain_from_finnhub, ticker): "finnhub",
        }
        
        results: List[DomainResult] = []
        domain_scores: Dict[str, float] = defaultdict(float)
        
        # Collect results as they complete, with early stopping
        # Each source has individual timeout, and we break early when confident
        for future in as_completed(futures):
            try:
                result = future.result(timeout=30)  # 30s per individual source
                if result.domain:
                    results.append(result)
                    
                    # Update scores (weighted by source reliability and result confidence)
                    weight = source_weights.get(result.source, 1.0)
                    domain_scores[result.domain] += weight * result.confidence
                    
                    # Early stopping: if we have high confidence, stop waiting
                    if domain_scores:
                        max_score = max(domain_scores.values())
                        max_possible = sum(source_weights.values())  # All sources agree
                        current_confidence = max_score / max_possible
                        
                        # If we have 2+ sources agreeing on same domain, we're confident
                        if len(results) >= 2:
                            domains_found = [r.domain for r in results]
                            unique_domains = set(domains_found)
                            
                            # All sources agree on same domain - very high confidence
                            if len(unique_domains) == 1:
                                # High confidence - stop waiting for other sources
                                break
                            
                            # Or if weighted confidence exceeds threshold
                            if current_confidence >= early_stop_confidence:
                                break
            except TimeoutError:
                if _logger:
                    _logger.debug(f"Timeout collecting domain for {ticker} from one source")
            except Exception as e:
                if _logger:
                    _logger.debug(f"Error collecting domain for {ticker}: {e}")
    
    if not results or not domain_scores:
        return CompanyResult(
            cik=cik,
            ticker=ticker,
            name=company_name,
            domain=None,
            sources=[],
            confidence=0.0,
            votes=0,
            all_candidates={},
            description=None,
            description_source=None,
        )
    
    # Build domain votes for reporting
    domain_votes: Dict[str, List[str]] = defaultdict(list)
    for result in results:
        if result.domain:
            domain_votes[result.domain].append(result.source)
    
    # Collect descriptions from all sources (weighted by source reliability)
    description_scores: Dict[str, Tuple[float, str]] = {}  # description -> (score, source)
    for result in results:
        if result.description:
            weight = source_weights.get(result.source, 1.0)
            # Use existing score if description already seen, otherwise add new
            if result.description in description_scores:
                description_scores[result.description] = (
                    description_scores[result.description][0] + weight * result.confidence,
                    description_scores[result.description][1]  # Keep first source
                )
            else:
                description_scores[result.description] = (weight * result.confidence, result.source)
    
    # Get winner (already calculated during early stopping)
    winner_domain = max(domain_scores.items(), key=lambda x: x[1])[0]
    winner_sources = domain_votes[winner_domain]
    total_score = domain_scores[winner_domain]
    
    # Get best description (highest weighted score)
    best_description = None
    best_description_source = None
    if description_scores:
        best_description, (_, best_description_source) = max(
            description_scores.items(), 
            key=lambda x: x[1][0]
        )
    
    # Calculate confidence: normalize by sources that actually responded
    # Sum of weights for sources that provided results (not just winner)
    sources_that_responded = set()
    for result in results:
        if result.domain:
            sources_that_responded.add(result.source)
    
    # Max possible score given the sources that actually responded
    max_possible_given_sources = sum(
        source_weights.get(source, 1.0) 
        for source in sources_that_responded
    )
    
    # Confidence: how much of the available sources agree on this domain?
    # If all responding sources agree: confidence = 1.0
    # If only some agree: confidence = their_score / max_possible_from_responders
    if max_possible_given_sources > 0:
        confidence = min(total_score / max_possible_given_sources, 1.0)
    else:
        confidence = 0.0
    
    # If sources disagree (multiple domains with votes), use LLM adjudication
    if len(domain_votes) > 1 and OPENAI_AVAILABLE:
        top_candidates = sorted(domain_scores.items(), key=lambda x: -x[1])[:3]
        candidate_domains = [d[0] for d in top_candidates]
        
        try:
            client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            prompt = f"""Given ticker {ticker} and company name "{company_name}", which is the correct company website domain?
            
Candidates:
{chr(10).join(f"- {d}" for d in candidate_domains)}

Respond with ONLY the domain name (e.g., "apple.com"), nothing else."""
            
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You identify correct company website domains."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=50
            )
            
            adjudicated = normalize_domain(response.choices[0].message.content.strip())
            if adjudicated in candidate_domains:
                winner_domain = adjudicated
                winner_sources = ["llm_adjudicated"] + winner_sources
                confidence = 0.95
        except Exception as e:
            if _logger:
                _logger.debug(f"LLM adjudication failed for {ticker}: {e}")
    
    return CompanyResult(
        cik=cik,
        ticker=ticker,
        name=company_name,
        domain=winner_domain,
        sources=winner_sources,
        confidence=confidence,
        votes=len(winner_sources),
        all_candidates={d: sources for d, sources in domain_votes.items()},
        description=best_description,
        description_source=best_description_source,
    )


def fetch_company_tickers(session: requests.Session) -> Dict[str, Dict[str, str]]:
    """Fetch all company tickers from SEC EDGAR."""
    url = "https://www.sec.gov/files/company_tickers.json"
    headers = {
        "User-Agent": "domain_status_graph script (contact: your-email@example.com)",
        "Accept": "application/json",
    }
    
    response = session.get(url, headers=headers, timeout=30)
    response.raise_for_status()
    data = response.json()
    
    companies = {}
    for entry in data.values():
        cik = str(entry.get("cik_str", "")).zfill(10)
        if cik:
            companies[cik] = {
                "cik": cik,
                "ticker": entry.get("ticker", "").upper(),
                "name": entry.get("title", "").strip(),
            }
    
    return companies


def process_company(
    session: requests.Session,
    cik: str,
    company_info: Dict[str, str],
    existing_results: Dict[str, Dict],
) -> Optional[Dict]:
    """Process a single company and return result."""
    ticker = company_info["ticker"]
    name = company_info["name"]
    
    # Skip if already processed
    if cik in existing_results:
        return existing_results[cik]
    
    result = collect_domains_parallel(session, cik, ticker, name)
    
    if result.domain:
        output = {
            "cik": cik,
            "ticker": ticker,
            "name": name,
            "domain": result.domain,
            "source": "+".join(sorted(set(result.sources))),
            "confidence": result.confidence,
            "votes": result.votes,
            "all_sources": result.all_candidates,
            "collected_at": datetime.now(timezone.utc).isoformat(),
        }
        # Add description if available
        if result.description:
            output["description"] = result.description
            output["description_source"] = result.description_source
        return output
    
    return None


def main():
    """Main entry point."""
    import argparse
    
    global _logger
    
    parser = argparse.ArgumentParser(
        description="Parallel domain collection with multi-source consensus",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--max-companies",
        type=int,
        default=None,
        help="Limit number of companies to process (for testing)",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Quick test with a few major companies (AAPL, MSFT, GOOGL, etc.)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from existing output file",
    )
    
    args = parser.parse_args()
    
    # Set up logging
    log_dir = Path("logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # Use the script's filename (without .py extension) for the log file
    script_name = Path(__file__).stem
    log_file = log_dir / f"{script_name}_{timestamp}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            # Don't log to stdout - let tqdm handle progress display
        ]
    )
    _logger = logging.getLogger(__name__)
    
    # Suppress verbose HTTP logging
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    
    output_file = Path("data/public_company_domains.json")
    domain_list_file = Path("data/public_company_domains_list.txt")
    
    _logger.info("=" * 80)
    _logger.info("Starting parallel domain collection")
    _logger.info(f"Log file: {log_file}")
    
    # Create session
    session = requests.Session()
    adapter = requests.adapters.HTTPAdapter(
        pool_connections=20,
        pool_maxsize=20,
        max_retries=requests.adapters.Retry(total=3, backoff_factor=0.3)
    )
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    
    # Fetch companies
    _logger.info("Fetching company list from SEC EDGAR...")
    companies = fetch_company_tickers(session)
    _logger.info(f"Found {len(companies)} companies")
    
    # Test mode: just test a few major companies
    if args.test:
        test_tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "META", "NVDA"]
        test_companies = {}
        for cik, info in companies.items():
            if info["ticker"] in test_tickers:
                test_companies[cik] = info
        companies = test_companies
        _logger.info(f"TEST MODE: Processing {len(companies)} test companies")
    
    # Load existing results
    existing_results = {}
    if args.resume and output_file.exists():
        try:
            with open(output_file) as f:
                data = json.load(f)
                existing_results = {r["cik"]: r for r in data if "cik" in r}
            _logger.info(f"Loaded {len(existing_results)} existing results")
        except Exception as e:
            _logger.warning(f"Could not load existing results: {e}")
    
    # Filter companies to process
    to_process = {
        cik: info for cik, info in companies.items()
        if cik not in existing_results
    }
    
    if args.max_companies:
        to_process = dict(list(to_process.items())[:args.max_companies])
    
    _logger.info(f"Processing {len(to_process)} companies in parallel...")
    
    results = list(existing_results.values())
    
    # Process in parallel
    # With 30 workers, each making 4 parallel API calls, we can have up to 120 concurrent requests
    # Rate limits are enforced per-source to prevent overwhelming any single service
    # Finnhub (1 req/sec) is the bottleneck, but yfinance/Finviz/SEC can handle more
    max_workers = 30  # Process 30 companies concurrently
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_cik = {
            executor.submit(process_company, session, cik, info, existing_results): cik
            for cik, info in to_process.items()
        }
        
        save_interval = 10  # Save every 10 companies
        processed_count = 0
        
        with tqdm(total=len(to_process), desc="Processing companies") as pbar:
            start_time = time.time()
            for future in as_completed(future_to_cik):
                cik = future_to_cik[future]
                try:
                    # Timeout per company: 60 seconds max (4 sources * 30s each with overhead)
                    result = future.result(timeout=60)
                    if result:
                        results.append(result)
                        # Log successful finds periodically
                        if len(results) % 100 == 0:
                            elapsed = time.time() - start_time
                            rate = len(results) / elapsed if elapsed > 0 else 0
                            _logger.debug(f"Progress: {len(results)} companies found ({rate:.1f} companies/sec)")
                except TimeoutError:
                    _logger.warning(f"Timeout processing CIK {cik} - skipping")
                except Exception as e:
                    _logger.warning(f"Error processing CIK {cik}: {e}")
                finally:
                    pbar.update(1)
                    pbar.set_postfix({"found": len(results)})
                    processed_count += 1
                    
                    # Incremental save to prevent data loss
                    if processed_count % save_interval == 0:
                        output_file.parent.mkdir(parents=True, exist_ok=True)
                        with open(output_file, "w") as f:
                            json.dump(results, f, indent=2)
                        _logger.debug(f"Incremental save: {len(results)} companies saved to {output_file}")
    
    # Final save
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    
    # Save domain list
    domains = sorted(set(r["domain"] for r in results if r.get("domain")))
    with open(domain_list_file, "w") as f:
        for domain in domains:
            f.write(f"{domain}\n")
    
    # Print final summary to stdout (after tqdm is done)
    print(f"\n✓ Complete: {len(results)} companies, {len(domains)} unique domains")
    print(f"✓ Saved to {output_file}")
    print(f"✓ Log file: {log_file}")
    
    _logger.info(f"✓ Complete: {len(results)} companies, {len(domains)} unique domains")
    _logger.info(f"✓ Saved to {output_file}")
    _logger.info("=" * 80)


if __name__ == "__main__":
    main()


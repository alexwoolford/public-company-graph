#!/usr/bin/env python3
"""
Enrich Company nodes with additional properties from public data sources.

This script:
1. Fetches company properties from SEC EDGAR, Yahoo Finance, and Wikidata
2. Stores enriched data in unified cache (namespace: company_properties)
3. Updates Company nodes in Neo4j with new properties

New Company Properties:
- Industry classification: sic_code, naics_code, sector, industry
- Financial metrics: market_cap, revenue, employees
- Geographic data: headquarters_city, headquarters_state, headquarters_country
- Metadata: data_source, data_updated_at

Data Sources:
- SEC EDGAR API: SIC/NAICS codes (public domain)
- Yahoo Finance: Sector, industry, market cap, revenue, employees, HQ location
- Wikidata: Supplemental data (optional, lower priority)

Usage:
    python scripts/enrich_company_properties.py          # Dry-run (plan only)
    python scripts/enrich_company_properties.py --execute  # Actually enrich data
"""

import argparse
import sys
import time
from typing import Dict, List, Optional

import requests

from domain_status_graph.cache import get_cache
from domain_status_graph.cli import (
    add_execute_argument,
    get_driver_and_database,
    setup_logging,
    verify_neo4j_connection,
)
from domain_status_graph.company.enrichment import (
    fetch_sec_company_info,
    fetch_wikidata_info,
    fetch_yahoo_finance_info,
    merge_company_data,
)
from domain_status_graph.constants import BATCH_SIZE_SMALL
from domain_status_graph.neo4j import create_company_constraints

# Rate limiting for Yahoo Finance (be conservative)
_yahoo_last_call = 0
_yahoo_min_interval = 0.1  # 10 requests per second max


def rate_limit_yahoo():
    """Simple rate limiting for Yahoo Finance."""
    global _yahoo_last_call
    current_time = time.time()
    elapsed = current_time - _yahoo_last_call
    if elapsed < _yahoo_min_interval:
        time.sleep(_yahoo_min_interval - elapsed)
    _yahoo_last_call = time.time()


def enrich_company(
    cik: str, ticker: str, name: str, session: requests.Session, cache
) -> Optional[Dict]:
    """
    Enrich a single company with data from all sources.

    Args:
        cik: Company CIK
        ticker: Stock ticker
        name: Company name
        session: HTTP session for SEC API
        cache: Unified cache instance

    Returns:
        Enriched company data dictionary or None if failed
    """
    # Check cache first
    cache_key = cik
    cached = cache.get("company_properties", cache_key)
    if cached:
        return cached

    # Fetch from sources
    sec_data = fetch_sec_company_info(cik, session=session)
    rate_limit_yahoo()
    yahoo_data = fetch_yahoo_finance_info(ticker) if ticker else None
    wikidata_data = fetch_wikidata_info(ticker, name)  # Optional, may return None

    # Merge data
    enriched = merge_company_data(sec_data, yahoo_data, wikidata_data)

    # Store in cache (TTL: 30 days - company properties don't change often)
    if enriched:
        cache.set("company_properties", cache_key, enriched, ttl_days=30)
        return enriched

    return None


def enrich_all_companies(
    driver,
    cache,
    batch_size: int = BATCH_SIZE_SMALL,
    database: str = None,
    execute: bool = False,
    logger=None,
) -> int:
    """
    Enrich all Company nodes with properties from public data sources.

    Args:
        driver: Neo4j driver
        cache: Unified cache instance
        batch_size: Batch size for Neo4j updates
        database: Neo4j database name
        execute: If False, only print plan
        logger: Logger instance

    Returns:
        Number of companies enriched
    """
    if logger is None:
        import logging

        logger = logging.getLogger(__name__)

    # Get all companies from Neo4j
    with driver.session(database=database) as session:
        result = session.run(
            """
            MATCH (c:Company)
            RETURN c.cik AS cik, c.ticker AS ticker, c.name AS name
            ORDER BY c.ticker
            """
        )
        companies = [dict(row) for row in result]

    if not companies:
        logger.warning("No companies found in Neo4j. Run load_company_data.py first.")
        return 0

    logger.info(f"Found {len(companies)} companies to enrich")

    if not execute:
        logger.info("=" * 80)
        logger.info("DRY RUN MODE")
        logger.info("=" * 80)
        logger.info(f"Would enrich {len(companies)} companies")
        logger.info("Sources: SEC EDGAR, Yahoo Finance, Wikidata")
        logger.info("=" * 80)
        return 0

    # Create HTTP session for SEC API
    session = requests.Session()
    adapter = requests.adapters.HTTPAdapter(
        pool_connections=10,
        pool_maxsize=10,
        max_retries=requests.adapters.Retry(total=3, backoff_factor=0.3),
    )
    session.mount("http://", adapter)
    session.mount("https://", adapter)

    # Enrich companies
    enriched_count = 0
    failed_count = 0
    cached_count = 0
    update_batch = []

    logger.info("=" * 80)
    logger.info("Enriching Company Properties")
    logger.info("=" * 80)

    for i, company in enumerate(companies, 1):
        cik = company.get("cik")
        ticker = company.get("ticker", "")
        name = company.get("name", "")

        if not cik:
            logger.warning(f"Skipping company without CIK: {ticker}")
            failed_count += 1
            continue

        # Check cache first
        cache_key = cik
        cached = cache.get("company_properties", cache_key)
        if cached:
            cached_count += 1
            enriched_data = cached
        else:
            # Fetch from sources
            enriched_data = enrich_company(cik, ticker, name, session, cache)
            if not enriched_data:
                failed_count += 1
                logger.debug(f"Failed to enrich {ticker} (CIK: {cik})")
                continue

        enriched_count += 1

        # Add to batch for Neo4j update
        if enriched_data:
            update_batch.append({"cik": cik, **enriched_data})

        # Update Neo4j in batches
        if len(update_batch) >= batch_size or i == len(companies):
            if update_batch:
                _update_companies_batch(driver, update_batch, database=database, logger=logger)
                logger.info(
                    f"  Updated {len(update_batch)} companies in Neo4j... ({i}/{len(companies)})"
                )
                update_batch = []

    logger.info("=" * 80)
    logger.info("Enrichment Complete")
    logger.info("=" * 80)
    logger.info(f"  Total companies: {len(companies)}")
    logger.info(f"  Enriched: {enriched_count}")
    logger.info(f"  From cache: {cached_count}")
    logger.info(f"  Failed: {failed_count}")

    return enriched_count


def _update_companies_batch(driver, batch: List[Dict], database: str = None, logger=None) -> None:
    """Update a batch of Company nodes in Neo4j."""
    if logger is None:
        import logging

        logger = logging.getLogger(__name__)

    query = """
    UNWIND $batch AS company
    MATCH (c:Company {cik: company.cik})
    SET c.sic_code = company.sic_code,
        c.naics_code = company.naics_code,
        c.sector = company.sector,
        c.industry = company.industry,
        c.market_cap = company.market_cap,
        c.revenue = company.revenue,
        c.employees = company.employees,
        c.headquarters_city = company.headquarters_city,
        c.headquarters_state = company.headquarters_state,
        c.headquarters_country = company.headquarters_country,
        c.founded_year = company.founded_year,
        c.data_source = company.data_source,
        c.data_updated_at = company.data_updated_at
    """

    try:
        with driver.session(database=database) as session:
            session.run(query, batch=batch)
    except Exception as e:
        logger.error(f"Error updating companies batch: {e}")
        raise


def main():
    """Run the company property enrichment script."""
    parser = argparse.ArgumentParser(
        description="Enrich Company nodes with properties from public data sources"
    )
    add_execute_argument(parser)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=BATCH_SIZE_SMALL,
        help=f"Batch size for Neo4j updates (default: {BATCH_SIZE_SMALL})",
    )

    args = parser.parse_args()

    logger = setup_logging("enrich_company_properties", execute=args.execute)
    cache = get_cache()

    if not args.execute:
        # Dry-run: show plan
        driver, database = get_driver_and_database(logger)
        try:
            enrich_all_companies(
                driver,
                cache,
                batch_size=args.batch_size,
                database=database,
                execute=False,
                logger=logger,
            )
        finally:
            driver.close()
        return

    logger.info("=" * 80)
    logger.info("Company Property Enrichment")
    logger.info("=" * 80)

    driver, database = get_driver_and_database(logger)

    try:
        # Verify connection
        if not verify_neo4j_connection(driver, database, logger):
            sys.exit(1)

        # Ensure constraints exist
        logger.info("\n1. Creating/verifying constraints...")
        create_company_constraints(driver, database=database, logger=logger)

        # Enrich companies
        logger.info("\n2. Enriching company properties...")
        enriched = enrich_all_companies(
            driver,
            cache,
            batch_size=args.batch_size,
            database=database,
            execute=True,
            logger=logger,
        )

        logger.info("\n" + "=" * 80)
        logger.info("✓ Complete!")
        logger.info("=" * 80)
        logger.info(f"Enriched {enriched} companies")

    finally:
        driver.close()


if __name__ == "__main__":
    main()

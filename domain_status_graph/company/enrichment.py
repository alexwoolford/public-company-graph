"""
Company property enrichment from public data sources.

This module provides functions to fetch and enrich Company node properties
from SEC EDGAR, Yahoo Finance, and Wikidata.

Data Sources:
- SEC EDGAR API: https://www.sec.gov/edgar/sec-api-documentation
  License: Public domain (17 CFR 240.12g-1)
- Yahoo Finance (via yfinance): https://github.com/ranaroussi/yfinance
  License: Free, no explicit license restrictions
- Wikidata: https://www.wikidata.org/
  License: CC0 (public domain)

Reference: CompanyKG paper - Company node attributes (employees, sector, etc.)
"""

import logging
import time
from threading import Lock
from typing import Dict, Optional

import requests

logger = logging.getLogger(__name__)

# Rate limiting for SEC EDGAR API (10 requests per second)
_sec_rate_limit = {"lock": Lock(), "last_call": 0, "min_interval": 0.1}


def _rate_limit_sec():
    """Enforce SEC EDGAR API rate limiting (10 req/sec)."""
    with _sec_rate_limit["lock"]:
        current_time = time.time()
        elapsed = current_time - _sec_rate_limit["last_call"]
        if elapsed < _sec_rate_limit["min_interval"]:
            time.sleep(_sec_rate_limit["min_interval"] - elapsed)
        _sec_rate_limit["last_call"] = time.time()


def fetch_sec_company_info(cik: str, session: Optional[requests.Session] = None) -> Optional[Dict]:
    """
    Fetch company information from SEC EDGAR API.

    Args:
        cik: SEC Central Index Key (10-digit string, zero-padded)
        session: Optional requests.Session for connection pooling

    Returns:
        Dictionary with company info (SIC, NAICS, etc.) or None if not found

    Data Source: SEC EDGAR API (public domain)
    Reference: https://www.sec.gov/edgar/sec-api-documentation
    """
    _rate_limit_sec()

    try:
        if session is None:
            session = requests.Session()

        # SEC requires User-Agent header
        url = f"https://data.sec.gov/submissions/CIK{cik.zfill(10)}.json"
        headers = {
            "User-Agent": "domain_status_graph enrichment script (contact: alex@woolford.io)",
            "Accept": "application/json",
        }

        response = session.get(url, headers=headers, timeout=10)
        response.raise_for_status()

        data = response.json()
        company_info = data.get("name", "")

        # Extract SIC and NAICS from the submissions data
        sic_code = None
        naics_code = None

        # SIC code is typically in the first entry of sic array
        # Format: "7372 - Services-Prepackaged Software"
        sic_array = data.get("sic", [])
        if sic_array and len(sic_array) > 0:
            sic_code = str(sic_array[0]).split("-")[0].strip()

        # NAICS might be in the filings or we need to check companyfacts endpoint
        # For now, we'll get it from companyfacts if available
        naics_array = data.get("naics", [])
        if naics_array and len(naics_array) > 0:
            naics_code = str(naics_array[0]).split("-")[0].strip()

        result = {
            "sic_code": sic_code,
            "naics_code": naics_code,
            "company_name": company_info,
        }

        # Normalize codes
        normalized = normalize_industry_codes(sic_code, naics_code)
        result.update(normalized)

        return result

    except requests.exceptions.RequestException as e:
        logger.debug(f"SEC EDGAR API error for CIK {cik}: {e}")
        return None
    except Exception as e:
        logger.warning(f"Unexpected error fetching SEC data for CIK {cik}: {e}")
        return None


def fetch_yahoo_finance_info(ticker: str) -> Optional[Dict]:
    """
    Fetch company information from Yahoo Finance.

    Args:
        ticker: Stock ticker symbol (e.g., 'AAPL')

    Returns:
        Dictionary with company info (sector, industry, market_cap, revenue, employees) or None

    Data Source: Yahoo Finance via yfinance library
    """
    try:
        import yfinance as yf

        ticker_obj = yf.Ticker(ticker)
        info = ticker_obj.info

        # Extract relevant fields
        result = {
            "sector": info.get("sector"),
            "industry": info.get("industry"),
            "market_cap": info.get("marketCap"),
            "revenue": info.get("totalRevenue"),
            "employees": info.get("fullTimeEmployees"),
            "headquarters_city": info.get("city"),
            "headquarters_state": info.get("state"),
            "headquarters_country": info.get("country", "US"),
            "founded_year": info.get("founded"),
        }

        # Filter out None values for cleaner output
        return {k: v for k, v in result.items() if v is not None}

    except ImportError:
        logger.warning("yfinance not available. Install with: pip install yfinance")
        return None
    except Exception as e:
        logger.warning(f"Failed to fetch Yahoo Finance data for {ticker}: {e}")
        return None


def fetch_wikidata_info(ticker: str, company_name: str) -> Optional[Dict]:
    """
    Fetch company information from Wikidata using SPARQL.

    Args:
        ticker: Stock ticker symbol
        company_name: Company name for disambiguation

    Returns:
        Dictionary with company info (employees, HQ location, etc.) or None

    Data Source: Wikidata SPARQL endpoint (CC0/public domain)
    Reference: https://www.wikidata.org/wiki/Wikidata:Main_Page
    """
    # TODO: Implement Wikidata SPARQL queries
    # This is lower priority - Yahoo Finance and SEC provide most of what we need
    # Wikidata can supplement with employees and HQ location if missing
    logger.debug("Wikidata queries not yet implemented (lower priority)")
    return None


def normalize_industry_codes(sic: Optional[str], naics: Optional[str]) -> Dict:
    """
    Normalize and validate industry classification codes.

    Args:
        sic: Standard Industrial Classification code
        naics: North American Industry Classification System code

    Returns:
        Dictionary with normalized codes
    """
    result = {}
    if sic:
        # SIC codes are typically 4 digits, extract numeric part
        sic_clean = "".join(filter(str.isdigit, str(sic)))
        if sic_clean and len(sic_clean) >= 2:
            result["sic_code"] = sic_clean[:4].zfill(4)
    if naics:
        # NAICS codes are typically 6 digits, extract numeric part
        naics_clean = "".join(filter(str.isdigit, str(naics)))
        if naics_clean and len(naics_clean) >= 2:
            result["naics_code"] = naics_clean[:6].zfill(6)
    return result


def merge_company_data(
    sec_data: Optional[Dict], yahoo_data: Optional[Dict], wikidata_data: Optional[Dict]
) -> Dict:
    """
    Merge data from multiple sources, with priority order.

    Priority: SEC > Yahoo Finance > Wikidata (for overlapping fields)

    Args:
        sec_data: Data from SEC EDGAR
        yahoo_data: Data from Yahoo Finance
        wikidata_data: Data from Wikidata

    Returns:
        Merged dictionary with all available data
    """
    result = {}

    # Start with Yahoo Finance (most complete for financials)
    if yahoo_data:
        result.update(yahoo_data)

    # Override with SEC data (more authoritative for SIC/NAICS)
    if sec_data:
        # SEC provides SIC/NAICS codes
        if sec_data.get("sic_code"):
            result["sic_code"] = sec_data["sic_code"]
        if sec_data.get("naics_code"):
            result["naics_code"] = sec_data["naics_code"]

    # Add Wikidata data (supplemental)
    if wikidata_data:
        # Only add fields not already present
        for key, value in wikidata_data.items():
            if key not in result or result[key] is None:
                result[key] = value

    # Add metadata
    sources = []
    if sec_data:
        sources.append("SEC_EDGAR")
    if yahoo_data:
        sources.append("YAHOO_FINANCE")
    if wikidata_data:
        sources.append("WIKIDATA")

    result["data_source"] = ",".join(sources) if sources else None
    result["data_updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    return result

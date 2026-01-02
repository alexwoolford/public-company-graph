"""
Business relationship extraction from 10-K filings.

This module extracts business relationships (competitors, customers, suppliers, partners)
from Item 1 (Business) and Item 1A (Risk Factors) sections of 10-K filings.

Relationship Types (matching CompanyKG schema):
- competitor: Direct competitors mentioned in competitive landscape sections
- customer: Significant customers (SEC requires disclosure if >10% of revenue)
- supplier: Key suppliers and vendors
- partner: Business partners, strategic alliances

The approach:
1. Find sentences containing relationship-specific keywords
2. Extract potential company names from those sentences
3. Resolve against known companies using entity resolution lookup
4. Return resolved relationships for graph creation

Based on CompanyKG paper: https://arxiv.org/abs/2306.10649
"""

import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class RelationshipType(Enum):
    """Business relationship types (matching CompanyKG schema)."""

    COMPETITOR = "competitor"
    CUSTOMER = "customer"
    SUPPLIER = "supplier"
    PARTNER = "partner"


# Relationship type to Neo4j relationship type mapping
RELATIONSHIP_TYPE_TO_NEO4J = {
    RelationshipType.COMPETITOR: "HAS_COMPETITOR",
    RelationshipType.CUSTOMER: "HAS_CUSTOMER",
    RelationshipType.SUPPLIER: "HAS_SUPPLIER",
    RelationshipType.PARTNER: "HAS_PARTNER",
}


@dataclass
class CompanyLookup:
    """Lookup table for entity resolution."""

    # Maps various name forms → (cik, ticker, official_name)
    name_to_company: dict[str, tuple[str, str, str]] = field(default_factory=dict)
    ticker_to_company: dict[str, tuple[str, str, str]] = field(default_factory=dict)
    # Set of all company names (for quick membership check)
    all_names: set[str] = field(default_factory=set)
    # Set of all tickers
    all_tickers: set[str] = field(default_factory=set)


@dataclass
class RelationshipMention:
    """A business relationship mention extracted from 10-K."""

    relationship_type: RelationshipType
    raw_text: str  # The raw text that was extracted
    context: str  # Surrounding context
    resolved_cik: str | None = None
    resolved_ticker: str | None = None
    resolved_name: str | None = None
    confidence: float = 0.0


# =============================================================================
# KEYWORD DEFINITIONS FOR EACH RELATIONSHIP TYPE
# =============================================================================

# Keywords that indicate competitor context
COMPETITOR_KEYWORDS = {
    "competitor",
    "competitors",
    "compete",
    "competes",
    "competing",
    "competition",
    "competitive",
    "rival",
    "rivals",
}

# Keywords that indicate customer context
# SEC requires disclosure of customers >10% of revenue
CUSTOMER_KEYWORDS = {
    "customer",
    "customers",
    "client",
    "clients",
    "significant customer",
    "major customer",
    "largest customer",
    "key customer",
    "principal customer",
    "revenue concentration",
    "customer concentration",
    "accounts for",
    "accounted for",
    "represents",
    "represented",
    "% of revenue",
    "percent of revenue",
    "% of sales",
    "percent of sales",
    "% of net revenue",
    "% of total revenue",
}

# Keywords that indicate supplier context
SUPPLIER_KEYWORDS = {
    "supplier",
    "suppliers",
    "vendor",
    "vendors",
    "supply chain",
    "supply agreement",
    "purchase agreement",
    "source",
    "sources",
    "sourcing",
    "procure",
    "procurement",
    "key supplier",
    "principal supplier",
    "sole supplier",
    "single source",
    "sole source",
    "depend on",
    "reliance on",
    "raw material",
    "raw materials",
    "component",
    "components",
    "manufacturer",
    "manufacturers",
    "contract manufacturer",
}

# Keywords that indicate partner context
PARTNER_KEYWORDS = {
    "partner",
    "partners",
    "partnership",
    "partnerships",
    "alliance",
    "alliances",
    "strategic alliance",
    "joint venture",
    "joint ventures",
    "collaboration",
    "collaborate",
    "collaborates",
    "collaborating",
    "agreement with",
    "arrangement with",
    "relationship with",
    "licensing agreement",
    "distribution agreement",
    "strategic relationship",
    "business relationship",
}

# Map relationship type to keywords
RELATIONSHIP_KEYWORDS = {
    RelationshipType.COMPETITOR: COMPETITOR_KEYWORDS,
    RelationshipType.CUSTOMER: CUSTOMER_KEYWORDS,
    RelationshipType.SUPPLIER: SUPPLIER_KEYWORDS,
    RelationshipType.PARTNER: PARTNER_KEYWORDS,
}


# =============================================================================
# BLOCKLISTS (shared across all relationship types)
# =============================================================================

# Blocklist: Common words/abbreviations that match ticker symbols
# These are words that appear frequently in 10-K text but happen to match real ticker symbols
TICKER_BLOCKLIST = {
    # Single letters (all blocked)
    "A",
    "B",
    "C",
    "D",
    "E",
    "F",
    "G",
    "H",
    "I",
    "J",
    "K",
    "L",
    "M",
    "N",
    "O",
    "P",
    "Q",
    "R",
    "S",
    "T",
    "U",
    "V",
    "W",
    "X",
    "Y",
    "Z",
    # 2-letter common words/abbreviations
    "AD",  # AD (advertisement) - matches ARRAY DIGITAL ticker
    "AI",  # AI (Artificial Intelligence) - common industry term
    "AN",
    "AS",
    "AT",
    "AB",
    "BE",
    "BY",
    "CE",  # CE (Conformité Européenne mark) - matches Celanese ticker
    "DO",
    "EC",
    "EU",
    "GO",
    "HE",  # HE (pronoun) - matches Hawaiian Electric ticker
    "HR",
    "IN",
    "IP",
    "IS",
    "IT",
    "ON",
    "OR",
    "PC",
    "PR",
    "SA",
    "SO",
    "UK",
    "UP",
    "US",
    "WE",
    # 3-letter common words that match tickers
    "ACT",
    "ACA",
    "ADD",
    "AGE",
    "AID",
    "AIR",
    "ALL",
    "AND",
    "ANY",
    "API",  # API (Application Programming Interface) - common tech term
    "ARE",
    "BDC",
    "BIG",
    "CAR",
    "CAN",
    "CMS",  # CMS (Content Management System) - matches CMS Energy ticker
    "CRM",  # CRM (Customer Relationship Management) - matches Salesforce ticker
    "CPU",
    "DEI",  # DEI (Diversity Equity Inclusion) - matches Douglas Emmett ticker
    "DMA",
    "DNA",  # DNA (biology term) - matches Ginkgo Bioworks ticker
    "DSP",
    "EEA",
    "EPC",  # EPC (Engineering Procurement Construction) - matches Edgewell ticker
    "ESG",
    "FOR",
    "GDP",
    "GLP",  # GLP (Good Laboratory Practice) - matches Global Partners ticker
    "GPU",
    "HAS",
    "HHS",
    "III",  # III (Roman numeral) - matches Information Services Group ticker
    "IOT",  # IoT (Internet of Things) - matches Samsara ticker
    "KEY",  # KEY (common word) - matches KeyCorp ticker
    "NET",
    "NEW",
    "NOW",
    "OLD",
    "ONE",
    "OUR",
    "OUT",
    "PER",
    "QSR",  # QSR (Quick Service Restaurant) - matches Restaurant Brands ticker
    "SEC",
    "SEE",
    "SOC",
    "THE",
    "TPG",  # TPG (common abbreviation) - matches TPG Inc ticker
    "TWO",
    "USA",
    "VIA",
    "YOU",
    # 4-letter common words that match tickers
    "ALSO",
    "ASIC",
    "BACK",
    "BEEN",
    "BOTH",
    "CARE",
    "CASH",
    "DRUG",
    "EACH",
    "EVEN",
    "FAST",
    "FLEX",  # FLEX (common word) - matches Flex Ltd ticker
    "FORM",
    "GOOD",
    "HALF",
    "JUST",
    "MANY",
    "MORE",
    "MOST",
    "MUST",
    "NEED",
    "NOTE",  # NOTE (common word) - matches FiscalNote ticker
    "ONCE",
    "ONLY",
    "OPEN",  # OPEN (common word) - matches Opendoor ticker
    "PLUS",
    "REAL",
    "RISK",
    "SAFE",
    "SELF",
    "SOME",
    "SUCH",
    "TERM",
    "VERY",
    "WELL",
    "WHEN",
    "WILL",
    "WITH",
    # 5+ letter common words that match tickers
    "ABOUT",
    "AUDIO",
    "BEING",
    "CLEAR",
    "FOCUS",
    "GLOBE",
    "IDEAL",
    "LIFE",
    "LIGHT",
    "MEDIA",
    "PRIME",
    "RANGE",
    "SMILE",
    "STILL",
    "THINK",
    "THREE",
    "TOTAL",
    "UNITY",
    "VALUE",
    "VITAL",
}

# Company names that are also common English words
# These are blocked to reduce false positives from generic terms in 10-K filings
NAME_BLOCKLIST = {
    # Generic business terms
    "advantage",
    "alliance",
    "associates",
    "capital",
    "catalyst",
    "cost",  # "cost of..." is common, matches COST (Costco)
    "enterprises",
    "financial",
    "focus",
    "group",
    "holdings",
    "industries",
    "insight",
    "investment",
    "investments",
    "joint",  # "joint venture" is common, matches JYNT (Joint Corp)
    "management",
    "partners",
    "platform",
    "platforms",
    "premier",
    "progress",
    "regis",  # Common name that matches RGS (Regis Corp hair salons)
    "reliance",
    "resources",
    "securities",
    "service",
    "services",
    "solution",
    # Common words that match company names (discovered via ground truth analysis)
    "target",  # Goal/objective, not Target Corp - high false positive rate
    "comstock",  # Power plant names, geographic locations - not Comstock Inc
    "enact",  # Verb meaning "to make law" - not Enact Holdings
    # Exchange/listing venue names (these refer to the exchange, not the company)
    "nasdaq",  # Stock exchange listings like "NASDAQ: AAPL"
    "nyse",  # NYSE listings
    "new york stock exchange",
    "solutions",
    "system",
    "systems",
    "technology",
    "technologies",
    "ventures",
    # Common words that match company names (high false positive rate)
    "ad",  # advertisement
    "audio",
    "care",
    "cash",
    "core",
    "data",
    "digital",
    "direct",
    "energy",
    "global",
    "growth",
    "health",
    "key",  # common word - matches KeyCorp
    "life",
    "light",
    "local",
    "media",
    "mobile",
    "national",
    "network",
    "note",  # common word - matches FiscalNote
    "open",  # common word - matches Opendoor
    "plus",
    "power",
    "prime",
    "pro",
    "pure",
    "real",
    "smart",
    "source",
    "star",
    "total",
    "trust",
    "unity",
    "value",
    "vital",
    "web",
    # Industry/regulatory terms that match company names
    "cms",  # Content Management System
    "dei",  # Diversity, Equity & Inclusion
    "dna",  # biology term
    "epc",  # Engineering, Procurement, Construction
    "esg",  # Environmental, Social, Governance
    "glp",  # Good Laboratory Practice
    "iot",  # Internet of Things
    "qsr",  # Quick Service Restaurant
}

# High-value company names that SHOULD be extracted even if they look like common words
# These override NAME_BLOCKLIST and short-candidate filtering when the company is in the lookup
# Only include truly distinctive company names that are unambiguous in business context
HIGH_VALUE_COMPANY_NAMES = {
    # Major tech companies (distinctive names)
    "adobe",
    "alphabet",
    "amazon",
    "apple",
    "cisco",
    "dell",
    "google",
    "intel",
    "microsoft",
    "nvidia",
    "oracle",
    "qualcomm",
    "salesforce",
    "samsung",
    # Major financial/retail (distinctive names)
    "berkshire",
    "blackrock",
    "citi",
    "citibank",
    "citigroup",
    "costco",
    "disney",
    "goldman",
    "jpmorgan",
    "mastercard",
    "netflix",
    "paypal",
    "pfizer",
    "starbucks",
    "tesla",
    "visa",
    "walmart",
    # Multi-word distinctive names
    "berkshire hathaway",
    "dollar general",
    "dollar tree",
    "goldman sachs",
    "home depot",
    "johnson & johnson",
    "jpmorgan chase",
    "lockheed martin",
    "morgan stanley",
    "procter & gamble",
    "wells fargo",
}


# =============================================================================
# LOOKUP TABLE BUILDING
# =============================================================================


def build_company_lookup(driver, database: str | None = None) -> CompanyLookup:
    """
    Build a lookup table from Neo4j Company nodes for entity resolution.

    Creates multiple name variations for each company to improve matching.

    Args:
        driver: Neo4j driver
        database: Neo4j database name

    Returns:
        CompanyLookup with name → company mappings
    """
    lookup = CompanyLookup()

    query = """
    MATCH (c:Company)
    WHERE c.cik IS NOT NULL AND c.name IS NOT NULL
    RETURN c.cik as cik, c.ticker as ticker, c.name as name
    """

    with driver.session(database=database) as session:
        result = session.run(query)
        for record in result:
            cik = record["cik"]
            ticker = record["ticker"] or ""
            name = record["name"] or ""

            company_tuple = (cik, ticker, name)

            # Add full name (lowercased)
            name_lower = name.lower().strip()
            if name_lower:
                lookup.name_to_company[name_lower] = company_tuple
                lookup.all_names.add(name_lower)

            # Add name without common suffixes
            clean_name = _normalize_company_name(name)
            if clean_name and clean_name != name_lower:
                lookup.name_to_company[clean_name] = company_tuple
                lookup.all_names.add(clean_name)

            # Add ticker (uppercase)
            if ticker:
                ticker_upper = ticker.upper().strip()
                lookup.ticker_to_company[ticker_upper] = company_tuple
                lookup.all_tickers.add(ticker_upper)

    logger.info(
        f"Built company lookup: {len(lookup.name_to_company)} name variants, "
        f"{len(lookup.ticker_to_company)} tickers"
    )
    return lookup


def _normalize_company_name(name: str) -> str:
    """Normalize a company name by removing common suffixes."""
    name = name.lower().strip()

    suffixes = [
        " corporation",
        " incorporated",
        " holdings ltd",
        " holding ltd",
        " holdings",
        " holding",
        " technologies",
        " technology",
        " solutions",
        " platforms",
        " services",
        " systems",
        " group",
        " corp.",
        " corp",
        " inc.",
        " inc",
        " ltd.",
        " ltd",
        " llc",
        " plc",
        " co.",
        " co",
        "/de/",
        "/md/",
        "/nv/",
    ]

    for suffix in suffixes:
        if name.endswith(suffix):
            name = name[: -len(suffix)]

    name = re.sub(r"^[\s,.\-]+|[\s,.\-]+$", "", name)
    return name.strip()


# =============================================================================
# EXTRACTION FUNCTIONS
# =============================================================================


def extract_relationship_sentences(
    text: str,
    relationship_type: RelationshipType,
) -> list[tuple[str, int]]:
    """
    Find sentences that mention a specific relationship type.

    Args:
        text: Full text to search
        relationship_type: Type of relationship to find

    Returns:
        List of (sentence, start_position) tuples
    """
    if not text:
        return []

    keywords = RELATIONSHIP_KEYWORDS[relationship_type]
    sentences = []
    current_pos = 0

    for sentence in re.split(r"(?<=[.!?])\s+", text):
        sentence = sentence.strip()
        sentence_lower = sentence.lower()

        # Check if any keyword appears in sentence
        if sentence and any(kw in sentence_lower for kw in keywords):
            sentences.append((sentence, current_pos))
        current_pos += len(sentence) + 1

    return sentences


def extract_and_resolve_relationships(
    business_description: str | None,
    risk_factors: str | None,
    lookup: CompanyLookup,
    relationship_type: RelationshipType,
    self_cik: str | None = None,
    use_layered_validation: bool = False,
    embedding_threshold: float = 0.30,
) -> list[dict[str, Any]]:
    """
    Extract and resolve business relationships from 10-K text.

    Args:
        business_description: Item 1 Business description text
        risk_factors: Item 1A Risk Factors text
        lookup: CompanyLookup table
        relationship_type: Type of relationship to extract
        self_cik: CIK of the company filing (to exclude self-references)
        use_layered_validation: If True, apply embedding + pattern validation
        embedding_threshold: Minimum similarity for embedding check (default 0.30)

    Returns:
        List of dicts with: cik, ticker, name, confidence, raw_mention, context
    """
    results = []
    seen_ciks: set[str] = set()

    # Initialize layered validator if requested
    validator = None
    if use_layered_validation:
        from public_company_graph.entity_resolution.layered_validator import (
            LayeredEntityValidator,
        )

        validator = LayeredEntityValidator(embedding_threshold=embedding_threshold)

    # Combine texts
    texts = []
    if business_description:
        texts.append(business_description)
    if risk_factors:
        texts.append(risk_factors)

    for text in texts:
        # Find sentences with relationship keywords
        sentences = extract_relationship_sentences(text, relationship_type)

        for sentence, _ in sentences:
            # Extract potential company names from this sentence
            # Pattern 1: Capitalized multi-word sequences (1-4 words)
            candidates = re.findall(
                r"\b([A-Z][a-zA-Z&\.\-]*(?:\s+[A-Z][a-zA-Z&\.\-]*){0,3})\b",
                sentence,
            )

            # Pattern 2: All-caps sequences (2-5 chars) - likely tickers
            candidates += re.findall(r"\b([A-Z]{2,5})\b", sentence)

            for candidate in candidates:
                candidate = candidate.strip()
                if len(candidate) < 2:
                    continue

                # Try to resolve against lookup
                resolved = _resolve_candidate(candidate, lookup, self_cik)

                if resolved and resolved["cik"] not in seen_ciks:
                    # Apply layered validation if enabled
                    validation_result = None
                    if validator:
                        validation_result = validator.validate(
                            context=sentence[:500],
                            mention=candidate,
                            ticker=resolved["ticker"],
                            company_name=resolved["name"],
                            relationship_type=relationship_type.value,
                        )
                        if not validation_result.accepted:
                            # Skip this candidate - failed validation
                            continue

                    seen_ciks.add(resolved["cik"])
                    result_dict = {
                        "target_cik": resolved["cik"],
                        "target_ticker": resolved["ticker"],
                        "target_name": resolved["name"],
                        "confidence": resolved["confidence"],
                        "raw_mention": candidate,
                        "context": sentence[:200],
                        "relationship_type": relationship_type.value,
                    }

                    # Add validation metadata if available
                    if validation_result:
                        result_dict["embedding_similarity"] = validation_result.embedding_similarity
                        result_dict["validation_passed"] = True

                    results.append(result_dict)

    return results


def _is_high_value_company(name: str) -> bool:
    """
    Check if a company name (or its normalized form) is in the high-value list.

    This checks both the exact name and common variations.
    """
    name_lower = name.lower().strip()
    normalized = _normalize_company_name(name_lower)

    # Check exact match
    if name_lower in HIGH_VALUE_COMPANY_NAMES:
        return True

    # Check normalized match
    if normalized in HIGH_VALUE_COMPANY_NAMES:
        return True

    # Check if any high-value name is contained in the company name
    # e.g., "NVIDIA Corporation" contains "nvidia"
    for hv_name in HIGH_VALUE_COMPANY_NAMES:
        if hv_name in name_lower:
            return True

    return False


def _resolve_candidate(
    candidate: str,
    lookup: CompanyLookup,
    self_cik: str | None,
) -> dict[str, Any] | None:
    """
    Try to resolve a candidate string to a known company.

    Uses multiple layers of filtering to reduce false positives:
    1. Blocklist filtering for common words that match tickers/names
    2. High-value company allowlist override (checked on resolved company name)
    3. Minimum length requirements for ticker-style matches
    4. Confidence scoring based on match quality
    """
    candidate_lower = candidate.lower().strip()
    candidate_upper = candidate.upper().strip()

    # Skip blocklisted tickers (common words that match ticker symbols)
    if candidate_upper in TICKER_BLOCKLIST:
        return None

    # Check if the CANDIDATE itself is a high-value company name
    candidate_is_high_value = candidate_lower in HIGH_VALUE_COMPANY_NAMES

    # Skip blocklisted names UNLESS it's a high-value company
    if candidate_lower in NAME_BLOCKLIST and not candidate_is_high_value:
        return None

    # For very short candidates (2-4 chars), apply stricter matching
    # These are likely ticker-style matches which have high false positive rates
    is_short_candidate = len(candidate) <= 4

    # Try exact ticker match
    if candidate_upper in lookup.ticker_to_company:
        cik, ticker, name = lookup.ticker_to_company[candidate_upper]
        if cik != self_cik:
            # Check if the RESOLVED company is high-value (even if candidate isn't)
            resolved_is_high_value = _is_high_value_company(name)
            # For short ticker matches, only allow if the resolved company is high-value
            if is_short_candidate and not candidate_is_high_value and not resolved_is_high_value:
                # Skip short ticker matches for non-high-value companies
                return None
            return {"cik": cik, "ticker": ticker, "name": name, "confidence": 1.0}

    # Try exact name match
    if candidate_lower in lookup.name_to_company:
        cik, ticker, name = lookup.name_to_company[candidate_lower]
        if cik != self_cik:
            resolved_is_high_value = _is_high_value_company(name)
            # For short name matches, only allow if high-value
            if is_short_candidate and not candidate_is_high_value and not resolved_is_high_value:
                return None
            return {"cik": cik, "ticker": ticker, "name": name, "confidence": 1.0}

    # Try normalized name match
    normalized = _normalize_company_name(candidate)
    if normalized and normalized in lookup.name_to_company:
        cik, ticker, name = lookup.name_to_company[normalized]
        if cik != self_cik:
            resolved_is_high_value = _is_high_value_company(name)
            # For normalized matches, require longer candidates unless high-value
            if len(normalized) <= 4 and not candidate_is_high_value and not resolved_is_high_value:
                return None
            return {"cik": cik, "ticker": ticker, "name": name, "confidence": 0.95}

    return None


def extract_all_relationships(
    business_description: str | None,
    risk_factors: str | None,
    lookup: CompanyLookup,
    self_cik: str | None = None,
    relationship_types: list[RelationshipType] | None = None,
) -> dict[RelationshipType, list[dict[str, Any]]]:
    """
    Extract all business relationships from 10-K text.

    Convenience function to extract all relationship types in one pass.

    Args:
        business_description: Item 1 Business description text
        risk_factors: Item 1A Risk Factors text
        lookup: CompanyLookup table
        self_cik: CIK of the company filing
        relationship_types: Types to extract (default: all)

    Returns:
        Dict mapping relationship type → list of extracted relationships
    """
    if relationship_types is None:
        relationship_types = list(RelationshipType)

    results = {}
    for rel_type in relationship_types:
        results[rel_type] = extract_and_resolve_relationships(
            business_description=business_description,
            risk_factors=risk_factors,
            lookup=lookup,
            relationship_type=rel_type,
            self_cik=self_cik,
        )

    return results

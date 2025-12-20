# Company Similarity Expansion Plan

## Overview

This document outlines the plan to expand the Domain Status Graph with additional company-to-company similarity relationships, inspired by the CompanyKG paper approach but using only public data sources.

**Reference**: This expansion is inspired by "CompanyKG: A Large-Scale Heterogeneous Graph for Company Similarity Quantification" (arXiv). All edge types and approaches are based on published academic research and public data sources.

**Goal**: Create a "CompanyKG-lite" that combines multiple similarity signals (industry, size, business model, geography, executives, M&A, news) to enable more accurate company comparability analysis.

---

## Current State

### Existing Company Schema

**Company Node Properties**:
- `cik` (STRING, unique) - SEC Central Index Key
- `ticker` (STRING, indexed) - Stock ticker symbol
- `name` (STRING) - Company name
- `description` (STRING) - Business description
- `description_embedding` (LIST[FLOAT]) - OpenAI embedding vector
- `embedding_model` (STRING) - Model used for embedding
- `embedding_dimension` (INTEGER) - Embedding dimension
- `loaded_at` (DATETIME) - When node was created

**Existing Relationships**:
- `(Company)-[:HAS_DOMAIN]->(Domain)` - Links company to its domain
- `(Company)-[:SIMILAR_DESCRIPTION {score}]->(Company)` - Cosine similarity on embeddings
- `(Company)-[:SIMILAR_TECHNOLOGY {score}]->(Company)` - Jaccard similarity on technology sets

---

## Proposed Schema Extensions

### New Company Node Properties

| Property | Type | Source | Description |
|----------|------|--------|-------------|
| `sic_code` | STRING | SEC EDGAR | Standard Industrial Classification code |
| `naics_code` | STRING | SEC EDGAR | North American Industry Classification System code |
| `sector` | STRING | Yahoo Finance / Wikidata | GICS sector (e.g., "Technology", "Finance") |
| `industry` | STRING | Yahoo Finance / Wikidata | GICS industry (e.g., "Software", "Banks") |
| `market_cap` | FLOAT | Yahoo Finance / SEC | Market capitalization (in USD) |
| `revenue` | FLOAT | SEC XBRL | Annual revenue (in USD) |
| `employees` | INTEGER | SEC 10-K / Wikidata | Number of employees |
| `headquarters_city` | STRING | SEC / Wikidata | HQ city |
| `headquarters_state` | STRING | SEC / Wikidata | HQ state |
| `headquarters_country` | STRING | SEC / Wikidata | HQ country (default: "US") |
| `business_model` | STRING | Derived from description | B2B, B2C, B2G, or mixed |
| `keywords` | LIST[STRING] | Extracted from 10-K | Key business terms/phrases |
| `founded_year` | INTEGER | Wikidata / Wikipedia | Year company was founded |
| `data_source` | STRING | Metadata | Source of data (e.g., "SEC_EDGAR", "YAHOO_FINANCE") |
| `data_updated_at` | DATETIME | Metadata | When property was last updated |

### New Relationship Types

| Relationship | Type | Properties | Description |
|--------------|------|------------|-------------|
| `SIMILAR_INDUSTRY` | `(Company)-[:SIMILAR_INDUSTRY]->(Company)` | `{method: "SIC"\|"NAICS"\|"SECTOR"\|"INDUSTRY", computed_at: DATETIME}` | Companies in same industry classification |
| `SIMILAR_SIZE` | `(Company)-[:SIMILAR_SIZE]->(Company)` | `{method: "REVENUE"\|"MARKET_CAP"\|"EMPLOYEES"\|"COMPOSITE", score: FLOAT, computed_at: DATETIME}` | Companies of similar financial/operational size |
| `SIMILAR_KEYWORDS` | `(Company)-[:SIMILAR_KEYWORDS]->(Company)` | `{score: FLOAT, shared_count: INTEGER, computed_at: DATETIME}` | Companies sharing business keywords |
| `SIMILAR_MARKET` | `(Company)-[:SIMILAR_MARKET]->(Company)` | `{method: "HQ_REGION"\|"REVENUE_SEGMENT", computed_at: DATETIME}` | Companies operating in same geographic markets |
| `COMMON_EXECUTIVE` | `(Company)-[:COMMON_EXECUTIVE]->(Company)` | `{executive_name: STRING, role: STRING, computed_at: DATETIME}` | Companies sharing a board member or executive |
| `MERGED_OR_ACQUIRED` | `(Company)-[:MERGED_OR_ACQUIRED]->(Company)` | `{year: INTEGER, type: "ACQUISITION"\|"MERGER", computed_at: DATETIME}` | M&A relationship between companies |
| `NEWS_MENTIONED_WITH` | `(Company)-[:NEWS_MENTIONED_WITH]->(Company)` | `{co_mention_count: INTEGER, computed_at: DATETIME}` | Companies frequently mentioned together in news |

**Note**: All relationships are undirected (bidirectional) for consistency with CompanyKG paper approach.

---

## Implementation Phases

### Phase 1: Foundation - Company Property Enrichment

**Goal**: Enrich Company nodes with basic attributes needed for similarity computation.

**Tasks**:
1. Create `scripts/enrich_company_properties.py` to:
   - Fetch SIC/NAICS codes from SEC EDGAR API
   - Fetch sector/industry from Yahoo Finance (yfinance library)
   - Fetch market cap, revenue, employees from Yahoo Finance / SEC
   - Fetch HQ location from SEC / Wikidata
   - Store in unified cache (namespace: `company_properties`)
   - Update Company nodes in Neo4j

2. Create `domain_status_graph/company/enrichment.py` with:
   - `fetch_sec_company_info(cik: str) -> dict` - SEC EDGAR API client
   - `fetch_yahoo_finance_info(ticker: str) -> dict` - yfinance wrapper
   - `fetch_wikidata_info(ticker: str, name: str) -> dict` - Wikidata SPARQL queries
   - `normalize_industry_codes(sic: str, naics: str) -> dict` - Standardize codes

3. Update `domain_status_graph/neo4j/constraints.py`:
   - Add indexes for new properties: `sector`, `industry`, `sic_code`, `naics_code`

**Data Sources**:
- SEC EDGAR API: https://www.sec.gov/edgar/sec-api-documentation
- Yahoo Finance (via yfinance): Already in requirements.txt
- Wikidata SPARQL: https://query.wikidata.org/

**Citations**:
- SEC EDGAR data is public domain (17 CFR 240.12g-1)
- Wikidata content is CC0 (public domain)

---

### Phase 2: Industry & Size Similarity

**Goal**: Create `SIMILAR_INDUSTRY` and `SIMILAR_SIZE` relationships.

**Tasks**:
1. Create `scripts/compute_industry_similarity.py`:
   - Query all Company nodes
   - Group by SIC code (exact match)
   - Group by NAICS code (exact match)
   - Group by sector (exact match)
   - Group by industry (exact match)
   - Create `SIMILAR_INDUSTRY` relationships for each group
   - Use dry-run pattern

2. Create `scripts/compute_size_similarity.py`:
   - Query all Company nodes with revenue/market_cap/employees
   - Bucket companies into size tiers (e.g., <$100M, $100M-$1B, $1B-$10B, >$10B)
   - Create `SIMILAR_SIZE` relationships within same bucket
   - Optionally: Create weighted edges based on numeric closeness (e.g., revenue within 2x)

3. Create `domain_status_graph/company/similarity.py` with:
   - `compute_industry_similarity(driver, database, execute)` - Industry grouping logic
   - `compute_size_similarity(driver, database, execute, method="COMPOSITE")` - Size bucketing logic

**Edge Creation Strategy**:
- Use MERGE to ensure idempotency
- Store `computed_at` timestamp
- Store `method` property to indicate which classification was used

---

### Phase 3: Business Model & Keywords

**Goal**: Extract business keywords and create `SIMILAR_KEYWORDS` relationships.

**Tasks**:
1. Create `scripts/extract_company_keywords.py`:
   - For each Company with description, extract key phrases
   - Use simple NLP (spaCy or NLTK) or LLM-based extraction
   - Store keywords in Company.keywords property
   - Cache keyword extraction results

2. Create `scripts/compute_keyword_similarity.py`:
   - Compute Jaccard similarity on keyword sets between companies
   - Create `SIMILAR_KEYWORDS` relationships above threshold (e.g., >0.1)
   - Store `shared_count` property

3. Create `domain_status_graph/company/keywords.py` with:
   - `extract_keywords(text: str, method: str = "spacy") -> List[str]` - Keyword extraction
   - `compute_keyword_similarity(companies: List[Company]) -> List[Tuple]` - Similarity computation

**Data Sources**:
- Company descriptions from existing cache
- SEC 10-K Item 1 (Business Description) - if we want to expand beyond current descriptions

**Citations**:
- Keyword extraction approach inspired by CompanyKG ET9 (shared keywords)

---

### Phase 4: Geography & Market Overlap

**Goal**: Create `SIMILAR_MARKET` relationships based on geographic presence.

**Tasks**:
1. Enhance `scripts/enrich_company_properties.py`:
   - Extract geographic segments from SEC 10-K filings (if available)
   - Classify as Domestic vs International focus
   - Store in Company properties

2. Create `scripts/compute_market_similarity.py`:
   - Group companies by HQ region (e.g., "US-West", "US-East", "US-South")
   - Create `SIMILAR_MARKET` relationships for same region
   - Optionally: Link companies with similar geographic revenue breakdowns

3. Create `domain_status_graph/company/geography.py` with:
   - `classify_hq_region(city: str, state: str) -> str` - Regional classification
   - `extract_geographic_segments(sec_filing_text: str) -> dict` - Parse 10-K geographic data

**Data Sources**:
- HQ location from SEC / Wikidata (already in Phase 1)
- Geographic segments from SEC 10-K filings (requires parsing)

---

### Phase 5: Executive & Board Overlap

**Goal**: Create `COMMON_EXECUTIVE` relationships.

**Tasks**:
1. Create `scripts/collect_executive_data.py`:
   - Query Wikidata for board members and executives
   - Use SPARQL to find people with multiple company affiliations
   - Store in cache (namespace: `executives`)
   - Store executive-company mappings

2. Create `scripts/compute_executive_overlap.py`:
   - For each person with multiple company affiliations
   - Create `COMMON_EXECUTIVE` relationships between those companies
   - Store `executive_name` and `role` properties

3. Create `domain_status_graph/company/executives.py` with:
   - `query_wikidata_executives(ticker: str) -> List[dict]` - Wikidata SPARQL queries
   - `find_common_executives(companies: List[str]) -> List[dict]` - Find overlaps

**Data Sources**:
- Wikidata SPARQL endpoint (free, public domain)
- SEC DEF 14A (proxy statements) - more complex, optional

**Citations**:
- Executive overlap approach from CompanyKG ET15 (common executives)

---

### Phase 6: M&A Relationships

**Goal**: Create `MERGED_OR_ACQUIRED` relationships.

**Tasks**:
1. Create `scripts/collect_ma_data.py`:
   - Scrape Wikipedia "List of acquisitions by [Company]" pages
   - Parse acquisition lists (e.g., "List of acquisitions by Google")
   - Store in cache (namespace: `mergers_acquisitions`)
   - Map company names to CIKs/tickers

2. Create `scripts/compute_ma_relationships.py`:
   - Load M&A data from cache
   - Create `MERGED_OR_ACQUIRED` relationships
   - Store `year` and `type` properties

3. Create `domain_status_graph/company/mergers.py` with:
   - `scrape_wikipedia_acquisitions(company_name: str) -> List[dict]` - Wikipedia scraping
   - `normalize_company_name(name: str) -> str` - Name matching to CIK/ticker

**Data Sources**:
- Wikipedia acquisition lists (Creative Commons licensed)
- SEC 8-K filings (acquisition announcements) - optional, more complex

**Citations**:
- M&A relationship approach from CompanyKG ET11 (historical M&A events)

---

### Phase 7: News Co-Mentions

**Goal**: Create `NEWS_MENTIONED_WITH` relationships (optional, advanced).

**Tasks**:
1. Create `scripts/collect_news_co_mentions.py`:
   - Use NewsAPI or GDELT to find co-mentions
   - Query for "Company A AND Company B" patterns
   - Store co-mention counts in cache

2. Create `scripts/compute_news_similarity.py`:
   - Create `NEWS_MENTIONED_WITH` relationships above threshold
   - Store `co_mention_count` property

**Data Sources**:
- NewsAPI (free tier: 100 requests/day)
- GDELT Project (open data, requires filtering)

**Note**: This phase is lower priority due to API limitations and complexity.

---

### Phase 8: LLM Judge for Ranking Validation

**Goal**: Use LLM to validate and tune similarity rankings.

**Tasks**:
1. Create `scripts/generate_golden_rankings.py`:
   - Select 10-20 well-known companies as seeds
   - Use GPT-4 to generate "ideal" top-5 similar companies for each
   - Store in cache (namespace: `golden_rankings`)
   - Include LLM justifications

2. Create `scripts/validate_similarity_rankings.py`:
   - Query graph for top-N similar companies for each seed
   - Compare with LLM golden rankings
   - Compute metrics (precision@k, recall@k)
   - Generate report

3. Create `scripts/tune_similarity_weights.py`:
   - Adjust edge weights based on LLM feedback
   - Test different weighting schemes
   - Document optimal weights

4. Create `domain_status_graph/company/llm_judge.py` with:
   - `generate_golden_ranking(company: Company, llm_client) -> List[dict]` - LLM ranking
   - `compare_rankings(graph_ranking: List, golden_ranking: List) -> dict` - Evaluation metrics

**LLM Usage**:
- Use OpenAI GPT-4 for golden rankings (one-time cost)
- Store results in cache to avoid repeated API calls
- Include prompts in documentation for reproducibility

**Citations**:
- LLM judge approach inspired by CompanyKG evaluation methodology

---

## Integration with Existing Pipeline

### Update `scripts/run_all_pipelines.py`

Add new steps after company data loading:

```python
# Step 2.4: Enrich Company Properties
run_script(ENRICH_COMPANY_PROPERTIES_SCRIPT, execute=True, ...)

# Step 2.5: Compute Industry Similarity
run_script(COMPUTE_INDUSTRY_SIMILARITY_SCRIPT, execute=True, ...)

# Step 2.6: Compute Size Similarity
run_script(COMPUTE_SIZE_SIMILARITY_SCRIPT, execute=True, ...)

# Step 2.7: Extract Keywords & Compute Keyword Similarity
run_script(EXTRACT_KEYWORDS_SCRIPT, execute=True, ...)
run_script(COMPUTE_KEYWORD_SIMILARITY_SCRIPT, execute=True, ...)

# Step 2.8: Compute Market Similarity
run_script(COMPUTE_MARKET_SIMILARITY_SCRIPT, execute=True, ...)

# Step 2.9: Collect & Compute Executive Overlap
run_script(COLLECT_EXECUTIVE_DATA_SCRIPT, execute=True, ...)
run_script(COMPUTE_EXECUTIVE_OVERLAP_SCRIPT, execute=True, ...)

# Step 2.10: Collect & Compute M&A Relationships
run_script(COLLECT_MA_DATA_SCRIPT, execute=True, ...)
run_script(COMPUTE_MA_RELATIONSHIPS_SCRIPT, execute=True, ...)
```

### Update Schema Documentation

Update `docs/graph_schema.md` to include:
- New Company properties
- New relationship types with examples
- Data source citations

---

## Data Provenance & Citations

### Documentation Requirements

1. **Data Source Inventory** (`docs/DATA_SOURCES.md`):
   - List all data sources with URLs
   - Document licensing (public domain, CC0, etc.)
   - Include API endpoints and rate limits
   - Note data collection dates

2. **Code Comments**:
   - Add citations in docstrings
   - Reference CompanyKG paper for edge type justifications
   - Include data source URLs in comments

3. **README Updates**:
   - Add "Data Sources" section
   - Include bibliography/references section
   - Note that all data is from public sources

### Example Citation Format

```python
"""
Extract company industry classification from SEC EDGAR.

Data Source: SEC EDGAR API (https://www.sec.gov/edgar/sec-api-documentation)
License: Public domain (17 CFR 240.12g-1)

Reference: CompanyKG paper - Industry classification edges (C2: industry sector similarity)
"""
```

---

## Similarity Scoring & Querying

### Composite Similarity Score

Create a function to compute overall similarity by aggregating all edge types:

```cypher
// Example: Find top similar companies with composite score
MATCH (c1:Company {ticker: 'AAPL'})-[r]-(c2:Company)
WHERE type(r) IN ['SIMILAR_INDUSTRY', 'SIMILAR_SIZE', 'SIMILAR_KEYWORDS',
                  'SIMILAR_MARKET', 'COMMON_EXECUTIVE', 'MERGED_OR_ACQUIRED']
WITH c2,
     count(r) as edge_count,
     sum(CASE type(r)
         WHEN 'SIMILAR_INDUSTRY' THEN 1.0
         WHEN 'SIMILAR_SIZE' THEN 0.8
         WHEN 'SIMILAR_KEYWORDS' THEN r.score
         WHEN 'SIMILAR_MARKET' THEN 0.6
         WHEN 'COMMON_EXECUTIVE' THEN 0.9
         WHEN 'MERGED_OR_ACQUIRED' THEN 0.7
         ELSE 0.0
     END) as composite_score
RETURN c2.name, c2.ticker, edge_count, composite_score
ORDER BY composite_score DESC, edge_count DESC
LIMIT 10
```

### GDS Integration

Consider using GDS Node Similarity on a projected Company-Company graph:
- Project all SIMILAR_* relationships into a homogeneous graph
- Run Node Similarity to find additional indirect connections
- Store results as `SIMILAR_COMPANY {score}` relationships

---

## Testing & Validation

### Unit Tests

- Test each data fetching function (mocked API responses)
- Test similarity computation logic
- Test edge creation with sample data

### Integration Tests

- Test full pipeline with small dataset (10-20 companies)
- Verify idempotency (run twice, same results)
- Validate data quality (no nulls where required, valid ranges)

### Validation Queries

Create validation queries to check:
- Coverage: % of companies with each property type
- Relationship counts: Expected number of edges per type
- Data freshness: Age of data sources

---

## Next Steps

1. **Start with Phase 1** (Company Property Enrichment):
   - Implement SEC EDGAR API client
   - Implement Yahoo Finance integration
   - Test with 10-20 companies
   - Verify data quality

2. **Incremental Rollout**:
   - Complete one phase before starting next
   - Validate results at each phase
   - Document learnings and adjustments

3. **Prioritization**:
   - **High Priority**: Phases 1-3 (Industry, Size, Keywords) - Most impactful
   - **Medium Priority**: Phases 4-5 (Geography, Executives) - Good signals
   - **Lower Priority**: Phases 6-7 (M&A, News) - Nice to have
   - **Validation**: Phase 8 (LLM Judge) - Important for tuning

---

## References

1. **CompanyKG Paper**: "CompanyKG: A Large-Scale Heterogeneous Graph for Company Similarity Quantification" (arXiv)
2. **SEC EDGAR API**: https://www.sec.gov/edgar/sec-api-documentation
3. **Wikidata SPARQL**: https://query.wikidata.org/
4. **Yahoo Finance (yfinance)**: https://github.com/ranaroussi/yfinance
5. **GDELT Project**: https://www.gdeltproject.org/

---

*Last Updated: 2024-12-XX*

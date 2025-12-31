# Public Company Graph Schema Documentation

## Overview

The Public Company Graph is a knowledge graph modeling **public companies**, their **domains**, and **technology stacks**. The graph combines SEC filing data with domain intelligence to enable competitive analysis, technology adoption prediction, and business relationship mapping.

**Primary Use Cases**:
- Competitive landscape analysis (who competes with whom)
- Business relationship mapping (customers, suppliers, partners)
- Company similarity (by description, industry, size, risk profile)
- Technology adoption prediction and affinity analysis

**Data Sources**:
- SEC EDGAR (10-K filings, company metadata)
- Yahoo Finance (market data, sector/industry classification)
- Domain intelligence (`domain_status` tool for technology detection)

**Graph Database**: Neo4j (database: `domain`)

---

## Data Quality Notes

> **Important**: Not all properties are populated for all companies. See coverage below:

| Property | Coverage | Notes |
|----------|----------|-------|
| `ticker` | 100% (5,398/5,398) | From SEC EDGAR |
| `name` | 100% (5,398/5,398) | From SEC EDGAR |
| `description` | 99.85% (5,390/5,398) | From 10-K Item 1 |
| `description_embedding` | 99.85% (5,390/5,398) | OpenAI text-embedding-3-small |
| `sector` / `industry` | ~18% (959/5,398) | Yahoo Finance (only actively traded stocks) |
| `market_cap` | ~18% (960/5,398) | Yahoo Finance (only actively traded stocks) |
| `revenue` | ~15% (815/5,398) | Yahoo Finance (only actively traded stocks) |

**Missing Descriptions (8 companies)**: These are edge cases:
- BNS (Bank of Nova Scotia) - Canadian company, files 20-F not 10-K
- GJR, GJS, GJT (STRATS Trusts) - Securities trusts, not operating companies
- KIDZ, YHGJ, ESP, MNGG - Non-standard 10-K formats

**Technology Data**: Only **web technologies** are captured (JavaScript, CMS, CDN, etc.)—not backend infrastructure like Kubernetes or Docker. This is because technology detection is based on HTTP fingerprinting of company domains.

---

## Graph Statistics

| Metric | Count |
|--------|-------|
| **Total Nodes** | 10,562 |
| - Company | 5,398 |
| - Domain | 4,337 |
| - Technology | 827 |
| **Total Relationships** | ~2,033,384 |
| - Similarity relationships | 1,890,697 |
|   - SIMILAR_INDUSTRY | 520,672 |
|   - SIMILAR_DESCRIPTION | 436,973 |
|   - SIMILAR_SIZE | 414,096 |
|   - SIMILAR_RISK | 394,372 |
|   - SIMILAR_TECHNOLOGY | 124,584 |
| - Business relationships | 10,293 |
|   - HAS_COMPETITOR | 3,843 |
|   - HAS_SUPPLIER | 2,597 |
|   - HAS_PARTNER | 2,139 |
|   - HAS_CUSTOMER | 1,714 |
| - Technology relationships | 128,551 |
|   - USES | 46,081 |
|   - LIKELY_TO_ADOPT | 41,250 |
|   - CO_OCCURS_WITH | 41,220 |
| - Domain-Company links | 3,745 |

---

## Node Types

### Company

**Label**: `Company`

**Description**: A publicly traded company that files with the SEC. This is the primary entity for business intelligence and competitive analysis.

**Unique Identifier**: `cik` (SEC Central Index Key)

**Properties**:

| Property | Type | Required | Description | Example |
|----------|------|----------|-------------|---------|
| `cik` | STRING | ✅ | SEC Central Index Key (unique) | `0000320193` |
| `name` | STRING | ✅ | Company legal name | `Apple Inc.` |
| `ticker` | STRING | ❌ | Stock ticker symbol | `AAPL` |
| `sector` | STRING | ❌ | Business sector | `Technology` |
| `industry` | STRING | ❌ | Industry classification | `Consumer Electronics` |
| `sic_code` | STRING | ❌ | SIC industry code | `3571` |
| `description` | STRING | ❌ | Business description from 10-K | (Item 1 text) |
| `description_source` | STRING | ❌ | How description was extracted | `datamule`, `custom_parser` |
| `description_embedding` | LIST<FLOAT> | ❌ | OpenAI embedding vector | 1536-dim vector |
| `embedding_model` | STRING | ❌ | Embedding model used | `text-embedding-3-small` |
| `embedding_dimension` | INTEGER | ❌ | Embedding vector dimension | `1536` |
| `revenue` | INTEGER | ❌ | Annual revenue (USD) | `416161005568` |
| `market_cap` | INTEGER | ❌ | Market capitalization (USD) | `4058697957376` |
| `employees` | INTEGER | ❌ | Number of employees | `166000` |
| `headquarters_city` | STRING | ❌ | HQ city | `Cupertino` |
| `headquarters_state` | STRING | ❌ | HQ state | `CA` |
| `headquarters_country` | STRING | ❌ | HQ country | `United States` |
| `filing_date` | STRING | ❌ | Latest 10-K filing date | `2021-10-29` |
| `filing_year` | INTEGER | ❌ | Fiscal year of filing | `2021` |
| `fiscal_year_end` | STRING | ❌ | Fiscal year end date | `2021-09-25` |
| `accession_number` | STRING | ❌ | SEC filing accession number | `0000320193-21-000105` |
| `sec_filing_url` | STRING | ❌ | URL to SEC filing | `https://www.sec.gov/...` |
| `data_source` | STRING | ❌ | Data provenance | `SEC_EDGAR,YAHOO_FINANCE` |
| `data_updated_at` | STRING | ❌ | Last data refresh | `2025-12-29T19:16:06Z` |
| `loaded_at` | DATETIME | ✅ | When node was loaded | `2025-12-30T19:29:27Z` |

**Relationships**:
- `(Company)-[:HAS_DOMAIN]->(Domain)` - Company's web domain
- `(Company)-[:HAS_COMPETITOR]->(Company)` - Cited competitors from 10-K
- `(Company)-[:HAS_CUSTOMER]->(Company)` - Cited customers from 10-K
- `(Company)-[:HAS_SUPPLIER]->(Company)` - Cited suppliers from 10-K
- `(Company)-[:HAS_PARTNER]->(Company)` - Cited partners from 10-K
- `(Company)-[:SIMILAR_DESCRIPTION]->(Company)` - Similar business descriptions
- `(Company)-[:SIMILAR_INDUSTRY]->(Company)` - Same industry/sector
- `(Company)-[:SIMILAR_SIZE]->(Company)` - Similar revenue/market cap
- `(Company)-[:SIMILAR_RISK]->(Company)` - Similar risk factors
- `(Company)-[:SIMILAR_TECHNOLOGY]->(Company)` - Similar technology stacks

---

### Domain

**Label**: `Domain`

**Description**: A web domain (e.g., `apple.com`) with associated technology detection and metadata.

**Unique Identifier**: `final_domain` (normalized domain name, lowercase, no `www.`)

**Properties**:

| Property | Type | Required | Description | Example |
|----------|------|----------|-------------|---------|
| `final_domain` | STRING | ✅ | Normalized domain (unique key) | `apple.com` |
| `domain` | STRING | ❌ | Original domain from input | `www.apple.com` |
| `initial_domain` | STRING | ❌ | Initial domain before redirects | `apple.com` |
| `http_status` | INTEGER | ❌ | HTTP status code | `200` |
| `http_status_text` | STRING | ❌ | HTTP status text | `OK` |
| `status` | INTEGER | ❌ | Status code (legacy) | `200` |
| `status_description` | STRING | ❌ | Status description | `OK` |
| `response_time` | FLOAT | ❌ | Response time (seconds) | `0.226` |
| `response_time_seconds` | FLOAT | ❌ | Response time (seconds) | `0.226` |
| `timestamp` | INTEGER | ❌ | Unix timestamp of check | `1735689600` |
| `observed_at_ms` | INTEGER | ❌ | Observation time (ms) | `1735689600000` |
| `is_mobile_friendly` | INTEGER | ❌ | Mobile-friendly flag (0/1) | `1` |
| `title` | STRING | ❌ | HTML page title | `Apple` |
| `description` | STRING | ❌ | Meta description | `Discover the innovative...` |
| `dmarc_record` | STRING | ❌ | DMARC record | `v=DMARC1; p=quarantine` |
| `creation_date` | INTEGER | ❌ | Domain registration (Unix) | `1735689600` |
| `creation_date_ms` | INTEGER | ❌ | Registration (ms) | `1735689600000` |
| `expiration_date` | INTEGER | ❌ | Domain expiration (Unix) | `1767225600` |
| `expiration_date_ms` | INTEGER | ❌ | Expiration (ms) | `1767225600000` |
| `registrar` | STRING | ❌ | Domain registrar | `CSC Corporate Domains` |
| `registrant_org` | STRING | ❌ | Registrant organization | `Apple Inc.` |
| `loaded_at` | DATETIME | ✅ | When node was loaded | `2025-12-30T19:29:27Z` |

**Relationships**:
- `(Domain)-[:USES]->(Technology)` - Technologies detected on domain
- `(Domain)-[:LIKELY_TO_ADOPT]->(Technology)` - Predicted technology adoption
- `(Company)-[:HAS_DOMAIN]->(Domain)` - Company that owns this domain

---

### Technology

**Label**: `Technology`

**Description**: A technology, framework, library, or service detected on domains (e.g., `WordPress`, `React`, `Cloudflare`).

**Unique Identifier**: `name`

**Properties**:

| Property | Type | Required | Description | Example |
|----------|------|----------|-------------|---------|
| `name` | STRING | ✅ | Technology name (unique) | `WordPress` |
| `category` | STRING | ❌ | Technology category | `CMS` |
| `loaded_at` | DATETIME | ✅ | When node was loaded | `2025-12-30T19:29:27Z` |

**Technology Categories** (60+ categories):
- Analytics: Google Analytics, Adobe Analytics
- CMS: WordPress, Drupal, Contentful
- JavaScript frameworks: React, Vue.js, Angular
- JavaScript libraries: jQuery, Lodash
- CDN: Cloudflare, Akamai, Fastly
- E-commerce: Shopify, Magento, WooCommerce
- Marketing automation: HubSpot, Marketo, Pardot

**Relationships**:
- `(Domain)-[:USES]->(Technology)` - Domains using this technology
- `(Technology)-[:CO_OCCURS_WITH]->(Technology)` - Technologies that appear together
- `(Domain)-[:LIKELY_TO_ADOPT]->(Technology)` - Predicted adoption

---

## Relationship Types

### Business Relationships (from 10-K filings)

#### HAS_COMPETITOR

**Pattern**: `(Company)-[:HAS_COMPETITOR]->(Company)`

**Description**: Company A explicitly mentioned Company B as a competitor in their 10-K filing. Directional - reverse may not exist.

**Count**: 3,843

**Properties**:
| Property | Type | Description |
|----------|------|-------------|
| `confidence` | FLOAT | Entity resolution confidence (0-1) |
| `confidence_tier` | STRING | Confidence bucket (`high`, `medium`, `low`) |
| `raw_mention` | STRING | How competitor was mentioned |
| `context` | STRING | Surrounding text from filing |
| `source` | STRING | Always `ten_k_filing` |
| `source_cik` | STRING | CIK of company making the citation |
| `target_cik` | STRING | CIK of cited company |
| `is_mutual` | BOOLEAN | Whether the relationship is bidirectional |
| `inbound_citations` | INTEGER | How many companies cite this competitor |
| `extracted_at` | DATETIME | When relationship was extracted |

**Example**:
```cypher
// Find NVIDIA's cited competitors
MATCH (c:Company {ticker:'NVDA'})-[r:HAS_COMPETITOR]->(comp:Company)
RETURN comp.ticker, comp.name, r.confidence, r.raw_mention
ORDER BY r.confidence DESC

// Find mutual competitors
MATCH (a:Company)-[:HAS_COMPETITOR]->(b:Company)-[:HAS_COMPETITOR]->(a)
WHERE a.ticker < b.ticker
RETURN a.ticker, b.ticker
```

---

#### HAS_CUSTOMER

**Pattern**: `(Company)-[:HAS_CUSTOMER]->(Company)`

**Description**: Company A mentioned Company B as a customer in their 10-K filing.

**Count**: 1,714

**Properties**: Same as HAS_COMPETITOR

**Example**:
```cypher
// Find Microsoft's major customers
MATCH (c:Company {ticker:'MSFT'})-[r:HAS_CUSTOMER]->(cust:Company)
RETURN cust.name, r.raw_mention
ORDER BY r.confidence DESC
```

---

#### HAS_SUPPLIER

**Pattern**: `(Company)-[:HAS_SUPPLIER]->(Company)`

**Description**: Company A mentioned Company B as a supplier/vendor in their 10-K filing.

**Count**: 2,597

**Properties**: Same as HAS_COMPETITOR

**Example**:
```cypher
// Find Apple's suppliers
MATCH (c:Company {ticker:'AAPL'})-[r:HAS_SUPPLIER]->(supp:Company)
RETURN supp.name, r.raw_mention
```

---

#### HAS_PARTNER

**Pattern**: `(Company)-[:HAS_PARTNER]->(Company)`

**Description**: Company A mentioned Company B as a partner/alliance in their 10-K filing.

**Count**: 2,139

**Properties**: Same as HAS_COMPETITOR

---

### Similarity Relationships

#### SIMILAR_DESCRIPTION

**Pattern**: `(Company)-[:SIMILAR_DESCRIPTION]->(Company)`

**Description**: Companies with similar business descriptions based on embedding cosine similarity.

**Count**: 420,531

**Properties**:
| Property | Type | Description |
|----------|------|-------------|
| `score` | FLOAT | Cosine similarity (0-1, higher = more similar) |
| `metric` | STRING | `COSINE` |
| `computed_at` | DATETIME | When similarity was computed |

**Example**:
```cypher
MATCH (c1:Company {ticker: 'AAPL'})-[r:SIMILAR_DESCRIPTION]->(c2:Company)
RETURN c2.name, c2.ticker, r.score
ORDER BY r.score DESC LIMIT 10
```

---

#### SIMILAR_INDUSTRY

**Pattern**: `(Company)-[:SIMILAR_INDUSTRY]->(Company)`

**Description**: Companies in the same industry and/or sector.

**Count**: 520,672

**Properties**:
| Property | Type | Description |
|----------|------|-------------|
| `score` | FLOAT | Similarity score |
| `classification` | STRING | How companies matched (`same_industry`, `same_sector`) |
| `method` | STRING | Classification method |
| `computed_at` | DATETIME | When computed |

---

#### SIMILAR_SIZE

**Pattern**: `(Company)-[:SIMILAR_SIZE]->(Company)`

**Description**: Companies of similar size (revenue, market cap, employees).

**Count**: 414,096

**Properties**:
| Property | Type | Description |
|----------|------|-------------|
| `score` | FLOAT | Size similarity score |
| `bucket` | STRING | Size bucket (`micro`, `small`, `mid`, `large`, `mega`) |
| `method` | STRING | Bucketing method |
| `metric` | STRING | Metric used (`revenue`, `market_cap`, `employees`) |
| `computed_at` | DATETIME | When computed |

---

#### SIMILAR_RISK

**Pattern**: `(Company)-[:SIMILAR_RISK]->(Company)`

**Description**: Companies with similar risk factor profiles based on 10-K Item 1A.

**Count**: 394,372

**Properties**:
| Property | Type | Description |
|----------|------|-------------|
| `score` | FLOAT | Risk profile similarity (0-1) |
| `metric` | STRING | `COSINE` |
| `computed_at` | DATETIME | When computed |

---

#### SIMILAR_TECHNOLOGY

**Pattern**: `(Company)-[:SIMILAR_TECHNOLOGY]->(Company)`

**Description**: Companies using similar technology stacks (via their domains).

**Count**: 124,584

**Properties**:
| Property | Type | Description |
|----------|------|-------------|
| `score` | FLOAT | Technology Jaccard similarity |
| `metric` | STRING | `JACCARD` |
| `computed_at` | DATETIME | When computed |

---

### Technology Relationships

#### USES

**Pattern**: `(Domain)-[:USES]->(Technology)`

**Description**: Indicates a domain uses a specific technology.

**Count**: 46,081

**Properties**:
| Property | Type | Description |
|----------|------|-------------|
| `loaded_at` | DATETIME | When relationship was created |

**Example**:
```cypher
MATCH (d:Domain {final_domain: 'apple.com'})-[:USES]->(t:Technology)
RETURN t.name, t.category
ORDER BY t.category
```

---

#### LIKELY_TO_ADOPT

**Pattern**: `(Domain)-[:LIKELY_TO_ADOPT]->(Technology)`

**Description**: GDS-predicted technologies a domain is likely to adopt, based on Personalized PageRank.

**Count**: 41,250

**Properties**:
| Property | Type | Description |
|----------|------|-------------|
| `score` | FLOAT | Adoption likelihood score |
| `computed_at` | DATETIME | When prediction was made |

---

#### CO_OCCURS_WITH

**Pattern**: `(Technology)-[:CO_OCCURS_WITH]->(Technology)`

**Description**: Technologies that frequently appear together on domains (Jaccard similarity).

**Count**: 41,220

**Properties**:
| Property | Type | Description |
|----------|------|-------------|
| `similarity` | FLOAT | Jaccard similarity (0-1) |
| `metric` | STRING | `JACCARD` |
| `computed_at` | DATETIME | When computed |

---

### Domain-Company Link

#### HAS_DOMAIN

**Pattern**: `(Company)-[:HAS_DOMAIN]->(Domain)`

**Description**: Links a company to its primary web domain.

**Count**: 3,745

**Properties**:
| Property | Type | Description |
|----------|------|-------------|
| `loaded_at` | DATETIME | When link was created |

**Example**:
```cypher
MATCH (c:Company {ticker:'AAPL'})-[:HAS_DOMAIN]->(d:Domain)
RETURN d.final_domain, d.title
```

---

## Constraints and Indexes

### Constraints

```cypher
CREATE CONSTRAINT company_cik IF NOT EXISTS FOR (c:Company) REQUIRE c.cik IS UNIQUE;
CREATE CONSTRAINT domain_name IF NOT EXISTS FOR (d:Domain) REQUIRE d.final_domain IS UNIQUE;
CREATE CONSTRAINT technology_name IF NOT EXISTS FOR (t:Technology) REQUIRE t.name IS UNIQUE;
```

### Indexes

```cypher
-- Company indexes
CREATE INDEX company_ticker IF NOT EXISTS FOR (c:Company) ON (c.ticker);
CREATE INDEX company_sector IF NOT EXISTS FOR (c:Company) ON (c.sector);
CREATE INDEX company_industry IF NOT EXISTS FOR (c:Company) ON (c.industry);
CREATE INDEX company_sic_code IF NOT EXISTS FOR (c:Company) ON (c.sic_code);
CREATE INDEX company_naics_code IF NOT EXISTS FOR (c:Company) ON (c.naics_code);
CREATE INDEX company_filing_date IF NOT EXISTS FOR (c:Company) ON (c.filing_date);
CREATE INDEX company_filing_year IF NOT EXISTS FOR (c:Company) ON (c.filing_year);
CREATE INDEX company_accession_number IF NOT EXISTS FOR (c:Company) ON (c.accession_number);

-- Domain indexes
CREATE INDEX domain_domain IF NOT EXISTS FOR (d:Domain) ON (d.domain);

-- Relationship indexes
CREATE INDEX has_competitor_confidence IF NOT EXISTS FOR ()-[r:HAS_COMPETITOR]->() ON (r.confidence);
CREATE INDEX has_customer_confidence IF NOT EXISTS FOR ()-[r:HAS_CUSTOMER]->() ON (r.confidence);
CREATE INDEX has_supplier_confidence IF NOT EXISTS FOR ()-[r:HAS_SUPPLIER]->() ON (r.confidence);
CREATE INDEX has_partner_confidence IF NOT EXISTS FOR ()-[r:HAS_PARTNER]->() ON (r.confidence);
```

---

## Example Queries

### Find a company's competitive landscape
```cypher
MATCH (c:Company {ticker: 'NVDA'})
OPTIONAL MATCH (c)-[r:HAS_COMPETITOR]->(comp:Company)
OPTIONAL MATCH (c)-[:HAS_DOMAIN]->(d:Domain)-[:USES]->(t:Technology)
RETURN c.name,
       collect(DISTINCT comp.ticker) as competitors,
       collect(DISTINCT t.name) as technologies
```

### Find companies similar to Apple
```cypher
MATCH (apple:Company {ticker: 'AAPL'})-[r:SIMILAR_DESCRIPTION]->(similar:Company)
WHERE r.score > 0.8
RETURN similar.ticker, similar.name, similar.sector, r.score
ORDER BY r.score DESC
LIMIT 10
```

### Map supply chain relationships
```cypher
// Find suppliers and customers of a company
MATCH (c:Company {ticker: 'TSLA'})
OPTIONAL MATCH (c)-[:HAS_SUPPLIER]->(supp:Company)
OPTIONAL MATCH (c)-[:HAS_CUSTOMER]->(cust:Company)
RETURN c.name,
       collect(DISTINCT supp.name) as suppliers,
       collect(DISTINCT cust.name) as customers
```

### Technology adoption analysis
```cypher
// Find what technologies Apple's domain might adopt next
MATCH (c:Company {ticker:'AAPL'})-[:HAS_DOMAIN]->(d:Domain)
MATCH (d)-[r:LIKELY_TO_ADOPT]->(t:Technology)
WHERE NOT (d)-[:USES]->(t)
RETURN t.name, t.category, r.score
ORDER BY r.score DESC
LIMIT 10
```

### Find industry leaders
```cypher
// Most-cited competitors by industry
MATCH (c:Company)-[r:HAS_COMPETITOR]->(comp:Company)
WHERE comp.sector = 'Technology'
WITH comp, count(r) as citations
RETURN comp.ticker, comp.name, citations
ORDER BY citations DESC
LIMIT 20
```

---

## Data Loading Scripts

| Script | Purpose |
|--------|---------|
| `scripts/bootstrap_graph.py` | Load Domain and Technology nodes + USES relationships |
| `scripts/load_company_data.py` | Load Company nodes from SEC EDGAR |
| `scripts/extract_business_relationships.py` | Extract HAS_COMPETITOR/CUSTOMER/SUPPLIER/PARTNER from 10-Ks |
| `scripts/compute_gds_features.py` | Compute LIKELY_TO_ADOPT, CO_OCCURS_WITH |
| `scripts/compute_company_similarity.py` | Compute all SIMILAR_* relationships |

---

## Related Documentation

- **Architecture**: `architecture.md` - Package structure and design principles
- **High-Value Queries**: `money_queries.md` - Business intelligence queries
- **Step-by-Step Guide**: `step_by_step_guide.md` - Pipeline walkthrough

# Company Enrichment Status Queries

Use these Cypher queries to check the status of company property enrichment.

## Check Enrichment Progress

```cypher
// Count companies with enriched properties
MATCH (c:Company)
RETURN
  count(c) as total_companies,
  count(c.sector) as has_sector,
  count(c.industry) as has_industry,
  count(c.sic_code) as has_sic_code,
  count(c.market_cap) as has_market_cap,
  count(c.revenue) as has_revenue,
  count(c.employees) as has_employees,
  count(c.headquarters_city) as has_hq_city
```

## Find Companies with Full Enrichment

```cypher
// Companies with all major properties
MATCH (c:Company)
WHERE c.sector IS NOT NULL
  AND c.industry IS NOT NULL
  AND c.market_cap IS NOT NULL
  AND c.sic_code IS NOT NULL
RETURN c.ticker, c.name, c.sector, c.industry, c.market_cap
LIMIT 20
```

## Find Companies Missing Enrichment

```cypher
// Companies with no enrichment data
MATCH (c:Company)
WHERE c.sector IS NULL
  AND c.industry IS NULL
  AND c.sic_code IS NULL
RETURN c.ticker, c.name, c.cik
LIMIT 20
```

## Check Data Sources

```cypher
// See which data sources were used
MATCH (c:Company)
WHERE c.data_source IS NOT NULL
RETURN
  c.data_source,
  count(*) as company_count
ORDER BY company_count DESC
```

## Sample Enriched Company

```cypher
// View a fully enriched company
MATCH (c:Company)
WHERE c.sector IS NOT NULL
  AND c.industry IS NOT NULL
  AND c.market_cap IS NOT NULL
RETURN c
LIMIT 1
```

## Property Coverage Statistics

```cypher
// Detailed property coverage
MATCH (c:Company)
RETURN
  count(c) as total,
  sum(CASE WHEN c.sector IS NOT NULL THEN 1 ELSE 0 END) as has_sector,
  sum(CASE WHEN c.industry IS NOT NULL THEN 1 ELSE 0 END) as has_industry,
  sum(CASE WHEN c.sic_code IS NOT NULL THEN 1 ELSE 0 END) as has_sic,
  sum(CASE WHEN c.naics_code IS NOT NULL THEN 1 ELSE 0 END) as has_naics,
  sum(CASE WHEN c.market_cap IS NOT NULL THEN 1 ELSE 0 END) as has_market_cap,
  sum(CASE WHEN c.revenue IS NOT NULL THEN 1 ELSE 0 END) as has_revenue,
  sum(CASE WHEN c.employees IS NOT NULL THEN 1 ELSE 0 END) as has_employees,
  sum(CASE WHEN c.headquarters_city IS NOT NULL THEN 1 ELSE 0 END) as has_hq
```

# Company Similarity Expansion - Quick Summary

## What We're Adding

Expanding the graph from **tech-centric** relationships to **multi-faceted company similarity** using public data sources.

### New Company Properties
- Industry classification (SIC, NAICS, sector, industry)
- Financial metrics (market cap, revenue, employees)
- Geographic data (HQ location, market segments)
- Business keywords (extracted from descriptions)
- Executive/board data (from Wikidata)

### New Relationship Types
1. **SIMILAR_INDUSTRY** - Same SIC/NAICS/sector/industry
2. **SIMILAR_SIZE** - Comparable revenue/market cap/employees
3. **SIMILAR_KEYWORDS** - Shared business terms (Jaccard similarity)
4. **SIMILAR_MARKET** - Same geographic markets
5. **COMMON_EXECUTIVE** - Shared board members/executives
6. **MERGED_OR_ACQUIRED** - M&A relationships
7. **NEWS_MENTIONED_WITH** - Co-mentions in news (optional)

## Implementation Approach

**8 Phases** - Start with high-impact, low-complexity features:

1. ✅ **Phase 1**: Enrich Company properties (SEC, Yahoo Finance, Wikidata)
2. ✅ **Phase 2**: Industry & Size similarity (exact matches + bucketing)
3. ✅ **Phase 3**: Business keywords extraction & similarity
4. ⚠️ **Phase 4**: Geography/market overlap
5. ⚠️ **Phase 5**: Executive/board overlap (Wikidata)
6. ⚠️ **Phase 6**: M&A relationships (Wikipedia scraping)
7. ⚠️ **Phase 7**: News co-mentions (optional, complex)
8. ✅ **Phase 8**: LLM judge for validation & tuning

## Key Principles

- **Public Data Only**: SEC EDGAR, Yahoo Finance, Wikidata, Wikipedia
- **Reproducible**: All scripts follow dry-run pattern, cache results
- **Well-Cited**: Reference CompanyKG paper, document all data sources
- **Idempotent**: Re-running scripts produces same results
- **Incremental**: One phase at a time, validate before moving on

## Immediate Next Steps

1. **Create Phase 1 script structure**:
   ```bash
   domain_status_graph/company/
     ├── __init__.py
     ├── enrichment.py      # SEC, Yahoo Finance, Wikidata clients
     └── similarity.py       # Similarity computation functions

   scripts/
     ├── enrich_company_properties.py
     └── compute_industry_similarity.py
   ```

2. **Test with small dataset** (10-20 companies from S&P 500)

3. **Update schema documentation** as properties/relationships are added

## Data Sources Summary

| Source | What We Get | License | API/Rate Limits |
|--------|-------------|---------|-----------------|
| SEC EDGAR | SIC/NAICS, financials, filings | Public domain | 10 req/sec |
| Yahoo Finance | Sector, industry, market cap, revenue, employees | Free | Rate limited |
| Wikidata | HQ location, executives, employees | CC0 (public domain) | SPARQL endpoint |
| Wikipedia | M&A lists, company info | Creative Commons | Web scraping |
| NewsAPI/GDELT | News co-mentions | Free tier / Open data | Limited |

## Expected Outcomes

- **More accurate company comparability**: Multiple signals vs. just tech stack
- **Better sales targeting**: Find companies similar across many dimensions
- **Research value**: Reproducible "CompanyKG-lite" using only public data
- **Publication-ready**: Well-documented, cited, open-source

## Full Documentation

See `docs/COMPANY_SIMILARITY_EXPANSION.md` for complete implementation plan.

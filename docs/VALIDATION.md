# Company Similarity Validation

## Purpose

Validation helps us catch data quality issues and verify the similarity system works correctly. **Important**: The current validation results (32.6% pass rate) reveal that the graph has sparse/weak input data, not algorithm problems. This validation is useful for quick sanity checks but **should not be used for investment decisions** due to missing relationships.

---

## Famous Pairs Validation

**Quick smoke test** to verify known competitor pairs rank correctly.

### Script
- **File**: `scripts/validate_famous_pairs.py`
- **CLI**: `validate-famous-pairs`

### Usage

```bash
# Test all 46 pairs
validate-famous-pairs

# Quick test (first 10 pairs)
validate-famous-pairs --limit 10

# Save report to file
validate-famous-pairs --output validation_report.md
```

### What It Tests

Tests 46 famous competitor pairs (e.g., Coca-Cola → Pepsi, Apple → Microsoft) and verifies the target company ranks within the expected position (typically top-1 or top-3).

**Full list**: See `docs/FAMOUS_COMPETITOR_PAIRS.md` for complete corpus.

### Current Status

**Pass Rate**: 32.6% (15/46 pairs)

**Breakdown**:
- ✅ **Passed**: 15 pairs (32.6%)
- ❌ **Failed**: 17 pairs (ranked but not in expected position)
- ⚠️ **Not Found**: 14 pairs (30.4% - no relationships exist at all)

### What This Reveals

1. **Missing Relationships** (30.4%): Many pairs have no relationships at all - this is a **data quality issue**, not an algorithm problem.
2. **Sparse Data**: The input data (SEC filings, company enrichment) is patchy and weak.
3. **Weight Tuning Won't Help**: Optimization showed no improvement - the issue is missing data, not suboptimal weights.

### Limitations

- **Not for Investment Decisions**: The graph is missing too many relationships to be reliable for investment contexts.
- **Useful for Development**: Quick sanity check after code changes.
- **Reveals Data Gaps**: Helps identify which companies/pairs need better data sources.

---

## Example Output

```
## Coca-Cola → Pepsi
❌ **Rank #2** (expected ≤1) - **FAIL**

## Home Depot → Lowes
✅ **Rank #1** (expected ≤1) - **PASS**

## Apple → Microsoft
❌ **Rank #7** (expected ≤1) - **FAIL**

## Microsoft → Apple
❌ **AAPL not found in top-20 similar companies**
   ℹ️ Relationships exist: SIMILAR_INDUSTRY(SECTOR), SIMILAR_SIZE(COMPOSITE)
```

---

## Next Steps

To improve validation results, focus on **data quality**, not algorithm tuning:

1. **Better Data Sources**: Explore datamule for comprehensive SEC filing data
2. **More Complete Enrichment**: Ensure all companies have industry/sector data
3. **Relationship Coverage**: Verify all expected relationships are created

---

## Experimental Tools (Future Use)

When data quality improves, these tools will be useful:

### `validate-ranking-quality`
Comprehensive validation that generates top-20 rankings for manual review. Useful for understanding overall ranking quality beyond just #1 positions.

**Status**: Experimental - kept for future use when data improves

### `optimize-similarity-weights`
Systematic weight optimization using grid search. Useful for finding optimal weight combinations once relationships are more complete.

**Status**: Experimental - kept for future use when data improves

---

## Related Documentation

- **Famous Pairs Corpus**: `docs/FAMOUS_COMPETITOR_PAIRS.md` - Complete list of 46 pairs
- **Graph Schema**: `docs/graph_schema.md` - Understanding relationship types
- **Company Queries**: `docs/COMPANY_SIMILARITY_QUERIES.md` - Example Cypher queries

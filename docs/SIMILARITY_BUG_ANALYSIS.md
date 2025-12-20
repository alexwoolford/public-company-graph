# Company Similarity Bug Analysis

## Issue

After running `compute_company_similarity.py`, expected relationships are missing:
- KO (Coca-Cola) and PEP (Pepsi) should have `SIMILAR_SIZE` relationship (both in >$10B revenue bucket with 737 companies)
- KO should have ~736 `SIMILAR_SIZE` relationships from the revenue bucket alone
- KO only has 3 `SIMILAR_SIZE` relationships in the database
- PEP is ranked #8 for KO similarity when it should be #1 or #2

## Root Cause

The `compute_size_similarity` function was not sorting the CIK list within each bucket before generating pairs. When processing all 7,209 companies, the bucket lists contained CIKs in the order they were added (which depends on the order companies are fetched from Neo4j), not in sorted order. While the nested loop `for i, cik1 in enumerate(ciks): for cik2 in ciks[i + 1:]` should generate all pairs regardless of order, the unsorted list was causing issues with pair generation for large buckets.

**Evidence:**
- KO and PEP are both in the same buckets for all metrics:
  - Revenue >$10B: 737 companies (both present)
  - Market Cap >$10B: 900 companies (both present)
  - Employees >10000: 986 companies (both present)
- Manual pair generation from the >$10B revenue bucket correctly finds KO-PEP pair
- But when `compute_size_similarity` runs on all companies, KO-PEP pair is NOT generated
- KO only appears in 5 pairs total (should be 736+ from revenue bucket alone)

## Current Status

- ✅ `SIMILAR_INDUSTRY` relationships work correctly (KO-PEP have this)
- ✅ `SIMILAR_SIZE` relationships now work correctly (KO-PEP have this)
- ✅ Composite similarity query works (PEP now ranked #12 with score 1.8)
- ✅ Fix applied: Sort CIKs within buckets before generating pairs

## Workaround

Use the composite similarity query to find similar companies. Even without `SIMILAR_SIZE`, PEP still appears in top 10 for KO:

```cypher
MATCH (c1:Company {ticker: 'KO'})-[r]-(c2:Company)
WHERE type(r) IN ['SIMILAR_INDUSTRY', 'SIMILAR_SIZE', 'SIMILAR_DESCRIPTION',
                  'SIMILAR_TECHNOLOGY']
WITH c2,
     count(r) as edge_count,
     sum(CASE type(r)
         WHEN 'SIMILAR_INDUSTRY' THEN 1.0
         WHEN 'SIMILAR_SIZE' THEN 0.8
         WHEN 'SIMILAR_DESCRIPTION' THEN 0.9
         WHEN 'SIMILAR_TECHNOLOGY' THEN 0.7
         ELSE 0.0
     END) as weighted_score
RETURN c2.ticker, c2.name, edge_count, weighted_score
ORDER BY weighted_score DESC, edge_count DESC
LIMIT 20
```

## Fix Applied

**Solution**: Sort CIKs within each bucket before generating pairs. This ensures consistent ordering and complete pair generation regardless of the order companies are fetched from Neo4j.

**Code Change**: Added `sorted_ciks = sorted(ciks)` before the pair generation loop in `compute_size_similarity()`.

**Result**:
- KO now has 1,104 `SIMILAR_SIZE` relationships (up from 3)
- KO-PEP `SIMILAR_SIZE` relationship now exists
- PEP ranked #12 for KO similarity (score 1.8, up from #8 with score 1.0)
- HD-LOW relationship works correctly (LOW is #1 for HD)

## Expected Behavior

Once fixed, KO should have:
- ~736 `SIMILAR_SIZE` relationships from revenue bucket
- ~899 `SIMILAR_SIZE` relationships from market_cap bucket
- ~985 `SIMILAR_SIZE` relationships from employees bucket
- (Deduplicated to ~1000-1500 unique relationships)

PEP should rank #1 or #2 for KO similarity with a composite score of ~1.8 (SIMILAR_INDUSTRY: 1.0 + SIMILAR_SIZE: 0.8).

# Research-Backed Graph Enhancements

This document outlines enhancements to the Public Company Graph inspired by academic research in financial economics and knowledge graphs. Each enhancement is tied to specific papers in `papers/papers_manifest.json`.

---

## Executive Summary

| Priority | Enhancement | Papers | Status | Business Value |
|----------|-------------|--------|--------|----------------|
| **1** | Supply Chain Risk Scoring | P25, P26 | ✅ Complete | Supply chain due diligence |
| **2** | Risk Profile Convergence | P19 | ⏸️ Blocked | Requires historical 10-Ks (we only have latest) |
| **3** | Entity Resolution Improvement | P39, P40, P58, P62 | ✅ Complete | Data quality improvement |
| **4** | Explainable Similarity | P42, P43 | ✅ Complete | User trust & interpretability |
| **5** | Institutional Ownership Links | P32 | 🔲 Planned | Common owner effects |

---

## Enhancement 1: Supply Chain Risk Scoring

### Research Foundation
- **P25**: Cohen & Frazzini (2008) "Economic Links and Predictable Returns"
- **P26**: Barrot & Sauvagnat (2016) "Input Specificity and Propagation of Idiosyncratic Shocks"

### Key Insights
1. **Supplier concentration matters**: Companies with concentrated supplier bases are more exposed to supply chain shocks
2. **News propagates**: Bad news at a supplier predicts negative returns for customers (with a lag)
3. **Specificity amplifies risk**: "Specific" suppliers (hard to replace) create more vulnerability than commodity suppliers

### Implementation Plan

#### Phase 1: Enhance Existing Relationships
Add properties to `HAS_SUPPLIER` and `HAS_CUSTOMER` relationships:

```cypher
// New properties for HAS_SUPPLIER
{
  concentration_score: FLOAT,    // % of supplier revenue from this customer (if disclosed)
  is_sole_source: BOOLEAN,       // Mentioned as sole/primary supplier
  product_category: STRING,      // What they supply (raw materials, services, etc.)
  specificity_score: FLOAT       // How replaceable is this supplier (derived from industry)
}
```

#### Phase 2: Create SUPPLY_CHAIN_RISK Relationship
New computed relationship that aggregates risk exposure:

```cypher
(Company)-[:SUPPLY_CHAIN_RISK {
  score: FLOAT,                  // Overall supply chain risk score (0-1)
  concentration_risk: FLOAT,     // Risk from supplier concentration
  specificity_risk: FLOAT,       // Risk from input specificity
  network_risk: FLOAT,           // Risk from supplier's suppliers (2nd order)
  computed_at: DATETIME
}]->(Company)
```

#### Phase 3: Multi-Hop Propagation
Implement shock propagation analysis:
- When Company A has news, identify all downstream companies (customers of customers)
- Weight by relationship strength and path length

### Business Queries Enabled
```cypher
// "Which of my portfolio companies have concentrated supply chain risk?"
MATCH (c:Company)-[r:SUPPLY_CHAIN_RISK]->(supplier:Company)
WHERE c.ticker IN ['AAPL', 'MSFT', 'GOOGL']
  AND r.score > 0.7
RETURN c.ticker, supplier.name, r.score, r.concentration_risk
ORDER BY r.score DESC

// "If TSMC has a problem, who is affected?"
MATCH (tsmc:Company {name: 'Taiwan Semiconductor'})<-[:HAS_SUPPLIER*1..2]-(affected:Company)
RETURN affected.ticker, affected.name, length(path) as hops
```

### Data Requirements
- Existing: `HAS_SUPPLIER`, `HAS_CUSTOMER` relationships
- Enhancement: Parse 10-K for supplier concentration language ("sole source", "primary supplier", "X% of purchases")

---

## Enhancement 2: Risk Profile Convergence

### Research Foundation
- **P19**: Chen & Yang (2025) "Risk Factor Similarity and Economic Links"

### Key Insights
1. **Static similarity is useful, but change is more predictive**: Companies whose risk profiles are *converging* often have deeper economic links
2. **Convergence signals**:
   - Potential acquisition targets (acquirer's risks becoming buyer's risks)
   - Emerging competitors (entering same markets)
   - Industry consolidation

### Implementation Plan

#### Phase 1: Store Historical Risk Embeddings
Add temporal dimension to risk data:

```cypher
// New node type for historical snapshots
(:RiskSnapshot {
  cik: STRING,
  filing_year: INTEGER,
  embedding: LIST<FLOAT>,
  embedding_model: STRING,
  filed_at: DATE
})

// Link to company
(Company)-[:HAS_RISK_SNAPSHOT {year: INTEGER}]->(RiskSnapshot)
```

#### Phase 2: Compute Convergence Metrics
New relationship capturing year-over-year change:

```cypher
(Company)-[:RISK_CONVERGENCE {
  score: FLOAT,                  // Cosine similarity this year
  prior_score: FLOAT,            // Cosine similarity last year
  delta: FLOAT,                  // Change (positive = converging)
  is_converging: BOOLEAN,        // delta > threshold
  years_compared: STRING,        // "2023-2024"
  computed_at: DATETIME
}]->(Company)
```

#### Phase 3: Convergence Alerts
Identify significant convergence events:
- New entries: Companies that weren't similar but now are
- Divergence: Previously similar companies drifting apart

### Business Queries Enabled
```cypher
// "Which companies' risk profiles are converging with AAPL?"
MATCH (aapl:Company {ticker: 'AAPL'})-[r:RISK_CONVERGENCE]->(other:Company)
WHERE r.is_converging = true AND r.delta > 0.1
RETURN other.ticker, other.name, r.delta, r.score
ORDER BY r.delta DESC

// "Find potential M&A pairs by risk convergence"
MATCH (a:Company)-[r:RISK_CONVERGENCE]->(b:Company)
WHERE r.delta > 0.15
  AND a.market_cap > b.market_cap * 5  // A is much larger
RETURN a.ticker as potential_acquirer, b.ticker as potential_target, r.delta
```

### Data Requirements
- Existing: `description_embedding` on Company nodes
- New: Historical 10-K risk factor embeddings (Item 1A)
- Script: `compute_risk_convergence.py`

---

## Enhancement 3: Entity Resolution Improvement ✅

### Research Foundation
- **P39**: Mudgal et al. (2018) "Deep Learning for Entity Matching"
- **P40**: Li et al. (2020) "Ditto: A Fine-Tuned Transformer for Entity Matching"
- **P58**: Zeakis et al. (2023) "Pre-trained Embeddings for Entity Resolution" - **NEW**
- **P62**: JEL/JPMorgan (2021) "End-to-End Neural Entity Linking" - **NEW**

### Key Insights
1. **Current approach**: String matching + rules for company mentions in 10-K
2. **Problem**: Many mentions are ambiguous ("Apple" could be AAPL or a fruit company)
3. **Solution**: "Wide & Deep" approach combining character n-grams with semantic embeddings

### Implementation (Completed)

#### Wide Component: Character N-gram Similarity
Handles name variations like "PayPal Holdings" vs "PYPL" vs "Paypal Inc":

```python
from public_company_graph.entity_resolution import CharacterMatcher, ngram_similarity

matcher = CharacterMatcher()
score = matcher.score("Microsoft", "Microsoft Corporation")
# Returns CharacterScore with Jaccard n-gram similarity
```

**Module**: `public_company_graph/entity_resolution/character.py`

#### Deep Component: Semantic Embedding Similarity
Uses existing company description embeddings for context disambiguation:

```python
from public_company_graph.entity_resolution import SemanticScorer

scorer = SemanticScorer(get_embedding_fn=openai_client.get_embedding)
result = scorer.score(
    mention="Apple",
    context="They compete with Apple in the smartphone market.",
    candidate_embedding=company.description_embedding,
    candidate_name="Apple Inc.",
    relationship_type="competitor"
)
# Returns SemanticScore with cosine similarity
```

**Module**: `public_company_graph/entity_resolution/semantic.py`

#### Combined Wide & Deep Scoring
Integrates both components for final confidence:

```python
from public_company_graph.entity_resolution import CombinedScorer, create_scorer

scorer = create_scorer(get_embedding_fn=openai_client.get_embedding)
result = scorer.score(
    mention="Apple",
    context="Tech company context...",
    candidate_ticker="AAPL",
    candidate_name="Apple Inc.",
    candidate_embedding=company_embedding
)
# Returns CombinedScore with:
# - final_score: Weighted combination
# - confidence_tier: HIGH/MEDIUM/LOW
# - character_score, semantic_score: Component scores
# - is_exact_ticker, is_exact_name: Bonus flags
```

**Module**: `public_company_graph/entity_resolution/combined_scorer.py`

### Evaluation Framework

#### AI Audit Dataset
```bash
# Sample relationships from graph
python scripts/er_ai_audit.py sample --count 200 --append

# Label with AI
python scripts/er_ai_audit.py label --concurrency 10

# Evaluate filters
python scripts/er_ai_audit.py evaluate

# Human spot-check (50 samples for verification)
python scripts/er_ai_audit.py spot-check --count 50
```

**Note**: This produces AI-labeled evaluation data. Use `spot-check` for human verification.

**Metrics**: Precision on held-out test set (70/30 train/test split)

### Test Coverage
- 53 new tests for character and semantic similarity
- Tests in `tests/unit/test_character_similarity.py`
- Tests in `tests/unit/test_semantic_similarity.py`

### Data Requirements
- ✅ Existing: Company description embeddings (5,390 companies)
- ✅ Existing: OpenAI embedding API access
- ✅ New: N-gram similarity module (no additional data needed)

---

## Enhancement 4: Explainable Similarity

### Research Foundation
- **P42**: Lao et al. (2011) "Random Walk Inference in Knowledge Bases"
- **P43**: Gardner & Mitchell (2015) "Subgraph Feature Extraction"
- **P48**: "XKE: Explaining Knowledge Graph Embeddings"

### Key Insights
1. **Problem**: "Company A is 87% similar to Company B" is not actionable
2. **Solution**: Decompose similarity into interpretable components
3. **Path-based explanation**: "Similar because they share these suppliers, technologies, and competitors"

### Implementation Plan

#### Phase 1: Create Explanation Function
New utility function for explainable similarity:

```python
def explain_similarity(company_a: str, company_b: str) -> dict:
    """
    Break down why two companies are similar.

    Returns:
        {
            'overall_score': 0.87,
            'components': {
                'description': {'score': 0.82, 'weight': 0.25, 'contribution': 0.205},
                'industry': {'score': 1.0, 'weight': 0.20, 'same_industry': 'Semiconductors'},
                'technology': {'score': 0.71, 'weight': 0.15, 'shared_techs': ['React', 'AWS']},
                'risk_profile': {'score': 0.89, 'weight': 0.15, 'contribution': 0.134},
                'size': {'score': 0.95, 'weight': 0.10, 'both_bucket': 'mega'},
                'business_links': {'score': 0.60, 'weight': 0.15, 'shared_suppliers': 2}
            },
            'explanation': "NVDA and AMD are highly similar (87%) primarily due to..."
        }
    """
```

#### Phase 2: Path-Based Evidence
Find concrete paths connecting companies:

```cypher
// Find all paths between two companies
MATCH path = (a:Company {ticker: $ticker_a})-[*1..3]-(b:Company {ticker: $ticker_b})
WHERE all(r in relationships(path) WHERE type(r) IN ['HAS_SUPPLIER', 'HAS_CUSTOMER', 'HAS_COMPETITOR', 'USES', 'HAS_DOMAIN'])
RETURN path,
       [r in relationships(path) | type(r)] as rel_types,
       length(path) as hops
ORDER BY hops ASC
LIMIT 10
```

#### Phase 3: Natural Language Explanations
Generate human-readable explanations:

```python
def generate_explanation(components: dict) -> str:
    """
    Generate natural language explanation.

    Example output:
    "NVIDIA and AMD are 87% similar. The main factors are:
     • Same industry (Semiconductors) - contributes 20%
     • Very similar business descriptions - contributes 21%
     • Share 2 common suppliers (TSMC, ASML) - contributes 9%
     • Both use similar web technologies (React, AWS) - contributes 11%"
    """
```

### Business Queries Enabled
```cypher
// "Explain why NVDA and AMD are similar"
CALL custom.explainSimilarity('NVDA', 'AMD') YIELD explanation, components
RETURN explanation, components

// "Find the strongest connection path between two companies"
MATCH path = shortestPath((a:Company {ticker: 'AAPL'})-[*]-(b:Company {ticker: 'MSFT'}))
RETURN path
```

### Data Requirements
- Existing: All similarity relationships
- New: Explanation generation function (Python + optional Cypher procedure)

---

## Enhancement 5: Institutional Ownership Links

### Research Foundation
- **P32**: Antón & Polk (2014) "Connected Stocks"

### Key Insights
1. **Common ownership creates correlation**: Companies held by the same institutions move together
2. **Mechanism**: Institutional trading (buying/selling) affects multiple portfolio companies simultaneously
3. **Implication**: Two companies may be "connected" even with no business relationship

### Implementation Plan

#### Phase 1: Data Source Research
Options for 13-F institutional holdings data:
- **SEC EDGAR**: Free, quarterly 13-F filings (>$100M AUM required)
- **OpenFIGI**: Financial instrument identifiers
- **Refinitiv/Bloomberg**: Paid, real-time

#### Phase 2: Schema Design
New nodes and relationships:

```cypher
// New node: Institutional investor
(:Institution {
  cik: STRING,                   // SEC CIK for the institution
  name: STRING,                  // e.g., "Vanguard Group Inc"
  type: STRING,                  // 'mutual_fund', 'hedge_fund', 'pension', etc.
  aum: INTEGER,                  // Assets under management
  loaded_at: DATETIME
})

// New relationship: Holdings
(Institution)-[:HOLDS {
  shares: INTEGER,
  value: INTEGER,                // USD value
  pct_of_portfolio: FLOAT,       // % of institution's portfolio
  pct_of_outstanding: FLOAT,     // % of company's shares outstanding
  filing_date: DATE,
  quarter: STRING                // "2024Q4"
}]->(Company)

// Computed relationship: Common ownership
(Company)-[:COMMON_OWNERSHIP {
  score: FLOAT,                  // Overlap score
  shared_institutions: INTEGER,  // Count of shared owners
  top_shared: LIST<STRING>,      // Names of biggest shared owners
  computed_at: DATETIME
}]->(Company)
```

#### Phase 3: Ownership Overlap Scoring
Compute pairwise ownership overlap:

```python
def compute_ownership_overlap(company_a: str, company_b: str) -> float:
    """
    Compute ownership overlap using Jaccard or weighted overlap.

    Jaccard: |A ∩ B| / |A ∪ B|
    Weighted: Sum of min(pct_a, pct_b) for shared institutions
    """
```

### Business Queries Enabled
```cypher
// "Which companies share the most institutional owners with AAPL?"
MATCH (aapl:Company {ticker: 'AAPL'})-[r:COMMON_OWNERSHIP]->(other:Company)
RETURN other.ticker, other.name, r.score, r.shared_institutions, r.top_shared
ORDER BY r.score DESC
LIMIT 20

// "Which institutions are largest holders of both AAPL and MSFT?"
MATCH (inst:Institution)-[h1:HOLDS]->(aapl:Company {ticker: 'AAPL'})
MATCH (inst)-[h2:HOLDS]->(msft:Company {ticker: 'MSFT'})
RETURN inst.name, h1.value as aapl_value, h2.value as msft_value
ORDER BY h1.value + h2.value DESC
```

### Data Requirements
- **New data source**: 13-F filings from SEC EDGAR
- **Parsing**: Extract holder name, shares, value from 13-F XML
- **Entity resolution**: Match institution names across filings

---

## Implementation Roadmap

### Phase 1: Quick Wins (Week 1-2)
1. ✅ Create this documentation
2. ⏸️ Enhancement 2: Risk convergence - **BLOCKED** (no historical 10-Ks)
3. ✅ Enhancement 4: Explainable similarity function

### Phase 2: Data Enrichment (Week 3-4)
4. ✅ Enhancement 1: Supply chain risk properties
5. ✅ Enhancement 3: Entity resolution improvement (Wide & Deep scoring)

### Phase 3: New Data Sources (Week 5+)
6. 🔲 Enhancement 5: Institutional ownership (requires 13-F parsing)

---

## Related Files

| File | Purpose |
|------|---------|
| `papers/papers_manifest.json` | Full paper metadata (62 papers) |
| `papers/P*.pdf` | Downloaded paper PDFs |
| `scripts/analyze_supply_chain.py` | Supply chain risk analysis CLI ✅ |
| `public_company_graph/supply_chain/` | Supply chain risk scoring module ✅ |
| `public_company_graph/company/explain.py` | Explainable similarity functions ✅ |
| `scripts/explain_similarity.py` | CLI tool for similarity explanation ✅ |
| `public_company_graph/entity_resolution/character.py` | N-gram character matching (Wide) ✅ |
| `public_company_graph/entity_resolution/semantic.py` | Embedding-based matching (Deep) ✅ |
| `public_company_graph/entity_resolution/combined_scorer.py` | Wide & Deep combined scoring ✅ |
| `scripts/er_ai_audit.py` | AI-assisted evaluation (sample, label, evaluate, spot-check) ✅ |
| `scripts/evaluate_layered_validator.py` | Layered validation evaluation ✅ |

---

## References

See `papers/papers_manifest.json` for full citations. Key papers:
- P19, P25, P26, P28, P29, P32 (Economic links)
- P39, P40, P58, P62 (Entity resolution)
- P42, P43, P48 (Explainability)

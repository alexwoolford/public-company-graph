"""
Explainable Similarity: Explain WHY two companies are similar.

This module provides functions to generate human-readable explanations
of company similarity, inspired by research on explainable AI/ML and
knowledge graph reasoning (Papers P42, P43 from research collection).

Key features:
- Feature breakdown showing contribution of each similarity dimension
- Path-based evidence (shared technologies, common business relationships)
- Human-readable summary for business users
- Confidence scoring based on evidence quality

References:
- P42: "Graph Explainability" - Explaining predictions in knowledge graphs
- P43: "Explainability in ML" - Feature attribution for similarity
"""

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class SimilarityDimension(Enum):
    """Types of similarity signals available in the graph."""

    DESCRIPTION = "SIMILAR_DESCRIPTION"
    RISK = "SIMILAR_RISK"
    INDUSTRY = "SIMILAR_INDUSTRY"
    SIZE = "SIMILAR_SIZE"
    TECHNOLOGY = "SIMILAR_TECHNOLOGY"
    COMPETITOR = "HAS_COMPETITOR"  # Direct competitor mention


@dataclass
class SimilarityEvidence:
    """Evidence for a single similarity dimension."""

    dimension: SimilarityDimension
    score: float
    contribution: float  # Weighted contribution to total score
    details: dict[str, Any] = field(default_factory=dict)
    explanation: str = ""


@dataclass
class PathEvidence:
    """Evidence from graph paths between companies."""

    path_type: str  # e.g., "shared_technology", "common_competitor"
    entities: list[str]  # Names of intermediate entities
    count: int
    explanation: str


@dataclass
class SimilarityExplanation:
    """Complete explanation of why two companies are similar."""

    company1_ticker: str
    company1_name: str
    company2_ticker: str
    company2_name: str
    total_score: float
    confidence: str  # "high", "medium", "low"
    feature_breakdown: list[SimilarityEvidence]
    path_evidence: list[PathEvidence]
    summary: str
    top_reasons: list[str]


# Weights for computing contribution percentages
# These match the weights in queries.py
DIMENSION_WEIGHTS = {
    SimilarityDimension.DESCRIPTION: 0.8,
    SimilarityDimension.RISK: 0.8,
    SimilarityDimension.INDUSTRY: 0.6,
    SimilarityDimension.TECHNOLOGY: 0.3,
    SimilarityDimension.SIZE: 0.2,
    SimilarityDimension.COMPETITOR: 4.0,
}


def _get_dimension_explanation(dim: SimilarityDimension, score: float, details: dict) -> str:
    """Generate human-readable explanation for a similarity dimension."""
    if dim == SimilarityDimension.DESCRIPTION:
        pct = int(score * 100)
        return (
            f"Business descriptions are {pct}% similar (based on semantic analysis of 10-K filings)"
        )

    elif dim == SimilarityDimension.RISK:
        pct = int(score * 100)
        return f"Risk factor profiles are {pct}% similar (based on 10-K Item 1A risk disclosures)"

    elif dim == SimilarityDimension.INDUSTRY:
        method = details.get("method", "")
        classification = details.get("classification", "")
        if method == "SIC":
            return f"Same SIC code ({classification}) - strong industry match"
        elif method == "INDUSTRY":
            return f"Same industry classification: {classification}"
        elif method == "SECTOR":
            return f"Same sector: {classification}"
        else:
            return "Same industry classification"

    elif dim == SimilarityDimension.SIZE:
        bucket = details.get("bucket", "")
        metric = details.get("metric", "")
        if bucket and metric:
            return f"Similar company size ({bucket} {metric})"
        return "Similar company size"

    elif dim == SimilarityDimension.TECHNOLOGY:
        pct = int(score * 100)
        return f"Technology stacks are {pct}% similar (based on web technologies)"

    elif dim == SimilarityDimension.COMPETITOR:
        conf = details.get("confidence", 1.0)
        pct = int(conf * 100)
        return f"Direct competitor relationship cited in 10-K filings ({pct}% confidence)"

    return f"{dim.value}: {score:.2f}"


def _get_confidence_level(total_score: float, num_dimensions: int) -> str:
    """Determine confidence level based on score and evidence breadth."""
    if total_score >= 2.0 and num_dimensions >= 3:
        return "high"
    elif total_score >= 1.0 and num_dimensions >= 2:
        return "medium"
    else:
        return "low"


def _generate_summary(
    c1_name: str,
    c2_name: str,
    top_evidence: list[SimilarityEvidence],
    path_evidence: list[PathEvidence],
) -> str:
    """Generate a human-readable summary of similarity."""
    if not top_evidence:
        return f"{c1_name} and {c2_name} have limited similarity signals in the graph."

    # Build list of reasons from evidence
    reasons = []

    for ev in top_evidence[:3]:  # Top 3 reasons
        if ev.dimension == SimilarityDimension.COMPETITOR:
            reasons.append("are direct competitors")
        elif ev.dimension == SimilarityDimension.DESCRIPTION:
            pct = int(ev.score * 100)
            reasons.append(f"have {pct}% similar business descriptions")
        elif ev.dimension == SimilarityDimension.RISK:
            pct = int(ev.score * 100)
            reasons.append(f"face {pct}% similar risk factors")
        elif ev.dimension == SimilarityDimension.INDUSTRY:
            reasons.append("operate in the same industry")
        elif ev.dimension == SimilarityDimension.SIZE:
            reasons.append("are similar in size")
        elif ev.dimension == SimilarityDimension.TECHNOLOGY:
            reasons.append("use similar technologies")

    # Add path evidence to summary
    for pe in path_evidence[:2]:
        if pe.path_type == "shared_technology" and pe.count > 0:
            tech_list = ", ".join(pe.entities[:3])
            if pe.count > 3:
                tech_list += f" and {pe.count - 3} more"
            reasons.append(f"share {pe.count} web technologies ({tech_list})")
        elif pe.path_type == "common_competitor_cites" and pe.count > 0:
            comp_list = ", ".join(pe.entities[:2])
            reasons.append(f"both cited as competitors by {comp_list}")

    if not reasons:
        return f"{c1_name} and {c2_name} show limited similarity."

    # Build natural sentence
    if len(reasons) == 1:
        return f"{c1_name} and {c2_name} {reasons[0]}."
    elif len(reasons) == 2:
        return f"{c1_name} and {c2_name} {reasons[0]} and {reasons[1]}."
    else:
        return f"{c1_name} and {c2_name} {reasons[0]}, {reasons[1]}, and {reasons[2]}."


def explain_similarity(
    driver,
    ticker1: str,
    ticker2: str,
    database: str | None = None,
) -> SimilarityExplanation | None:
    """
    Generate a detailed, human-readable explanation of why two companies are similar.

    This function queries the graph to find all similarity signals between two
    companies and produces:
    - Feature breakdown: Contribution of each similarity dimension
    - Path evidence: Shared entities (technologies, common competitors, etc.)
    - Human-readable summary: Plain English explanation for business users
    - Confidence score: Based on strength and breadth of evidence

    Args:
        driver: Neo4j driver instance
        ticker1: First company ticker symbol (e.g., "AAPL")
        ticker2: Second company ticker symbol (e.g., "MSFT")
        database: Neo4j database name (optional)

    Returns:
        SimilarityExplanation object with full breakdown, or None if companies not found

    Example:
        >>> explanation = explain_similarity(driver, "KO", "PEP")
        >>> print(explanation.summary)
        "COCA-COLA CO and PEPSICO INC are direct competitors, have 87% similar
        business descriptions, and share 5 web technologies (Google Tag Manager,
        HSTS, and 3 more)."

        >>> for reason in explanation.top_reasons:
        ...     print(f"- {reason}")
        - Direct competitor relationship cited in 10-K filings (95% confidence)
        - Business descriptions are 87% similar
        - Risk factor profiles are 79% similar

    Research References:
        - P42: Graph-based explainability for predictions
        - P43: Feature attribution in ML similarity models
    """
    query = """
    // Find both companies
    MATCH (c1:Company {ticker: $ticker1})
    MATCH (c2:Company {ticker: $ticker2})

    // Collect all direct similarity relationships (one direction only to avoid duplicates)
    OPTIONAL MATCH (c1)-[r]->(c2)
    WHERE type(r) IN [
        'SIMILAR_DESCRIPTION', 'SIMILAR_RISK', 'SIMILAR_INDUSTRY',
        'SIMILAR_SIZE', 'SIMILAR_TECHNOLOGY', 'HAS_COMPETITOR'
    ]
    WITH c1, c2, collect({
        rel_type: type(r),
        score: coalesce(r.score, r.confidence, 1.0),
        method: r.method,
        classification: r.classification,
        bucket: r.bucket,
        metric: r.metric,
        confidence: r.confidence
    }) AS relationships

    // Get shared technologies via Domain path
    OPTIONAL MATCH (c1)-[:HAS_DOMAIN]->(:Domain)-[:USES]->(t:Technology)<-[:USES]-(:Domain)<-[:HAS_DOMAIN]-(c2)
    WITH c1, c2, relationships, collect(DISTINCT t.name) AS shared_technologies

    // Get companies that cite BOTH as competitors (common competitive landscape)
    OPTIONAL MATCH (other:Company)-[:HAS_COMPETITOR]->(c1)
    WHERE (other)-[:HAS_COMPETITOR]->(c2)
    WITH c1, c2, relationships, shared_technologies,
         collect(DISTINCT other.ticker) AS common_competitor_cites

    // Get companies that are competitors of BOTH (shared competitive space)
    OPTIONAL MATCH (c1)-[:HAS_COMPETITOR]->(comp:Company)<-[:HAS_COMPETITOR]-(c2)
    WITH c1, c2, relationships, shared_technologies, common_competitor_cites,
         collect(DISTINCT comp.ticker) AS shared_competitors

    // Get shared customers (supply chain overlap)
    OPTIONAL MATCH (c1)-[:HAS_CUSTOMER]->(cust:Company)<-[:HAS_CUSTOMER]-(c2)
    WITH c1, c2, relationships, shared_technologies, common_competitor_cites,
         shared_competitors, collect(DISTINCT cust.ticker) AS shared_customers

    // Get shared suppliers (supply chain overlap)
    OPTIONAL MATCH (c1)-[:HAS_SUPPLIER]->(supp:Company)<-[:HAS_SUPPLIER]-(c2)

    RETURN
        c1.ticker AS ticker1,
        c1.name AS name1,
        c1.sector AS sector1,
        c1.industry AS industry1,
        c2.ticker AS ticker2,
        c2.name AS name2,
        c2.sector AS sector2,
        c2.industry AS industry2,
        relationships,
        shared_technologies,
        common_competitor_cites,
        shared_competitors,
        shared_customers,
        collect(DISTINCT supp.ticker) AS shared_suppliers
    """

    with driver.session(database=database) as session:
        result = session.run(query, ticker1=ticker1, ticker2=ticker2)
        record = result.single()

        if not record:
            logger.warning(f"Companies not found: {ticker1} and/or {ticker2}")
            return None

        # Parse relationships into evidence
        feature_breakdown: list[SimilarityEvidence] = []
        total_score = 0.0

        for rel in record["relationships"]:
            rel_type = rel.get("rel_type")
            if not rel_type:
                continue

            # Map to dimension enum
            dim_map = {
                "SIMILAR_DESCRIPTION": SimilarityDimension.DESCRIPTION,
                "SIMILAR_RISK": SimilarityDimension.RISK,
                "SIMILAR_INDUSTRY": SimilarityDimension.INDUSTRY,
                "SIMILAR_SIZE": SimilarityDimension.SIZE,
                "SIMILAR_TECHNOLOGY": SimilarityDimension.TECHNOLOGY,
                "HAS_COMPETITOR": SimilarityDimension.COMPETITOR,
            }
            dimension = dim_map.get(rel_type)
            if not dimension:
                continue

            score = rel.get("score", 1.0)
            weight = DIMENSION_WEIGHTS.get(dimension, 1.0)
            contribution = score * weight

            details = {
                k: v for k, v in rel.items() if k not in ("rel_type", "score") and v is not None
            }

            explanation = _get_dimension_explanation(dimension, score, details)

            feature_breakdown.append(
                SimilarityEvidence(
                    dimension=dimension,
                    score=score,
                    contribution=contribution,
                    details=details,
                    explanation=explanation,
                )
            )
            total_score += contribution

        # Sort by contribution (highest first)
        feature_breakdown.sort(key=lambda x: x.contribution, reverse=True)

        # Build path evidence
        path_evidence: list[PathEvidence] = []

        # Shared technologies
        shared_techs = record["shared_technologies"] or []
        if shared_techs:
            tech_list = ", ".join(shared_techs[:5])
            if len(shared_techs) > 5:
                tech_list += f" (+{len(shared_techs) - 5} more)"
            path_evidence.append(
                PathEvidence(
                    path_type="shared_technology",
                    entities=shared_techs,
                    count=len(shared_techs),
                    explanation=f"Both companies use: {tech_list}",
                )
            )

        # Companies that cite both as competitors
        common_cites = record["common_competitor_cites"] or []
        if common_cites:
            cite_list = ", ".join(common_cites[:3])
            path_evidence.append(
                PathEvidence(
                    path_type="common_competitor_cites",
                    entities=common_cites,
                    count=len(common_cites),
                    explanation=f"Both cited as competitors by: {cite_list}",
                )
            )

        # Shared competitors
        shared_comps = record["shared_competitors"] or []
        if shared_comps:
            comp_list = ", ".join(shared_comps[:3])
            path_evidence.append(
                PathEvidence(
                    path_type="shared_competitors",
                    entities=shared_comps,
                    count=len(shared_comps),
                    explanation=f"Both compete with: {comp_list}",
                )
            )

        # Shared customers
        shared_custs = record["shared_customers"] or []
        if shared_custs:
            cust_list = ", ".join(shared_custs[:3])
            path_evidence.append(
                PathEvidence(
                    path_type="shared_customers",
                    entities=shared_custs,
                    count=len(shared_custs),
                    explanation=f"Both sell to: {cust_list}",
                )
            )

        # Shared suppliers
        shared_supps = record["shared_suppliers"] or []
        if shared_supps:
            supp_list = ", ".join(shared_supps[:3])
            path_evidence.append(
                PathEvidence(
                    path_type="shared_suppliers",
                    entities=shared_supps,
                    count=len(shared_supps),
                    explanation=f"Both source from: {supp_list}",
                )
            )

        # Generate summary and top reasons
        c1_name = record["name1"] or ticker1
        c2_name = record["name2"] or ticker2

        summary = _generate_summary(c1_name, c2_name, feature_breakdown, path_evidence)

        top_reasons = [ev.explanation for ev in feature_breakdown[:5]]

        # Add path evidence to reasons
        for pe in path_evidence[:3]:
            if pe.count > 0:
                top_reasons.append(pe.explanation)

        confidence = _get_confidence_level(total_score, len(feature_breakdown))

        return SimilarityExplanation(
            company1_ticker=ticker1,
            company1_name=c1_name,
            company2_ticker=ticker2,
            company2_name=c2_name,
            total_score=total_score,
            confidence=confidence,
            feature_breakdown=feature_breakdown,
            path_evidence=path_evidence,
            summary=summary,
            top_reasons=top_reasons,
        )


def explain_similarity_to_dict(
    driver,
    ticker1: str,
    ticker2: str,
    database: str | None = None,
) -> dict | None:
    """
    Generate similarity explanation as a dictionary (JSON-serializable).

    Convenience wrapper around explain_similarity() for API responses.

    Args:
        driver: Neo4j driver instance
        ticker1: First company ticker
        ticker2: Second company ticker
        database: Neo4j database name

    Returns:
        Dictionary with explanation data, or None if companies not found
    """
    explanation = explain_similarity(driver, ticker1, ticker2, database)
    if not explanation:
        return None

    return {
        "company1": {
            "ticker": explanation.company1_ticker,
            "name": explanation.company1_name,
        },
        "company2": {
            "ticker": explanation.company2_ticker,
            "name": explanation.company2_name,
        },
        "total_score": round(explanation.total_score, 3),
        "confidence": explanation.confidence,
        "summary": explanation.summary,
        "top_reasons": explanation.top_reasons,
        "feature_breakdown": [
            {
                "dimension": ev.dimension.value,
                "score": round(ev.score, 3),
                "contribution": round(ev.contribution, 3),
                "explanation": ev.explanation,
                "details": ev.details,
            }
            for ev in explanation.feature_breakdown
        ],
        "path_evidence": [
            {
                "type": pe.path_type,
                "count": pe.count,
                "entities": pe.entities[:10],  # Limit for API response
                "explanation": pe.explanation,
            }
            for pe in explanation.path_evidence
        ],
    }


def format_explanation_text(
    driver,
    ticker1: str,
    ticker2: str,
    database: str | None = None,
) -> str:
    """
    Generate a nicely formatted text explanation for terminal/report output.

    Args:
        driver: Neo4j driver instance
        ticker1: First company ticker
        ticker2: Second company ticker
        database: Neo4j database name

    Returns:
        Formatted multi-line string explanation
    """
    explanation = explain_similarity(driver, ticker1, ticker2, database)
    if not explanation:
        return f"Could not find similarity data for {ticker1} and {ticker2}"

    lines = []
    lines.append("=" * 70)
    lines.append(f"SIMILARITY EXPLANATION: {ticker1} vs {ticker2}")
    lines.append("=" * 70)
    lines.append("")
    lines.append(f"Companies: {explanation.company1_name} ↔ {explanation.company2_name}")
    lines.append(
        f"Total Score: {explanation.total_score:.3f} (Confidence: {explanation.confidence})"
    )
    lines.append("")
    lines.append("SUMMARY")
    lines.append("-" * 40)
    lines.append(explanation.summary)
    lines.append("")

    if explanation.feature_breakdown:
        lines.append("FEATURE BREAKDOWN")
        lines.append("-" * 40)
        for ev in explanation.feature_breakdown:
            bar_len = int(ev.contribution * 10)
            bar = "█" * bar_len + "░" * (10 - bar_len)
            lines.append(f"  [{bar}] {ev.dimension.name}: {ev.score:.2f}")
            lines.append(f"           └─ {ev.explanation}")
        lines.append("")

    if explanation.path_evidence:
        lines.append("PATH EVIDENCE")
        lines.append("-" * 40)
        for pe in explanation.path_evidence:
            if pe.count > 0:
                lines.append(f"  • {pe.explanation}")
        lines.append("")

    lines.append("TOP REASONS")
    lines.append("-" * 40)
    for i, reason in enumerate(explanation.top_reasons[:5], 1):
        lines.append(f"  {i}. {reason}")

    lines.append("")
    lines.append("=" * 70)

    return "\n".join(lines)

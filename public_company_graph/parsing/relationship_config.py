"""
Configuration for tiered relationship storage.

Based on precision analysis:
- HAS_COMPETITOR: High precision (~90%+) at threshold 0.40
- HAS_PARTNER: Moderate precision (~75%) at threshold 0.40
- HAS_SUPPLIER/CUSTOMER: Low precision - store as candidates only

Tiered approach:
1. HIGH CONFIDENCE: Store as facts (HAS_COMPETITOR, HAS_SUPPLIER, etc.)
2. MEDIUM CONFIDENCE: Store as candidates (CANDIDATE_COMPETITOR, etc.)
3. LOW CONFIDENCE: Don't create edges
"""

from dataclasses import dataclass
from enum import Enum


class ConfidenceTier(Enum):
    """Confidence tiers for relationship extraction."""

    HIGH = "high"  # Store as fact
    MEDIUM = "medium"  # Store as candidate
    LOW = "low"  # Don't store


@dataclass
class RelationshipTypeConfig:
    """Configuration for a specific relationship type."""

    # Relationship type string (e.g., "HAS_COMPETITOR")
    relationship_type: str

    # Neo4j relationship type for high-confidence facts
    fact_type: str

    # Neo4j relationship type for medium-confidence candidates
    candidate_type: str

    # Embedding similarity threshold for HIGH confidence (fact)
    high_threshold: float

    # Embedding similarity threshold for MEDIUM confidence (candidate)
    medium_threshold: float

    # Whether to extract this relationship type at all
    enabled: bool = True

    # Whether high-confidence edges are reliable enough for analytics
    analytics_ready: bool = False


# Default configuration based on precision analysis
RELATIONSHIP_CONFIGS = {
    "HAS_COMPETITOR": RelationshipTypeConfig(
        relationship_type="HAS_COMPETITOR",
        fact_type="HAS_COMPETITOR",
        candidate_type="CANDIDATE_COMPETITOR",
        high_threshold=0.40,  # ~92% precision
        medium_threshold=0.25,  # ~85% precision
        enabled=True,
        analytics_ready=True,  # Can be used for PageRank, similarity, etc.
    ),
    "HAS_PARTNER": RelationshipTypeConfig(
        relationship_type="HAS_PARTNER",
        fact_type="HAS_PARTNER",
        candidate_type="CANDIDATE_PARTNER",
        high_threshold=0.50,  # ~79% precision
        medium_threshold=0.30,  # ~66% precision
        enabled=True,
        analytics_ready=False,  # Not reliable enough for analytics
    ),
    "HAS_SUPPLIER": RelationshipTypeConfig(
        relationship_type="HAS_SUPPLIER",
        fact_type="HAS_SUPPLIER",
        candidate_type="CANDIDATE_SUPPLIER",
        high_threshold=0.55,  # ~80% precision but very low recall
        medium_threshold=0.30,  # ~42% precision
        enabled=True,
        analytics_ready=False,  # Too noisy for analytics
    ),
    "HAS_CUSTOMER": RelationshipTypeConfig(
        relationship_type="HAS_CUSTOMER",
        fact_type="HAS_CUSTOMER",
        candidate_type="CANDIDATE_CUSTOMER",
        high_threshold=0.55,  # ~100% precision but very low recall
        medium_threshold=0.30,  # ~38% precision
        enabled=True,
        analytics_ready=False,  # Too noisy for analytics
    ),
}


def get_confidence_tier(
    relationship_type: str,
    embedding_similarity: float | None,
) -> ConfidenceTier:
    """
    Determine confidence tier based on embedding similarity.

    Args:
        relationship_type: The relationship type (e.g., "HAS_COMPETITOR")
        embedding_similarity: The embedding similarity score (0-1)

    Returns:
        ConfidenceTier indicating how to store this relationship
    """
    config = RELATIONSHIP_CONFIGS.get(relationship_type)
    if not config:
        return ConfidenceTier.LOW

    if embedding_similarity is None:
        # No embedding available - default to medium
        return ConfidenceTier.MEDIUM

    if embedding_similarity >= config.high_threshold:
        return ConfidenceTier.HIGH
    elif embedding_similarity >= config.medium_threshold:
        return ConfidenceTier.MEDIUM
    else:
        return ConfidenceTier.LOW


def get_neo4j_relationship_type(
    relationship_type: str,
    tier: ConfidenceTier,
) -> str | None:
    """
    Get the Neo4j relationship type for a given tier.

    Args:
        relationship_type: The relationship type (e.g., "HAS_COMPETITOR")
        tier: The confidence tier

    Returns:
        Neo4j relationship type string, or None if edge shouldn't be created
    """
    config = RELATIONSHIP_CONFIGS.get(relationship_type)
    if not config or not config.enabled:
        return None

    if tier == ConfidenceTier.HIGH:
        return config.fact_type
    elif tier == ConfidenceTier.MEDIUM:
        return config.candidate_type
    else:
        return None  # Don't create low-confidence edges

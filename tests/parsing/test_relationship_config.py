"""Tests for relationship configuration and tiered storage."""

from public_company_graph.parsing.relationship_config import (
    RELATIONSHIP_CONFIGS,
    ConfidenceTier,
    get_confidence_tier,
    get_neo4j_relationship_type,
)


class TestRelationshipConfig:
    """Tests for relationship configuration."""

    def test_all_relationship_types_have_config(self):
        """All standard relationship types should have configuration."""
        expected_types = ["HAS_COMPETITOR", "HAS_SUPPLIER", "HAS_CUSTOMER", "HAS_PARTNER"]
        for rel_type in expected_types:
            assert rel_type in RELATIONSHIP_CONFIGS

    def test_competitor_config_values(self):
        """Competitor config should have appropriate thresholds."""
        config = RELATIONSHIP_CONFIGS["HAS_COMPETITOR"]
        assert config.fact_type == "HAS_COMPETITOR"
        assert config.candidate_type == "CANDIDATE_COMPETITOR"
        assert config.high_threshold == 0.35  # Lowered from 0.40 for more recall
        assert config.medium_threshold == 0.25
        assert config.enabled is True
        assert config.analytics_ready is True

    def test_supplier_config_not_analytics_ready(self):
        """Supplier config should NOT be analytics ready (low precision)."""
        config = RELATIONSHIP_CONFIGS["HAS_SUPPLIER"]
        assert config.analytics_ready is False


class TestGetConfidenceTier:
    """Tests for confidence tier determination."""

    def test_high_confidence_competitor(self):
        """High embedding similarity should return HIGH tier for competitor."""
        tier = get_confidence_tier("HAS_COMPETITOR", 0.50)
        assert tier == ConfidenceTier.HIGH

    def test_medium_confidence_competitor(self):
        """Medium embedding similarity should return MEDIUM tier for competitor."""
        tier = get_confidence_tier("HAS_COMPETITOR", 0.30)
        assert tier == ConfidenceTier.MEDIUM

    def test_low_confidence_competitor(self):
        """Low embedding similarity should return LOW tier for competitor."""
        tier = get_confidence_tier("HAS_COMPETITOR", 0.20)
        assert tier == ConfidenceTier.LOW

    def test_none_embedding_returns_medium(self):
        """None embedding similarity should default to MEDIUM tier."""
        tier = get_confidence_tier("HAS_COMPETITOR", None)
        assert tier == ConfidenceTier.MEDIUM

    def test_unknown_relationship_type_returns_low(self):
        """Unknown relationship type should return LOW tier."""
        tier = get_confidence_tier("UNKNOWN_TYPE", 0.90)
        assert tier == ConfidenceTier.LOW


class TestGetNeo4jRelationshipType:
    """Tests for Neo4j relationship type mapping."""

    def test_high_tier_returns_fact_type(self):
        """HIGH tier should return fact type (HAS_COMPETITOR)."""
        neo4j_type = get_neo4j_relationship_type("HAS_COMPETITOR", ConfidenceTier.HIGH)
        assert neo4j_type == "HAS_COMPETITOR"

    def test_medium_tier_returns_candidate_type(self):
        """MEDIUM tier should return candidate type (CANDIDATE_COMPETITOR)."""
        neo4j_type = get_neo4j_relationship_type("HAS_COMPETITOR", ConfidenceTier.MEDIUM)
        assert neo4j_type == "CANDIDATE_COMPETITOR"

    def test_low_tier_returns_none(self):
        """LOW tier should return None (don't create edge)."""
        neo4j_type = get_neo4j_relationship_type("HAS_COMPETITOR", ConfidenceTier.LOW)
        assert neo4j_type is None

    def test_unknown_relationship_type_returns_none(self):
        """Unknown relationship type should return None."""
        neo4j_type = get_neo4j_relationship_type("UNKNOWN_TYPE", ConfidenceTier.HIGH)
        assert neo4j_type is None

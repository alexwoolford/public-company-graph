"""
Unit tests for the explainable similarity module.

Tests the core explanation functions without requiring a live Neo4j database.
"""

from unittest.mock import MagicMock

from public_company_graph.company.explain import (
    DIMENSION_WEIGHTS,
    PathEvidence,
    SimilarityDimension,
    SimilarityEvidence,
    SimilarityExplanation,
    _generate_summary,
    _get_confidence_level,
    _get_dimension_explanation,
    explain_similarity,
    explain_similarity_to_dict,
    format_explanation_text,
)


class TestSimilarityDimension:
    """Tests for the SimilarityDimension enum."""

    def test_all_dimensions_have_weights(self):
        """Every dimension should have a weight defined."""
        for dim in SimilarityDimension:
            assert dim in DIMENSION_WEIGHTS, f"Missing weight for {dim.name}"
            assert DIMENSION_WEIGHTS[dim] > 0, f"Weight for {dim.name} should be positive"

    def test_competitor_has_highest_weight(self):
        """HAS_COMPETITOR should be the strongest signal."""
        competitor_weight = DIMENSION_WEIGHTS[SimilarityDimension.COMPETITOR]
        other_weights = [
            w for d, w in DIMENSION_WEIGHTS.items() if d != SimilarityDimension.COMPETITOR
        ]
        assert competitor_weight > max(other_weights), "COMPETITOR should have highest weight"


class TestDimensionExplanation:
    """Tests for _get_dimension_explanation function."""

    def test_description_explanation_format(self):
        """SIMILAR_DESCRIPTION should produce percentage-based explanation."""
        explanation = _get_dimension_explanation(SimilarityDimension.DESCRIPTION, 0.87, {})
        assert "87%" in explanation
        assert "similar" in explanation.lower()
        assert "10-K" in explanation

    def test_risk_explanation_format(self):
        """SIMILAR_RISK should mention risk factors."""
        explanation = _get_dimension_explanation(SimilarityDimension.RISK, 0.91, {})
        assert "91%" in explanation
        assert "risk" in explanation.lower()

    def test_industry_sic_explanation(self):
        """SIMILAR_INDUSTRY with SIC should show the code."""
        explanation = _get_dimension_explanation(
            SimilarityDimension.INDUSTRY, 1.0, {"method": "SIC", "classification": "3674"}
        )
        assert "3674" in explanation
        assert "SIC" in explanation

    def test_industry_sector_explanation(self):
        """SIMILAR_INDUSTRY with sector should show sector name."""
        explanation = _get_dimension_explanation(
            SimilarityDimension.INDUSTRY, 1.0, {"method": "SECTOR", "classification": "Technology"}
        )
        assert "Technology" in explanation

    def test_size_explanation(self):
        """SIMILAR_SIZE should mention the bucket and metric."""
        explanation = _get_dimension_explanation(
            SimilarityDimension.SIZE, 1.0, {"bucket": ">$10B", "metric": "revenue"}
        )
        assert ">$10B" in explanation or "size" in explanation.lower()

    def test_technology_explanation(self):
        """SIMILAR_TECHNOLOGY should produce percentage."""
        explanation = _get_dimension_explanation(SimilarityDimension.TECHNOLOGY, 0.65, {})
        assert "65%" in explanation

    def test_competitor_explanation(self):
        """HAS_COMPETITOR should mention 10-K filings."""
        explanation = _get_dimension_explanation(
            SimilarityDimension.COMPETITOR, 1.0, {"confidence": 0.95}
        )
        assert "competitor" in explanation.lower()
        assert "10-K" in explanation or "filing" in explanation.lower()


class TestConfidenceLevel:
    """Tests for _get_confidence_level function."""

    def test_high_confidence(self):
        """High score with multiple dimensions should be high confidence."""
        confidence = _get_confidence_level(total_score=2.5, num_dimensions=4)
        assert confidence == "high"

    def test_medium_confidence(self):
        """Moderate score with some dimensions should be medium confidence."""
        confidence = _get_confidence_level(total_score=1.5, num_dimensions=2)
        assert confidence == "medium"

    def test_low_confidence(self):
        """Low score or few dimensions should be low confidence."""
        confidence = _get_confidence_level(total_score=0.5, num_dimensions=1)
        assert confidence == "low"

    def test_edge_case_high_score_few_dimensions(self):
        """High score but few dimensions should be low (need breadth of evidence)."""
        confidence = _get_confidence_level(total_score=2.5, num_dimensions=1)
        # Even high score needs multiple dimensions for high confidence
        assert confidence == "low"


class TestGenerateSummary:
    """Tests for _generate_summary function."""

    def test_empty_evidence_produces_message(self):
        """No evidence should produce a meaningful message."""
        summary = _generate_summary("Apple", "Microsoft", [], [])
        assert "Apple" in summary
        assert "Microsoft" in summary
        assert "limited" in summary.lower()

    def test_single_reason_summary(self):
        """Single reason should produce grammatical sentence."""
        evidence = [
            SimilarityEvidence(
                dimension=SimilarityDimension.COMPETITOR,
                score=1.0,
                contribution=4.0,
                details={},
                explanation="Direct competitor",
            )
        ]
        summary = _generate_summary("NVIDIA", "AMD", evidence, [])
        assert "NVIDIA" in summary
        assert "AMD" in summary
        assert "competitor" in summary.lower()

    def test_multiple_reasons_joined_correctly(self):
        """Multiple reasons should be joined with commas and 'and'."""
        evidence = [
            SimilarityEvidence(
                dimension=SimilarityDimension.DESCRIPTION,
                score=0.87,
                contribution=0.7,
                details={},
                explanation="87% similar descriptions",
            ),
            SimilarityEvidence(
                dimension=SimilarityDimension.RISK,
                score=0.85,
                contribution=0.68,
                details={},
                explanation="85% similar risk",
            ),
        ]
        summary = _generate_summary("KO", "PEP", evidence, [])
        assert "KO" in summary
        assert "PEP" in summary
        # Should have "and" joining reasons
        assert " and " in summary

    def test_path_evidence_included_in_summary(self):
        """Shared technologies should appear in summary."""
        evidence = [
            SimilarityEvidence(
                dimension=SimilarityDimension.DESCRIPTION,
                score=0.80,
                contribution=0.64,
                details={},
                explanation="80% similar",
            ),
        ]
        path_evidence = [
            PathEvidence(
                path_type="shared_technology",
                entities=["React", "AWS", "Google Analytics"],
                count=3,
                explanation="Both use: React, AWS, Google Analytics",
            )
        ]
        summary = _generate_summary("A", "B", evidence, path_evidence)
        assert "technolog" in summary.lower()


class TestSimilarityEvidenceDataclass:
    """Tests for the SimilarityEvidence dataclass."""

    def test_evidence_creation(self):
        """Evidence dataclass should hold all required fields."""
        evidence = SimilarityEvidence(
            dimension=SimilarityDimension.DESCRIPTION,
            score=0.87,
            contribution=0.696,
            details={"metric": "COSINE"},
            explanation="87% similar business descriptions",
        )
        assert evidence.dimension == SimilarityDimension.DESCRIPTION
        assert evidence.score == 0.87
        assert evidence.contribution == 0.696
        assert "COSINE" in evidence.details.values()

    def test_evidence_default_details(self):
        """Details should default to empty dict."""
        evidence = SimilarityEvidence(
            dimension=SimilarityDimension.RISK,
            score=0.9,
            contribution=0.72,
            explanation="Test",
        )
        assert evidence.details == {}


class TestExplainSimilarity:
    """Tests for the main explain_similarity function."""

    def _mock_neo4j_record(self, data: dict):
        """Create a mock Neo4j record."""
        record = MagicMock()
        record.__getitem__ = lambda self, key: data.get(key)
        return record

    def test_returns_none_when_companies_not_found(self):
        """Should return None when query returns no results."""
        mock_driver = MagicMock()
        mock_session = MagicMock()
        mock_driver.session.return_value.__enter__ = MagicMock(return_value=mock_session)
        mock_driver.session.return_value.__exit__ = MagicMock(return_value=False)
        mock_session.run.return_value.single.return_value = None

        result = explain_similarity(mock_driver, "FAKE1", "FAKE2")
        assert result is None

    def test_returns_explanation_with_all_fields(self):
        """Should return a complete SimilarityExplanation."""
        mock_driver = MagicMock()
        mock_session = MagicMock()
        mock_driver.session.return_value.__enter__ = MagicMock(return_value=mock_session)
        mock_driver.session.return_value.__exit__ = MagicMock(return_value=False)

        # Mock Neo4j response
        mock_record = self._mock_neo4j_record(
            {
                "ticker1": "KO",
                "name1": "COCA COLA CO",
                "sector1": "Consumer Defensive",
                "industry1": "Beverages",
                "ticker2": "PEP",
                "name2": "PEPSICO INC",
                "sector2": "Consumer Defensive",
                "industry2": "Beverages",
                "relationships": [
                    {"rel_type": "SIMILAR_DESCRIPTION", "score": 0.87, "metric": "COSINE"},
                    {"rel_type": "SIMILAR_RISK", "score": 0.85, "metric": "COSINE"},
                ],
                "shared_technologies": ["HSTS", "Google Tag Manager"],
                "common_competitor_cites": [],
                "shared_competitors": ["KDP"],
                "shared_customers": [],
                "shared_suppliers": [],
            }
        )
        mock_session.run.return_value.single.return_value = mock_record

        result = explain_similarity(mock_driver, "KO", "PEP")

        assert result is not None
        assert isinstance(result, SimilarityExplanation)
        assert result.company1_ticker == "KO"
        assert result.company2_ticker == "PEP"
        assert result.company1_name == "COCA COLA CO"
        assert result.company2_name == "PEPSICO INC"
        assert result.total_score > 0
        assert result.confidence in ["high", "medium", "low"]
        assert len(result.feature_breakdown) == 2  # DESCRIPTION and RISK
        assert len(result.path_evidence) >= 1  # At least shared technologies
        assert result.summary  # Should have a summary
        assert len(result.top_reasons) > 0

    def test_feature_breakdown_sorted_by_contribution(self):
        """Feature breakdown should be sorted by contribution (highest first)."""
        mock_driver = MagicMock()
        mock_session = MagicMock()
        mock_driver.session.return_value.__enter__ = MagicMock(return_value=mock_session)
        mock_driver.session.return_value.__exit__ = MagicMock(return_value=False)

        mock_record = self._mock_neo4j_record(
            {
                "ticker1": "NVDA",
                "name1": "NVIDIA CORP",
                "sector1": "Technology",
                "industry1": "Semiconductors",
                "ticker2": "AMD",
                "name2": "ADVANCED MICRO DEVICES",
                "sector2": "Technology",
                "industry2": "Semiconductors",
                "relationships": [
                    {"rel_type": "SIMILAR_DESCRIPTION", "score": 0.80, "metric": "COSINE"},
                    {"rel_type": "HAS_COMPETITOR", "score": 1.0, "confidence": 1.0},
                    {"rel_type": "SIMILAR_RISK", "score": 0.91, "metric": "COSINE"},
                ],
                "shared_technologies": [],
                "common_competitor_cites": [],
                "shared_competitors": [],
                "shared_customers": [],
                "shared_suppliers": [],
            }
        )
        mock_session.run.return_value.single.return_value = mock_record

        result = explain_similarity(mock_driver, "NVDA", "AMD")

        # COMPETITOR (weight 4.0) should be first, then RISK/DESCRIPTION
        assert result.feature_breakdown[0].dimension == SimilarityDimension.COMPETITOR
        contributions = [ev.contribution for ev in result.feature_breakdown]
        assert contributions == sorted(contributions, reverse=True)

    def test_path_evidence_captures_shared_entities(self):
        """Path evidence should include all types of shared entities."""
        mock_driver = MagicMock()
        mock_session = MagicMock()
        mock_driver.session.return_value.__enter__ = MagicMock(return_value=mock_session)
        mock_driver.session.return_value.__exit__ = MagicMock(return_value=False)

        mock_record = self._mock_neo4j_record(
            {
                "ticker1": "A",
                "name1": "Company A",
                "sector1": None,
                "industry1": None,
                "ticker2": "B",
                "name2": "Company B",
                "sector2": None,
                "industry2": None,
                "relationships": [],
                "shared_technologies": ["React", "AWS"],
                "common_competitor_cites": ["X", "Y"],
                "shared_competitors": ["Z"],
                "shared_customers": ["C1"],
                "shared_suppliers": ["S1", "S2"],
            }
        )
        mock_session.run.return_value.single.return_value = mock_record

        result = explain_similarity(mock_driver, "A", "B")

        path_types = [pe.path_type for pe in result.path_evidence]
        assert "shared_technology" in path_types
        assert "common_competitor_cites" in path_types
        assert "shared_competitors" in path_types
        assert "shared_customers" in path_types
        assert "shared_suppliers" in path_types


class TestExplainSimilarityToDict:
    """Tests for explain_similarity_to_dict function."""

    def _mock_neo4j_record(self, data: dict):
        """Create a mock Neo4j record."""
        record = MagicMock()
        record.__getitem__ = lambda self, key: data.get(key)
        return record

    def test_returns_none_when_no_data(self):
        """Should return None when companies not found."""
        mock_driver = MagicMock()
        mock_session = MagicMock()
        mock_driver.session.return_value.__enter__ = MagicMock(return_value=mock_session)
        mock_driver.session.return_value.__exit__ = MagicMock(return_value=False)
        mock_session.run.return_value.single.return_value = None

        result = explain_similarity_to_dict(mock_driver, "FAKE1", "FAKE2")
        assert result is None

    def test_returns_json_serializable_dict(self):
        """Output should be JSON-serializable."""
        import json

        mock_driver = MagicMock()
        mock_session = MagicMock()
        mock_driver.session.return_value.__enter__ = MagicMock(return_value=mock_session)
        mock_driver.session.return_value.__exit__ = MagicMock(return_value=False)

        mock_record = self._mock_neo4j_record(
            {
                "ticker1": "A",
                "name1": "Company A",
                "sector1": None,
                "industry1": None,
                "ticker2": "B",
                "name2": "Company B",
                "sector2": None,
                "industry2": None,
                "relationships": [
                    {"rel_type": "SIMILAR_DESCRIPTION", "score": 0.8, "metric": "COSINE"}
                ],
                "shared_technologies": ["Tech1"],
                "common_competitor_cites": [],
                "shared_competitors": [],
                "shared_customers": [],
                "shared_suppliers": [],
            }
        )
        mock_session.run.return_value.single.return_value = mock_record

        result = explain_similarity_to_dict(mock_driver, "A", "B")

        # Should not raise
        json_str = json.dumps(result)
        assert json_str  # Non-empty JSON

    def test_dict_has_expected_keys(self):
        """Output dict should have all expected top-level keys."""
        mock_driver = MagicMock()
        mock_session = MagicMock()
        mock_driver.session.return_value.__enter__ = MagicMock(return_value=mock_session)
        mock_driver.session.return_value.__exit__ = MagicMock(return_value=False)

        mock_record = self._mock_neo4j_record(
            {
                "ticker1": "A",
                "name1": "Company A",
                "sector1": None,
                "industry1": None,
                "ticker2": "B",
                "name2": "Company B",
                "sector2": None,
                "industry2": None,
                "relationships": [],
                "shared_technologies": [],
                "common_competitor_cites": [],
                "shared_competitors": [],
                "shared_customers": [],
                "shared_suppliers": [],
            }
        )
        mock_session.run.return_value.single.return_value = mock_record

        result = explain_similarity_to_dict(mock_driver, "A", "B")

        expected_keys = {
            "company1",
            "company2",
            "total_score",
            "confidence",
            "summary",
            "top_reasons",
            "feature_breakdown",
            "path_evidence",
        }
        assert set(result.keys()) == expected_keys


class TestFormatExplanationText:
    """Tests for format_explanation_text function."""

    def _mock_neo4j_record(self, data: dict):
        """Create a mock Neo4j record."""
        record = MagicMock()
        record.__getitem__ = lambda self, key: data.get(key)
        return record

    def test_returns_error_message_when_no_data(self):
        """Should return error message when companies not found."""
        mock_driver = MagicMock()
        mock_session = MagicMock()
        mock_driver.session.return_value.__enter__ = MagicMock(return_value=mock_session)
        mock_driver.session.return_value.__exit__ = MagicMock(return_value=False)
        mock_session.run.return_value.single.return_value = None

        result = format_explanation_text(mock_driver, "FAKE1", "FAKE2")
        assert "Could not find" in result

    def test_formatted_output_has_sections(self):
        """Formatted output should have all expected sections."""
        mock_driver = MagicMock()
        mock_session = MagicMock()
        mock_driver.session.return_value.__enter__ = MagicMock(return_value=mock_session)
        mock_driver.session.return_value.__exit__ = MagicMock(return_value=False)

        mock_record = self._mock_neo4j_record(
            {
                "ticker1": "KO",
                "name1": "COCA COLA CO",
                "sector1": None,
                "industry1": None,
                "ticker2": "PEP",
                "name2": "PEPSICO INC",
                "sector2": None,
                "industry2": None,
                "relationships": [
                    {"rel_type": "SIMILAR_DESCRIPTION", "score": 0.87, "metric": "COSINE"}
                ],
                "shared_technologies": ["HSTS"],
                "common_competitor_cites": [],
                "shared_competitors": [],
                "shared_customers": [],
                "shared_suppliers": [],
            }
        )
        mock_session.run.return_value.single.return_value = mock_record

        result = format_explanation_text(mock_driver, "KO", "PEP")

        # Check for expected sections
        assert "SIMILARITY EXPLANATION" in result
        assert "KO" in result and "PEP" in result
        assert "SUMMARY" in result
        assert "FEATURE BREAKDOWN" in result
        assert "TOP REASONS" in result

    def test_formatted_output_has_visual_bars(self):
        """Formatted output should include visual contribution bars."""
        mock_driver = MagicMock()
        mock_session = MagicMock()
        mock_driver.session.return_value.__enter__ = MagicMock(return_value=mock_session)
        mock_driver.session.return_value.__exit__ = MagicMock(return_value=False)

        mock_record = self._mock_neo4j_record(
            {
                "ticker1": "A",
                "name1": "Company A",
                "sector1": None,
                "industry1": None,
                "ticker2": "B",
                "name2": "Company B",
                "sector2": None,
                "industry2": None,
                "relationships": [
                    {"rel_type": "SIMILAR_DESCRIPTION", "score": 0.5, "metric": "COSINE"}
                ],
                "shared_technologies": [],
                "common_competitor_cites": [],
                "shared_competitors": [],
                "shared_customers": [],
                "shared_suppliers": [],
            }
        )
        mock_session.run.return_value.single.return_value = mock_record

        result = format_explanation_text(mock_driver, "A", "B")

        # Should have visual bar characters
        assert "█" in result or "░" in result

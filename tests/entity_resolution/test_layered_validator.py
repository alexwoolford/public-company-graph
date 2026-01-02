"""Tests for LayeredEntityValidator."""

from public_company_graph.entity_resolution.layered_validator import (
    LayeredEntityValidator,
    RejectionReason,
    ValidationResult,
    validate_entity,
)


class TestLayeredEntityValidator:
    """Tests for LayeredEntityValidator."""

    def test_skip_embedding_mode(self):
        """Validator should work without embeddings when skip_embedding=True."""
        validator = LayeredEntityValidator(skip_embedding=True)

        result = validator.validate(
            context="We compete with Microsoft in cloud services.",
            mention="Microsoft",
            ticker="MSFT",
            company_name="Microsoft Corporation",
            relationship_type="HAS_COMPETITOR",
        )

        assert isinstance(result, ValidationResult)
        assert result.embedding_similarity is None
        # When skipping embedding, embedding_passed defaults to True (not checked)
        assert result.embedding_passed is True

    def test_biographical_filter_rejects(self):
        """Should reject biographical context."""
        validator = LayeredEntityValidator(skip_embedding=True)

        result = validator.validate(
            context="Mr. Smith serves as a director of Microsoft Corporation.",
            mention="Microsoft",
            ticker="MSFT",
            company_name="Microsoft Corporation",
            relationship_type="HAS_PARTNER",
        )

        assert result.accepted is False
        assert result.rejection_reason == RejectionReason.BIOGRAPHICAL_CONTEXT
        assert result.biographical_passed is False

    def test_exchange_filter_rejects(self):
        """Should reject exchange listings."""
        validator = LayeredEntityValidator(skip_embedding=True)

        result = validator.validate(
            context="Our common stock is listed on the NASDAQ under the symbol TEST.",
            mention="NASDAQ",
            ticker="NDAQ",
            company_name="Nasdaq Inc.",
            relationship_type="HAS_PARTNER",
        )

        assert result.accepted is False
        assert result.rejection_reason == RejectionReason.EXCHANGE_REFERENCE

    def test_relationship_verifier_catches_mismatch(self):
        """Should catch relationship type mismatches."""
        validator = LayeredEntityValidator(skip_embedding=True)

        result = validator.validate(
            context="Our principal competitors include Google, Microsoft, and Amazon.",
            mention="Microsoft",
            ticker="MSFT",
            company_name="Microsoft Corporation",
            relationship_type="HAS_SUPPLIER",  # Wrong type!
        )

        assert result.accepted is False
        assert result.rejection_reason == RejectionReason.RELATIONSHIP_MISMATCH
        assert result.relationship_passed is False
        assert result.suggested_relationship == "HAS_COMPETITOR"

    def test_valid_competitor_accepted(self):
        """Should accept valid competitor relationships."""
        validator = LayeredEntityValidator(skip_embedding=True)

        result = validator.validate(
            context="Our main competitors include Google, Microsoft, and Amazon.",
            mention="Microsoft",
            ticker="MSFT",
            company_name="Microsoft Corporation",
            relationship_type="HAS_COMPETITOR",
        )

        assert result.accepted is True
        assert result.rejection_reason == RejectionReason.ACCEPTED
        assert result.biographical_passed is True
        assert result.relationship_passed is True

    def test_valid_supplier_accepted(self):
        """Should accept valid supplier relationships."""
        validator = LayeredEntityValidator(skip_embedding=True)

        result = validator.validate(
            context="We purchase semiconductor components from Intel Corporation.",
            mention="Intel",
            ticker="INTC",
            company_name="Intel Corporation",
            relationship_type="HAS_SUPPLIER",
        )

        assert result.accepted is True
        assert result.rejection_reason == RejectionReason.ACCEPTED

    def test_embedding_rejection_integration(self):
        """Integration test: embedding rejection with skip_embedding=True falls through."""
        # When skip_embedding=True, the embedding check is skipped
        # This tests that the validator properly handles the flow
        validator = LayeredEntityValidator(skip_embedding=True)

        # This would fail embedding check in production (Brady power plant vs Brady Corp)
        # But with skip_embedding=True, it falls through to other checks
        result = validator.validate(
            context="Our Brady power plant produced electricity last quarter.",
            mention="Brady",
            ticker="BRC",
            company_name="Brady Corp",
            relationship_type="HAS_CUSTOMER",
        )

        # Without embedding check, this passes (no biographical or relationship mismatch)
        assert result.embedding_similarity is None

    def test_layered_order_biographical_before_relationship(self):
        """Biographical check should run before relationship verifier."""
        validator = LayeredEntityValidator(skip_embedding=True)

        # This context has both biographical pattern AND relationship mismatch
        # Biographical should fail first
        result = validator.validate(
            context="Mr. Smith serves as director of Microsoft. Our competitors include Microsoft.",
            mention="Microsoft",
            ticker="MSFT",
            company_name="Microsoft Corporation",
            relationship_type="HAS_SUPPLIER",  # Would be mismatched if it got there
        )

        # Should fail on biographical, not relationship
        assert result.accepted is False
        assert result.rejection_reason == RejectionReason.BIOGRAPHICAL_CONTEXT

    def test_validation_result_has_all_fields(self):
        """ValidationResult should have all expected fields."""
        validator = LayeredEntityValidator(skip_embedding=True)

        result = validator.validate(
            context="We compete with Microsoft.",
            mention="Microsoft",
            ticker="MSFT",
            company_name="Microsoft Corporation",
            relationship_type="HAS_COMPETITOR",
        )

        assert hasattr(result, "accepted")
        assert hasattr(result, "rejection_reason")
        assert hasattr(result, "embedding_similarity")
        assert hasattr(result, "embedding_passed")
        assert hasattr(result, "biographical_passed")
        assert hasattr(result, "relationship_passed")
        assert hasattr(result, "suggested_relationship")
        assert hasattr(result, "details")


class TestConvenienceFunction:
    """Tests for validate_entity convenience function."""

    def test_convenience_function_works(self):
        """Convenience function should create validator and validate."""
        result = validate_entity(
            context="Our competitors include Google.",
            mention="Google",
            ticker="GOOGL",
            company_name="Alphabet Inc.",
            relationship_type="HAS_COMPETITOR",
            skip_embedding=True,
        )

        assert isinstance(result, ValidationResult)
        assert result.accepted is True


class TestRejectionReason:
    """Tests for RejectionReason enum."""

    def test_all_reasons_exist(self):
        """All expected rejection reasons should exist."""
        assert RejectionReason.ACCEPTED
        assert RejectionReason.LOW_EMBEDDING_SIMILARITY
        assert RejectionReason.BIOGRAPHICAL_CONTEXT
        assert RejectionReason.RELATIONSHIP_MISMATCH
        assert RejectionReason.EXCHANGE_REFERENCE

    def test_reason_values(self):
        """Rejection reasons should have string values."""
        assert RejectionReason.ACCEPTED.value == "accepted"
        assert RejectionReason.LOW_EMBEDDING_SIMILARITY.value == "low_embedding_similarity"

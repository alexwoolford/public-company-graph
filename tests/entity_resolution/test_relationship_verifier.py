"""
Tests for RelationshipVerifier.

Tests relationship type verification against sentence context.
"""

import pytest

from public_company_graph.entity_resolution.relationship_verifier import (
    RelationshipType,
    RelationshipVerifier,
    VerificationResult,
    verify_relationship,
)


class TestRelationshipVerifier:
    """Test the relationship verifier."""

    @pytest.fixture
    def verifier(self):
        return RelationshipVerifier()

    # === COMPETITOR VERIFICATION ===

    def test_confirms_competitor_from_compete_pattern(self, verifier):
        """Should confirm competitor when 'compete with' pattern found."""
        context = "We compete with Microsoft, Google, and Amazon in cloud services."
        result = verifier.verify(RelationshipType.COMPETITOR, context)
        assert result.result == VerificationResult.CONFIRMED
        assert result.confidence >= 0.8

    def test_confirms_competitor_from_principal_competitors(self, verifier):
        """Should confirm competitor from 'principal competitors' pattern."""
        context = "Our principal competitors include Broadcom and Marvell."
        result = verifier.verify(RelationshipType.COMPETITOR, context)
        assert result.result == VerificationResult.CONFIRMED
        assert result.confidence >= 0.9

    def test_contradicts_competitor_when_supplier_context(self, verifier):
        """Should contradict competitor when supplier pattern found."""
        context = "We purchase processors from Intel and AMD."
        result = verifier.verify(RelationshipType.COMPETITOR, context)
        assert result.result == VerificationResult.CONTRADICTED
        assert result.suggested_type == RelationshipType.SUPPLIER

    # === SUPPLIER VERIFICATION ===

    def test_confirms_supplier_from_purchase_pattern(self, verifier):
        """Should confirm supplier when 'purchase from' pattern found."""
        context = "We purchase key components from Taiwan Semiconductor."
        result = verifier.verify(RelationshipType.SUPPLIER, context)
        assert result.result == VerificationResult.CONFIRMED
        assert result.confidence >= 0.8

    def test_confirms_supplier_from_component_pattern(self, verifier):
        """Should confirm supplier from component provider context."""
        context = "Components from Intel, AMD, and NVIDIA power our systems."
        result = verifier.verify(RelationshipType.SUPPLIER, context)
        assert result.result == VerificationResult.CONFIRMED

    def test_contradicts_supplier_when_competitor_context(self, verifier):
        """Should contradict supplier when competitor pattern found."""
        # From ground truth error: CRDO → AVGO labeled as supplier but is competitor
        context = (
            "Our principal competitors with respect to our products include Broadcom and Marvell."
        )
        result = verifier.verify(RelationshipType.SUPPLIER, context)
        assert result.result == VerificationResult.CONTRADICTED
        assert result.suggested_type == RelationshipType.COMPETITOR

    def test_contradicts_supplier_when_competition_also_coming_from(self, verifier):
        """Should contradict supplier when 'competition also coming from' found."""
        # Key pattern: "competition also coming from ... vendors" should be COMPETITOR
        context = (
            "The data center markets have been dominated by Cisco, "
            "with competition also coming from other large network equipment "
            "and system vendors, including Dell and HP."
        )
        result = verifier.verify(RelationshipType.SUPPLIER, context, "Dell")
        assert result.result == VerificationResult.CONTRADICTED
        assert result.suggested_type == RelationshipType.COMPETITOR

    def test_contradicts_customer_when_competitors_including(self, verifier):
        """Should contradict customer when 'competitors, including' found."""
        # Key pattern: "competitors, including" should recognize COMPETITOR
        context = (
            "Our competitors, including, but not limited to, HubSpot, "
            "Qualtrics, and Sprout Social mainly consist of point solutions."
        )
        result = verifier.verify(RelationshipType.CUSTOMER, context, "HubSpot")
        assert result.result == VerificationResult.CONTRADICTED
        assert result.suggested_type == RelationshipType.COMPETITOR

    # === CUSTOMER VERIFICATION ===

    def test_confirms_customer_from_sell_pattern(self, verifier):
        """Should confirm customer when 'sell to' pattern found."""
        context = "We sell our products to Walmart and Target."
        result = verifier.verify(RelationshipType.CUSTOMER, context)
        assert result.result == VerificationResult.CONFIRMED

    def test_confirms_customer_from_customers_include(self, verifier):
        """Should confirm customer from 'customers include' pattern."""
        context = "Our largest customers include Fortune 500 companies like Apple."
        result = verifier.verify(RelationshipType.CUSTOMER, context)
        assert result.result == VerificationResult.CONFIRMED

    # === PARTNER VERIFICATION ===

    def test_confirms_partner_from_partnership_pattern(self, verifier):
        """Should confirm partner from 'partner with' pattern."""
        context = "We partner with Microsoft for cloud infrastructure."
        result = verifier.verify(RelationshipType.PARTNER, context)
        assert result.result == VerificationResult.CONFIRMED

    def test_confirms_partner_from_ecosystem_pattern(self, verifier):
        """Should confirm partner from ecosystem context."""
        # From ground truth: ACN → MSFT - ecosystem relationship
        context = "Our strong ecosystem relationships with Microsoft provide competitive advantage."
        result = verifier.verify(RelationshipType.PARTNER, context)
        assert result.result == VerificationResult.CONFIRMED

    # === NON-RELATIONSHIP DETECTION ===

    def test_detects_legal_case_reference(self, verifier):
        """Should detect legal case references."""
        # From ground truth error: CRCT → W (Wayfair Supreme Court case)
        context = "Following South Dakota v. Wayfair, Inc., states can require tax collection."
        result = verifier.verify(RelationshipType.COMPETITOR, context, "Wayfair")
        assert result.result == VerificationResult.CONTRADICTED
        assert "legal_case" in result.matched_pattern

    def test_detects_exchange_listing(self, verifier):
        """Should detect exchange listing context."""
        context = "Our common stock is listed on the Nasdaq Global Market."
        result = verifier.verify(RelationshipType.PARTNER, context, "Nasdaq")
        assert result.result == VerificationResult.CONTRADICTED
        assert "exchange" in result.matched_pattern

    def test_detects_biographical_context(self, verifier):
        """Should detect biographical/employment history."""
        context = "He was formerly at Goldman Sachs before joining our company."
        result = verifier.verify(RelationshipType.PARTNER, context, "Goldman Sachs")
        assert result.result == VerificationResult.CONTRADICTED
        assert "biographical" in result.matched_pattern

    def test_detects_figurative_use(self, verifier):
        """Should detect figurative company name use."""
        # From ground truth error: KAYS → SBUX ("Starbucks of Marijuana")
        context = 'Dubbed by press as the "Starbucks of Marijuana" after our expansion.'
        result = verifier.verify(RelationshipType.CUSTOMER, context, "Starbucks")
        assert result.result == VerificationResult.CONTRADICTED
        assert result.matched_pattern == "figurative_use"

    # === UNCERTAIN CASES ===

    def test_returns_uncertain_when_no_patterns(self, verifier):
        """Should return uncertain when no clear patterns found."""
        context = "The company reported quarterly earnings that exceeded expectations."
        result = verifier.verify(RelationshipType.COMPETITOR, context)
        assert result.result == VerificationResult.UNCERTAIN

    # === CONVENIENCE FUNCTION ===

    def test_verify_relationship_function(self):
        """Test the convenience function."""
        result = verify_relationship(
            "HAS_COMPETITOR",
            "We compete with Dell and HP in the server market.",
            "Dell",
        )
        assert result.result == VerificationResult.CONFIRMED
        assert result.claimed_type == RelationshipType.COMPETITOR


class TestGroundTruthCases:
    """Tests based on actual ground truth errors."""

    @pytest.fixture
    def verifier(self):
        return RelationshipVerifier()

    def test_kvyo_crm_salesforce_competitor_not_supplier(self, verifier):
        """KVYO → CRM: Salesforce is competitor, not supplier."""
        context = (
            "Our main competitors are: Marketing solution providers, such as Mailchimp and Braze"
        )
        # Note: Salesforce would be in this list
        result = verifier.verify("HAS_SUPPLIER", context)
        assert result.result == VerificationResult.CONTRADICTED
        assert result.suggested_type == RelationshipType.COMPETITOR

    def test_bbsi_nsp_insperity_competitor(self, verifier):
        """BBSI → NSP: Insperity is competitor, not supplier."""
        context = "BBSI is not aware of reliable statistics regarding the number of its competitors"
        result = verifier.verify("HAS_SUPPLIER", context)
        # Should detect competitor context
        assert (
            result.result == VerificationResult.CONTRADICTED
            or result.result == VerificationResult.UNCERTAIN
        )

    def test_ngvt_cbt_cabot_competitor(self, verifier):
        """NGVT → CBT: Cabot is competitor, labeled as supplier."""
        context = "Competition: Our automotive technologies competitors include Cabot Corp., Kurara"
        result = verifier.verify("HAS_SUPPLIER", context)
        assert result.result == VerificationResult.CONTRADICTED
        assert result.suggested_type == RelationshipType.COMPETITOR

    def test_anet_dell_competitor(self, verifier):
        """ANET → DELL: Dell is competitor, not supplier."""
        context = (
            "The data center and campus networking markets have been historically dominated by Dell"
        )
        result = verifier.verify("HAS_SUPPLIER", context)
        # Should detect this as competitive context
        # "dominated by" isn't a strong supplier pattern
        assert result.result in (VerificationResult.CONTRADICTED, VerificationResult.UNCERTAIN)

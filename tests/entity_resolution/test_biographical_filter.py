"""
Tests for BiographicalContextFilter and ExchangeReferenceFilter.

These filters address 20% and 12% of errors respectively, based on
ground truth analysis.
"""

import pytest

from public_company_graph.entity_resolution.candidates import Candidate
from public_company_graph.entity_resolution.filters import (
    BiographicalContextFilter,
    ExchangeReferenceFilter,
    FilterReason,
)


class TestBiographicalContextFilter:
    """Test the biographical context filter."""

    @pytest.fixture
    def filter(self):
        return BiographicalContextFilter()

    def test_filters_director_board_seat(self, filter):
        """Should filter mentions of director's other board seats."""
        # From error #4: Celestica mentioned as director's board seat
        candidate = Candidate(
            text="Celestica Inc",
            sentence="Müller also serves as a director for Celestica Inc., a solutions-based company.",
            start_pos=37,
            end_pos=50,
            source_pattern="test",
        )
        result = filter.filter(candidate)
        assert not result.passed
        assert result.reason == FilterReason.BIOGRAPHICAL_CONTEXT

    def test_filters_prior_employer(self, filter):
        """Should filter mentions of executive's prior employer."""
        # From error #6: American Express as prior employer
        candidate = Candidate(
            text="American Express",
            sentence="Prior to joining the company, she served as VP at American Express.",
            start_pos=51,
            end_pos=67,
            source_pattern="test",
        )
        result = filter.filter(candidate)
        assert not result.passed
        assert result.reason == FilterReason.BIOGRAPHICAL_CONTEXT

    def test_filters_former_employer(self, filter):
        """Should filter 'formerly at' patterns."""
        # From error #10: Honeywell as former employer
        candidate = Candidate(
            text="Honeywell",
            sentence="Colucci was formerly at Honeywell International where he led engineering.",
            start_pos=24,
            end_pos=33,
            source_pattern="test",
        )
        result = filter.filter(candidate)
        assert not result.passed
        assert result.reason == FilterReason.BIOGRAPHICAL_CONTEXT

    def test_filters_work_history(self, filter):
        """Should filter employment history mentions."""
        # From error #11: Chevron as past employer
        candidate = Candidate(
            text="Chevron",
            sentence="His experience includes 15 years at Chevron in various technical roles.",
            start_pos=36,
            end_pos=43,
            source_pattern="test",
        )
        result = filter.filter(candidate)
        assert not result.passed
        assert result.reason == FilterReason.BIOGRAPHICAL_CONTEXT

    def test_passes_competitor_mention(self, filter):
        """Should NOT filter legitimate competitor mentions."""
        candidate = Candidate(
            text="Salesforce",
            sentence="Our main competitors include Salesforce, Adobe, and Microsoft.",
            start_pos=29,
            end_pos=39,
            source_pattern="test",
        )
        result = filter.filter(candidate)
        assert result.passed
        assert result.reason == FilterReason.PASSED

    def test_passes_supplier_mention(self, filter):
        """Should NOT filter legitimate supplier mentions."""
        candidate = Candidate(
            text="Intel",
            sentence="We purchase processors from Intel and AMD.",
            start_pos=28,
            end_pos=33,
            source_pattern="test",
        )
        result = filter.filter(candidate)
        assert result.passed
        assert result.reason == FilterReason.PASSED

    def test_passes_customer_mention(self, filter):
        """Should NOT filter legitimate customer mentions."""
        candidate = Candidate(
            text="Walmart",
            sentence="Our largest customers include Walmart, Target, and Costco.",
            start_pos=30,
            end_pos=37,
            source_pattern="test",
        )
        result = filter.filter(candidate)
        assert result.passed
        assert result.reason == FilterReason.PASSED

    def test_filters_joined_from_pattern(self, filter):
        """Should filter 'joined from' employment patterns."""
        candidate = Candidate(
            text="Google",
            sentence="She joined us from Google where she was a product manager.",
            start_pos=19,
            end_pos=25,
            source_pattern="test",
        )
        result = filter.filter(candidate)
        assert not result.passed
        assert result.reason == FilterReason.BIOGRAPHICAL_CONTEXT

    def test_filters_career_history(self, filter):
        """Should filter career history mentions."""
        candidate = Candidate(
            text="IBM",
            sentence="His career at IBM spanned two decades before he moved to our company.",
            start_pos=14,
            end_pos=17,
            source_pattern="test",
        )
        result = filter.filter(candidate)
        assert not result.passed
        assert result.reason == FilterReason.BIOGRAPHICAL_CONTEXT


class TestExchangeReferenceFilter:
    """Test the exchange reference filter."""

    @pytest.fixture
    def filter(self):
        return ExchangeReferenceFilter()

    def test_filters_ticker_notation(self, filter):
        """Should filter 'NASDAQ: AAPL' style references."""
        # From error #25: NASDAQ as exchange identifier
        candidate = Candidate(
            text="NASDAQ",
            sentence="The company trades as NASDAQ: EKSO on the public markets.",
            start_pos=22,
            end_pos=28,
            source_pattern="test",
        )
        result = filter.filter(candidate)
        assert not result.passed
        assert result.reason == FilterReason.EXCHANGE_REFERENCE

    def test_filters_listed_on_exchange(self, filter):
        """Should filter 'listed on NASDAQ' references."""
        # From error #17: Nasdaq as exchange venue
        candidate = Candidate(
            text="Nasdaq",
            sentence="Our common stock is listed on the Nasdaq Global Select Market.",
            start_pos=34,
            end_pos=40,
            source_pattern="test",
        )
        result = filter.filter(candidate)
        assert not result.passed
        assert result.reason == FilterReason.EXCHANGE_REFERENCE

    def test_filters_supreme_court_case(self, filter):
        """Should filter legal case references."""
        # From error #23: Wayfair Supreme Court case
        candidate = Candidate(
            text="Wayfair",
            sentence="Following the Supreme Court's decision in South Dakota v. Wayfair, Inc. (2018), we updated our tax collection.",
            start_pos=58,
            end_pos=65,
            source_pattern="test",
        )
        result = filter.filter(candidate)
        assert not result.passed
        assert result.reason == FilterReason.EXCHANGE_REFERENCE

    def test_filters_trading_on_exchange(self, filter):
        """Should filter 'trading on NYSE' references."""
        candidate = Candidate(
            text="NYSE",
            sentence="Our securities are currently trading on the NYSE under the symbol XYZ.",
            start_pos=44,
            end_pos=48,
            source_pattern="test",
        )
        result = filter.filter(candidate)
        assert not result.passed
        assert result.reason == FilterReason.EXCHANGE_REFERENCE

    def test_passes_nasdaq_as_competitor(self, filter):
        """Should NOT filter Nasdaq when mentioned as actual competitor."""
        candidate = Candidate(
            text="Nasdaq",
            sentence="We compete with Nasdaq, ICE, and other exchange operators.",
            start_pos=16,
            end_pos=22,
            source_pattern="test",
        )
        result = filter.filter(candidate)
        # This should pass because the context is about competition, not listing
        assert result.passed

    def test_passes_regular_company_mention(self, filter):
        """Should NOT filter regular company mentions."""
        candidate = Candidate(
            text="Microsoft",
            sentence="We partner with Microsoft for cloud services.",
            start_pos=16,
            end_pos=25,
            source_pattern="test",
        )
        result = filter.filter(candidate)
        assert result.passed
        assert result.reason == FilterReason.PASSED

"""
Unit tests for character-level similarity (n-gram matching).

Tests the "Wide" component of the Wide & Deep approach from JEL paper.
"""

from public_company_graph.entity_resolution.character import (
    CharacterMatcher,
    CharacterScore,
    extract_ngrams,
    is_exact_name_match,
    is_prefix_match,
    is_ticker_match,
    ngram_similarity,
    normalize_company_name,
    score_character_similarity,
)


class TestNormalizeCompanyName:
    """Tests for company name normalization."""

    def test_lowercase(self):
        assert normalize_company_name("APPLE") == "apple"
        assert normalize_company_name("Apple Inc.") == "apple"

    def test_removes_corporate_suffixes(self):
        assert normalize_company_name("Apple Inc.") == "apple"
        assert normalize_company_name("Microsoft Corporation") == "microsoft"
        assert normalize_company_name("Alphabet Holdings LLC") == "alphabet"
        assert normalize_company_name("Toyota Motor Corp") == "toyota motor"

    def test_removes_punctuation(self):
        assert normalize_company_name("Johnson & Johnson") == "johnson johnson"
        assert normalize_company_name("AT&T Inc.") == "at t"
        assert normalize_company_name("3M Company") == "3m"

    def test_removes_accents(self):
        assert normalize_company_name("Nestlé") == "nestle"
        assert normalize_company_name("Société Générale") == "societe generale"

    def test_empty_string(self):
        assert normalize_company_name("") == ""
        assert normalize_company_name("   ") == ""

    def test_preserves_suffix_when_disabled(self):
        result = normalize_company_name("Apple Inc.", remove_suffixes=False)
        assert "inc" in result


class TestExtractNgrams:
    """Tests for n-gram extraction."""

    def test_basic_ngrams(self):
        ngrams = extract_ngrams("apple")
        # Should include 2-grams through 5-grams
        assert "*a" in ngrams  # 2-gram at start
        assert "le*" in ngrams  # 3-gram at end
        assert "appl" in ngrams  # 4-gram
        assert "apple" in ngrams  # 5-gram

    def test_includes_word_markers(self):
        ngrams = extract_ngrams("test word")
        # Words should be included as features
        assert "*test*" in ngrams
        assert "*word*" in ngrams

    def test_empty_string(self):
        ngrams = extract_ngrams("")
        assert len(ngrams) == 0

    def test_short_string(self):
        ngrams = extract_ngrams("ab")
        assert len(ngrams) > 0  # Should still produce some n-grams


class TestNgramSimilarity:
    """Tests for n-gram Jaccard similarity."""

    def test_identical_strings(self):
        score = ngram_similarity("apple", "apple")
        assert score == 1.0

    def test_completely_different(self):
        score = ngram_similarity("apple", "xyz")
        assert score < 0.1

    def test_similar_strings(self):
        # Similar company name variations
        score = ngram_similarity("Microsoft", "Microsoft Corporation")
        assert score > 0.5

    def test_abbreviation_vs_full(self):
        # PayPal vs PYPL - should have low similarity (different surface forms)
        score = ngram_similarity("PayPal", "PYPL")
        assert score < 0.3

    def test_case_insensitive(self):
        score1 = ngram_similarity("APPLE", "apple")
        assert score1 == 1.0  # Should be identical after normalization

    def test_empty_strings(self):
        assert ngram_similarity("", "apple") == 0.0
        assert ngram_similarity("apple", "") == 0.0
        assert ngram_similarity("", "") == 0.0


class TestScoreCharacterSimilarity:
    """Tests for the full character scoring function."""

    def test_returns_character_score(self):
        result = score_character_similarity("Apple", "Apple Inc.")
        assert isinstance(result, CharacterScore)
        assert 0 <= result.score <= 1
        assert result.mention_normalized == "apple"
        assert result.candidate_normalized == "apple"

    def test_high_score_for_similar_names(self):
        result = score_character_similarity("Microsoft", "Microsoft Corporation")
        assert result.score > 0.6

    def test_low_score_for_different_names(self):
        result = score_character_similarity("Apple", "Google")
        assert result.score < 0.2

    def test_includes_ngram_counts(self):
        result = score_character_similarity("Apple", "Apple Inc.")
        assert result.shared_ngrams > 0
        assert result.total_ngrams > 0


class TestCharacterMatcher:
    """Tests for the CharacterMatcher class."""

    def test_score_single(self):
        matcher = CharacterMatcher()
        result = matcher.score("Apple", "Apple Inc.")
        assert isinstance(result, CharacterScore)

    def test_score_multiple(self):
        matcher = CharacterMatcher()
        candidates = [
            ("CIK1", "Apple Inc."),
            ("CIK2", "Microsoft Corporation"),
            ("CIK3", "Google LLC"),
        ]
        results = matcher.score_multiple("Apple", candidates)

        # Should return sorted by score descending
        assert len(results) == 3
        assert results[0][1] == "Apple Inc."  # Apple should be first
        assert results[0][2] > results[1][2]  # First should have highest score


class TestExactMatchers:
    """Tests for exact matching utilities."""

    def test_is_ticker_match(self):
        assert is_ticker_match("AAPL", "AAPL") is True
        assert is_ticker_match("aapl", "AAPL") is True
        assert is_ticker_match("AAPL", "aapl") is True
        assert is_ticker_match("AAPL", "MSFT") is False
        assert is_ticker_match("  AAPL  ", "AAPL") is True

    def test_is_exact_name_match(self):
        assert is_exact_name_match("Apple Inc.", "Apple Inc.") is True
        assert is_exact_name_match("apple", "Apple Inc.") is True  # Both normalize to "apple"
        assert is_exact_name_match("APPLE INC", "apple inc") is True
        assert is_exact_name_match("Apple", "Microsoft") is False

    def test_is_prefix_match(self):
        assert is_prefix_match("Microsoft", "Microsoft Corporation") is True
        assert is_prefix_match("Apple", "Apple Inc.") is True
        assert is_prefix_match("App", "Apple") is False  # Too short
        assert is_prefix_match("Google", "Alphabet") is False


class TestEdgeCases:
    """Tests for edge cases and special characters."""

    def test_numbers_in_names(self):
        score = ngram_similarity("3M Company", "3M")
        assert score > 0.3

    def test_special_characters(self):
        score = ngram_similarity("AT&T", "AT&T Inc.")
        assert score > 0.5

    def test_very_short_names(self):
        # Single character names should still work
        result = score_character_similarity("A", "Apple")
        assert 0 <= result.score <= 1

    def test_unicode_names(self):
        score = ngram_similarity("Société Générale", "Societe Generale SA")
        assert score > 0.5  # Should normalize accents

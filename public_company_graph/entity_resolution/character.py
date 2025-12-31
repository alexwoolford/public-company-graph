"""
Character-level similarity for entity resolution.

Uses n-gram based matching to handle name variations, abbreviations,
and spelling differences between mentions and company names.

Research Foundation:
- P62 (JEL/JPMorgan, 2021): Wide & Deep Learning for Entity Linking
  - "Wide Character Learning" component uses n-grams for surface-form matching
  - Handles variations like "PayPal Holdings" vs "PYPL" vs "Paypal Inc"

Key Insight: Character n-grams capture morphological similarity even when
exact string matching fails. "Microsoft" and "MSFT" share few n-grams,
but "Microsoft Corporation" and "Microsoft Corp" share many.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass


@dataclass
class CharacterScore:
    """Result of character-level similarity scoring."""

    score: float  # Similarity score 0-1
    mention_normalized: str  # Normalized mention
    candidate_normalized: str  # Normalized candidate name
    shared_ngrams: int  # Number of shared n-grams
    total_ngrams: int  # Total unique n-grams
    method: str = "ngram_jaccard"


# Common corporate suffixes to normalize
CORPORATE_SUFFIXES = {
    # Full forms
    "incorporated",
    "corporation",
    "company",
    "limited",
    "holdings",
    "enterprises",
    "industries",
    "international",
    "group",
    "partners",
    # Abbreviations
    "inc",
    "corp",
    "co",
    "ltd",
    "llc",
    "llp",
    "plc",
    "sa",
    "ag",
    "nv",
    "bv",
    "gmbh",
    "lp",
    "intl",
    "hldgs",
    "hlds",
    "grp",
}

# Common words to remove from company names for matching
NOISE_WORDS = {
    "the",
    "and",
    "of",
    "for",
    "a",
    "an",
}


def normalize_company_name(name: str, remove_suffixes: bool = True) -> str:
    """
    Normalize a company name for matching.

    Steps:
    1. Convert to lowercase
    2. Remove accents/diacritics
    3. Remove punctuation
    4. Optionally remove corporate suffixes
    5. Normalize whitespace

    Args:
        name: Company name to normalize
        remove_suffixes: Whether to remove corporate suffixes (Inc, Corp, etc.)

    Returns:
        Normalized name string
    """
    if not name:
        return ""

    # Convert to lowercase
    result = name.lower()

    # Remove accents (é → e, ü → u, etc.)
    result = unicodedata.normalize("NFKD", result)
    result = "".join(c for c in result if not unicodedata.combining(c))

    # Remove punctuation except apostrophes in contractions
    result = re.sub(r"[^\w\s']", " ", result)
    result = re.sub(r"'\s", " ", result)  # Remove trailing apostrophes

    # Split into words
    words = result.split()

    # Remove suffixes if requested
    if remove_suffixes:
        words = [w for w in words if w not in CORPORATE_SUFFIXES]

    # Remove noise words
    words = [w for w in words if w not in NOISE_WORDS]

    # Rejoin and normalize whitespace
    result = " ".join(words)
    result = re.sub(r"\s+", " ", result).strip()

    return result


def extract_ngrams(text: str, n_range: tuple[int, int] = (2, 5)) -> set[str]:
    """
    Extract character n-grams from text.

    Following JEL paper approach:
    1. Pad start/end with special character
    2. Generate n-grams for multiple n values
    3. Also include full words

    Args:
        text: Text to extract n-grams from
        n_range: Range of n values (min, max) inclusive

    Returns:
        Set of n-gram strings
    """
    if not text:
        return set()

    # Remove spaces for character n-grams (as in JEL paper)
    compact = text.replace(" ", "")

    # Pad with boundary markers
    padded = f"*{compact}*"

    ngrams = set()

    # Extract character n-grams
    for n in range(n_range[0], n_range[1] + 1):
        for i in range(len(padded) - n + 1):
            ngrams.add(padded[i : i + n])

    # Also add full words as features
    words = text.split()
    for word in words:
        if len(word) >= 2:
            ngrams.add(f"*{word}*")

    return ngrams


def ngram_similarity(
    text1: str,
    text2: str,
    n_range: tuple[int, int] = (2, 5),
    normalize: bool = True,
) -> float:
    """
    Compute Jaccard similarity of n-gram sets.

    Args:
        text1: First text
        text2: Second text
        n_range: Range of n values for n-grams
        normalize: Whether to normalize inputs first

    Returns:
        Jaccard similarity score 0-1
    """
    if normalize:
        text1 = normalize_company_name(text1)
        text2 = normalize_company_name(text2)

    if not text1 or not text2:
        return 0.0

    ngrams1 = extract_ngrams(text1, n_range)
    ngrams2 = extract_ngrams(text2, n_range)

    if not ngrams1 or not ngrams2:
        return 0.0

    intersection = len(ngrams1 & ngrams2)
    union = len(ngrams1 | ngrams2)

    return intersection / union if union > 0 else 0.0


def score_character_similarity(
    mention: str,
    candidate_name: str,
    n_range: tuple[int, int] = (2, 5),
) -> CharacterScore:
    """
    Score character-level similarity between mention and candidate.

    Args:
        mention: Raw mention text
        candidate_name: Company name to compare against
        n_range: Range of n values for n-grams

    Returns:
        CharacterScore with similarity and metadata
    """
    # Normalize both
    mention_norm = normalize_company_name(mention)
    candidate_norm = normalize_company_name(candidate_name)

    if not mention_norm or not candidate_norm:
        return CharacterScore(
            score=0.0,
            mention_normalized=mention_norm,
            candidate_normalized=candidate_norm,
            shared_ngrams=0,
            total_ngrams=0,
            method="empty_input",
        )

    # Extract n-grams
    mention_ngrams = extract_ngrams(mention_norm, n_range)
    candidate_ngrams = extract_ngrams(candidate_norm, n_range)

    # Calculate Jaccard
    shared = mention_ngrams & candidate_ngrams
    total = mention_ngrams | candidate_ngrams

    score = len(shared) / len(total) if total else 0.0

    return CharacterScore(
        score=score,
        mention_normalized=mention_norm,
        candidate_normalized=candidate_norm,
        shared_ngrams=len(shared),
        total_ngrams=len(total),
        method="ngram_jaccard",
    )


def score_candidates_character(
    mention: str,
    candidates: list[tuple[str, str]],  # (cik, name)
    n_range: tuple[int, int] = (2, 5),
) -> list[tuple[str, str, float]]:
    """
    Score multiple candidates against a mention using character similarity.

    Args:
        mention: Raw mention text
        candidates: List of (cik, name) tuples
        n_range: Range of n values for n-grams

    Returns:
        List of (cik, name, score) tuples sorted by score descending
    """
    results = []

    for cik, name in candidates:
        char_score = score_character_similarity(mention, name, n_range)
        results.append((cik, name, char_score.score))

    # Sort by score descending
    results.sort(key=lambda x: x[2], reverse=True)
    return results


class CharacterMatcher:
    """
    Character-level matcher using n-gram similarity.

    Provides efficient matching with optional caching of normalized forms.
    """

    def __init__(self, n_range: tuple[int, int] = (2, 5)):
        """
        Initialize the character matcher.

        Args:
            n_range: Range of n values for n-grams
        """
        self.n_range = n_range
        self._norm_cache: dict[str, str] = {}
        self._ngram_cache: dict[str, set[str]] = {}

    def score(self, mention: str, candidate_name: str) -> CharacterScore:
        """Score character similarity between mention and candidate."""
        return score_character_similarity(mention, candidate_name, self.n_range)

    def score_multiple(
        self,
        mention: str,
        candidates: list[tuple[str, str]],
    ) -> list[tuple[str, str, float]]:
        """Score multiple candidates against a mention."""
        return score_candidates_character(mention, candidates, self.n_range)

    def normalize(self, name: str) -> str:
        """Normalize a name with caching."""
        if name not in self._norm_cache:
            self._norm_cache[name] = normalize_company_name(name)
        return self._norm_cache[name]

    def get_ngrams(self, text: str) -> set[str]:
        """Get n-grams for text with caching."""
        if text not in self._ngram_cache:
            norm = self.normalize(text)
            self._ngram_cache[text] = extract_ngrams(norm, self.n_range)
        return self._ngram_cache[text]


# Threshold constants based on empirical testing
CHARACTER_THRESHOLD_HIGH = 0.50  # Strong character match
CHARACTER_THRESHOLD_MEDIUM = 0.30  # Reasonable match
CHARACTER_THRESHOLD_LOW = 0.15  # Weak match


def interpret_character_score(score: float) -> str:
    """
    Interpret a character similarity score.

    Returns:
        Human-readable interpretation
    """
    if score >= CHARACTER_THRESHOLD_HIGH:
        return "strong_character_match"
    elif score >= CHARACTER_THRESHOLD_MEDIUM:
        return "moderate_character_match"
    elif score >= CHARACTER_THRESHOLD_LOW:
        return "weak_character_match"
    else:
        return "poor_character_match"


# Bonus: Exact match variants
def is_ticker_match(mention: str, ticker: str) -> bool:
    """Check if mention is an exact ticker match (case-insensitive)."""
    return mention.upper().strip() == ticker.upper().strip()


def is_exact_name_match(mention: str, name: str) -> bool:
    """Check if mention exactly matches company name (normalized)."""
    return normalize_company_name(mention) == normalize_company_name(name)


def is_prefix_match(mention: str, name: str, min_length: int = 4) -> bool:
    """
    Check if mention is a prefix of company name.

    E.g., "Microsoft" matches "Microsoft Corporation"
    """
    mention_norm = normalize_company_name(mention)
    name_norm = normalize_company_name(name)

    if len(mention_norm) < min_length:
        return False

    return name_norm.startswith(mention_norm)

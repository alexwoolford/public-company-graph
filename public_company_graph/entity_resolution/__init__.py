"""
Entity Resolution Module.

Provides testable, modular entity resolution for company mentions in 10-K filings.

This module separates concerns into distinct, testable components:
- Candidate extraction (finding potential company mentions)
- Candidate filtering (blocklists, length checks)
- Candidate matching (lookup table resolution)
- Confidence scoring (multi-factor scoring)
- Semantic similarity (embedding-based disambiguation) [NEW]
- Character similarity (n-gram surface-form matching) [NEW]
- Combined scoring (Wide & Deep approach) [NEW]

Each component can be tested independently and swapped out for improved versions.

Research Foundation:
- P58 (Zeakis et al., 2023): Pre-trained Embeddings for Entity Resolution
- P62 (JEL/JPMorgan, 2021): Wide & Deep Learning for Entity Linking
"""

from public_company_graph.entity_resolution.candidates import (
    CandidateExtractor,
    extract_candidates,
)
from public_company_graph.entity_resolution.character import (
    CharacterMatcher,
    CharacterScore,
    ngram_similarity,
    normalize_company_name,
    score_character_similarity,
)
from public_company_graph.entity_resolution.combined_scorer import (
    CombinedScore,
    CombinedScorer,
    ConfidenceTier,
    compute_combined_score,
    create_scorer,
)
from public_company_graph.entity_resolution.filters import (
    CandidateFilter,
    FilterResult,
    filter_candidate,
)
from public_company_graph.entity_resolution.matchers import (
    CandidateMatcher,
    MatchResult,
    match_candidate,
)
from public_company_graph.entity_resolution.resolver import (
    EntityResolver,
    ResolutionResult,
)
from public_company_graph.entity_resolution.scoring import (
    ConfidenceScorer,
    compute_confidence,
)
from public_company_graph.entity_resolution.semantic import (
    SemanticScore,
    SemanticScorer,
    cosine_similarity,
    score_semantic_similarity,
)

__all__ = [
    # Candidates
    "CandidateExtractor",
    "extract_candidates",
    # Filters
    "CandidateFilter",
    "FilterResult",
    "filter_candidate",
    # Matchers
    "CandidateMatcher",
    "MatchResult",
    "match_candidate",
    # Original Scoring
    "ConfidenceScorer",
    "compute_confidence",
    # Main resolver
    "EntityResolver",
    "ResolutionResult",
    # Character similarity (Wide)
    "CharacterMatcher",
    "CharacterScore",
    "ngram_similarity",
    "normalize_company_name",
    "score_character_similarity",
    # Semantic similarity (Deep)
    "SemanticScorer",
    "SemanticScore",
    "cosine_similarity",
    "score_semantic_similarity",
    # Combined scoring (Wide & Deep)
    "CombinedScorer",
    "CombinedScore",
    "ConfidenceTier",
    "compute_combined_score",
    "create_scorer",
]

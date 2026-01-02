"""
LLM-based relationship verification.

Uses GPT to verify that a claimed relationship is actually present in the context.
More expensive but much more accurate than pattern matching for ambiguous cases.

All verifications are cached using the project's unified AppCache (diskcache).

Usage:
    verifier = LLMRelationshipVerifier()
    result = verifier.verify(
        context="We purchase key components from Intel...",
        source_company="ACME Corp",
        target_company="Intel",
        claimed_relationship="HAS_SUPPLIER",
    )
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

from public_company_graph.cache import get_cache
from public_company_graph.embeddings import get_openai_client

if TYPE_CHECKING:
    from openai import OpenAI

logger = logging.getLogger(__name__)

# Cache namespace for LLM verification results
CACHE_NAMESPACE = "llm_verification"


class VerificationResult(Enum):
    """Result of LLM verification."""

    CONFIRMED = "confirmed"
    REJECTED = "rejected"
    UNCERTAIN = "uncertain"


@dataclass
class LLMVerificationResult:
    """Result from LLM verification."""

    result: VerificationResult
    confidence: float
    explanation: str
    suggested_relationship: str | None  # If different from claimed
    cost_tokens: int


VERIFICATION_PROMPT = """You are verifying business relationships extracted from SEC 10-K filings.

Given a context sentence and a claimed relationship, determine if the relationship is VALID.

RELATIONSHIP TYPES:
- HAS_SUPPLIER: Target company provides goods/services TO the source company
- HAS_CUSTOMER: Target company purchases goods/services FROM the source company
- HAS_COMPETITOR: Target and source companies compete in the same market
- HAS_PARTNER: Target and source companies have a strategic partnership/alliance

CONTEXT:
{context}

CLAIMED RELATIONSHIP:
{source_company} --[{relationship_type}]--> {target_company}

VERIFICATION CRITERIA:
1. Is {target_company} actually mentioned in this context?
2. Does the context describe a {relationship_description} relationship?
3. Is the DIRECTION correct? (supplier provides TO us, customer buys FROM us)
4. Is this a current/ongoing relationship (not historical/hypothetical)?

Respond with JSON:
{{
    "verified": true/false,
    "confidence": 0.0-1.0,
    "explanation": "brief explanation",
    "actual_relationship": "HAS_SUPPLIER/HAS_CUSTOMER/HAS_COMPETITOR/HAS_PARTNER/NONE"
}}"""

RELATIONSHIP_DESCRIPTIONS = {
    "HAS_SUPPLIER": "the target supplies goods/services to the source",
    "HAS_CUSTOMER": "the target purchases goods/services from the source",
    "HAS_COMPETITOR": "the target competes with the source",
    "HAS_PARTNER": "the target has a strategic partnership with the source",
}


class LLMRelationshipVerifier:
    """
    Verifies relationships using GPT.

    Use for SUPPLIER/CUSTOMER where pattern-based extraction has low precision.
    All verifications are cached using the project's unified AppCache (diskcache).
    """

    def __init__(
        self,
        client: OpenAI | None = None,
        model: str = "gpt-4o-mini",  # Fast and cheap for verification
    ):
        """Initialize the verifier."""
        self._client = client or get_openai_client()
        self.model = model
        self._cache = get_cache()

    def _get_cache_key(
        self,
        context: str,
        source_company: str,
        target_company: str,
        claimed_relationship: str,
    ) -> str:
        """Generate cache key from verification inputs."""
        # Use first 500 chars of context for key (enough to be unique)
        key_str = f"{context[:500]}|{source_company}|{target_company}|{claimed_relationship}"
        return hashlib.sha256(key_str.encode()).hexdigest()[:16]

    def verify(
        self,
        context: str,
        source_company: str,
        target_company: str,
        claimed_relationship: str,
    ) -> LLMVerificationResult:
        """
        Verify a single relationship using LLM.

        Results are cached to avoid repeated API calls.

        Args:
            context: The text containing the relationship
            source_company: The company making the 10-K filing
            target_company: The company mentioned as having the relationship
            claimed_relationship: e.g., "HAS_SUPPLIER"

        Returns:
            LLMVerificationResult with verification outcome
        """
        # Check cache first
        cache_key = self._get_cache_key(
            context, source_company, target_company, claimed_relationship
        )
        cached = self._cache.get(CACHE_NAMESPACE, cache_key)
        if cached is not None:
            return LLMVerificationResult(
                result=VerificationResult(cached["result"]),
                confidence=cached["confidence"],
                explanation=cached["explanation"],
                suggested_relationship=cached.get("suggested_relationship"),
                cost_tokens=0,  # No cost for cached results
            )

        # Not cached - call LLM
        rel_desc = RELATIONSHIP_DESCRIPTIONS.get(
            claimed_relationship,
            "has a business relationship with",
        )

        prompt = VERIFICATION_PROMPT.format(
            context=context[:1500],  # Limit context size
            source_company=source_company,
            target_company=target_company,
            relationship_type=claimed_relationship,
            relationship_description=rel_desc,
        )

        try:
            response = self._client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a precise relationship extraction validator. Respond only with JSON.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.0,
                max_tokens=200,
            )

            content = response.choices[0].message.content
            total_tokens = response.usage.total_tokens if response.usage else 0

            # Parse JSON response
            try:
                data = json.loads(content)
            except json.JSONDecodeError:
                logger.warning(f"Failed to parse LLM response: {content}")
                return LLMVerificationResult(
                    result=VerificationResult.UNCERTAIN,
                    confidence=0.0,
                    explanation="Failed to parse LLM response",
                    suggested_relationship=None,
                    cost_tokens=total_tokens,
                )

            verified = data.get("verified", False)
            confidence = data.get("confidence", 0.5)
            explanation = data.get("explanation", "")
            actual_rel = data.get("actual_relationship", claimed_relationship)

            if verified and confidence >= 0.7:
                result = VerificationResult.CONFIRMED
            elif not verified and confidence >= 0.7:
                result = VerificationResult.REJECTED
            else:
                result = VerificationResult.UNCERTAIN

            llm_result = LLMVerificationResult(
                result=result,
                confidence=confidence,
                explanation=explanation,
                suggested_relationship=actual_rel if actual_rel != claimed_relationship else None,
                cost_tokens=total_tokens,
            )

            # Cache the result (no TTL - verifications are deterministic)
            self._cache.set(
                CACHE_NAMESPACE,
                cache_key,
                {
                    "result": result.value,
                    "confidence": confidence,
                    "explanation": explanation,
                    "suggested_relationship": llm_result.suggested_relationship,
                },
            )

            return llm_result

        except Exception as e:
            logger.error(f"LLM verification failed: {e}")
            return LLMVerificationResult(
                result=VerificationResult.UNCERTAIN,
                confidence=0.0,
                explanation=f"Error: {e}",
                suggested_relationship=None,
                cost_tokens=0,
            )

    def cache_stats(self) -> dict:
        """Get cache statistics for LLM verifications."""
        return {
            "namespace": CACHE_NAMESPACE,
            "count": self._cache.count(CACHE_NAMESPACE),
        }

    def verify_batch(
        self,
        relationships: list[dict],
        max_concurrent: int = 5,
    ) -> list[LLMVerificationResult]:
        """
        Verify multiple relationships.

        Args:
            relationships: List of dicts with context, source, target, relationship
            max_concurrent: Max concurrent API calls

        Returns:
            List of verification results
        """
        results = []
        for rel in relationships:
            result = self.verify(
                context=rel["context"],
                source_company=rel["source_company"],
                target_company=rel["target_company"],
                claimed_relationship=rel["relationship_type"],
            )
            results.append(result)

        return results


def estimate_verification_cost(
    num_relationships: int,
    avg_context_tokens: int = 300,
    model: str = "gpt-4o-mini",
) -> dict[str, float]:
    """
    Estimate cost for LLM verification.

    Args:
        num_relationships: Number of relationships to verify
        avg_context_tokens: Average tokens per context
        model: Model to use

    Returns:
        Dict with estimated costs
    """
    # Approximate pricing (as of 2024)
    pricing = {
        "gpt-4o-mini": {"input": 0.00015, "output": 0.0006},  # per 1K tokens
        "gpt-4o": {"input": 0.005, "output": 0.015},
        "gpt-4-turbo": {"input": 0.01, "output": 0.03},
    }

    if model not in pricing:
        model = "gpt-4o-mini"

    # Estimate tokens per verification
    prompt_tokens = 200 + avg_context_tokens  # System + prompt + context
    output_tokens = 100  # JSON response

    total_input = num_relationships * prompt_tokens
    total_output = num_relationships * output_tokens

    input_cost = (total_input / 1000) * pricing[model]["input"]
    output_cost = (total_output / 1000) * pricing[model]["output"]

    return {
        "model": model,
        "num_relationships": num_relationships,
        "estimated_input_tokens": total_input,
        "estimated_output_tokens": total_output,
        "estimated_cost_usd": input_cost + output_cost,
        "per_relationship_cost": (input_cost + output_cost) / num_relationships,
    }


if __name__ == "__main__":
    # Example usage and cost estimation
    print("=" * 60)
    print("LLM Verification Cost Estimation")
    print("=" * 60)

    for count in [100, 1000, 10000]:
        est = estimate_verification_cost(count)
        print(f"\n{count:,} relationships:")
        print(f"  Model: {est['model']}")
        print(f"  Estimated cost: ${est['estimated_cost_usd']:.2f}")
        print(f"  Per relationship: ${est['per_relationship_cost']:.4f}")

"""
Unified Preference Derivation Engine.

Consolidates the three preference derivation mechanisms into a single
configurable engine:
1. RANKING_SCORE_THRESHOLD: genrm_preference.py style (ranking <= 2 -> A, >= 5 -> B)
2. RANKING_SCORE_DISCRETE: genrm_dspy.py style (A -> 1, B -> 6, tie -> 3)
3. ERROR_DIFFERENCE: oracle_preference.py style (error diff with tie margin)
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional, Tuple


class PreferenceDerivationStrategy(Enum):
    """Available strategies for deriving preference from scores."""

    # genrm_preference.py style: ranking_score <= 2 -> A, >= 5 -> B, else tie
    RANKING_SCORE_THRESHOLD = "ranking_threshold"

    # genrm_dspy.py style: maps preference string to ranking score
    RANKING_SCORE_DISCRETE = "ranking_discrete"

    # oracle_preference.py style: derives from error difference with tie margin
    ERROR_DIFFERENCE = "error_difference"


@dataclass
class PreferenceEngineConfig:
    """Configuration for preference derivation engine."""

    strategy: PreferenceDerivationStrategy = PreferenceDerivationStrategy.RANKING_SCORE_THRESHOLD

    # For RANKING_SCORE_THRESHOLD strategy
    ranking_threshold_a: int = 2  # ranking_score <= this -> A wins
    ranking_threshold_b: int = 5  # ranking_score >= this -> B wins

    # For ERROR_DIFFERENCE strategy
    tie_margin: float = 5.0  # errors within this margin are ties
    confidence_floor: float = 0.5  # minimum confidence for ties

    # For all strategies
    confidence_scale: float = 0.125  # for calculating confidence from score diff


class PreferenceEngine:
    """
    Unified engine for deriving preference from scores.

    Supports three strategies:
    1. RANKING_SCORE_THRESHOLD: Uses integer ranking (1-6) with thresholds
    2. RANKING_SCORE_DISCRETE: Maps preference string to discrete ranking
    3. ERROR_DIFFERENCE: Compares absolute errors with tie margin
    """

    def __init__(self, config: Optional[PreferenceEngineConfig] = None):
        """
        Initialize the preference engine.

        Args:
            config: Configuration for preference derivation
        """
        self.config = config or PreferenceEngineConfig()

    def derive_preference(
        self,
        score_a: Optional[float] = None,
        score_b: Optional[float] = None,
        ranking_score: Optional[int] = None,
        preference_string: Optional[str] = None,
        error_a: Optional[float] = None,
        error_b: Optional[float] = None,
    ) -> Tuple[str, float]:
        """
        Derive preference and confidence from scores.

        Args:
            score_a: Score for candidate A (helpfulness, quality, etc.)
            score_b: Score for candidate B
            ranking_score: Integer ranking (1-6, lower = A better)
            preference_string: String preference ("A", "B", "tie")
            error_a: Error/deviation for candidate A
            error_b: Error/deviation for candidate B

        Returns:
            Tuple of (preferred, confidence) where preferred is "A", "B", or "tie"
        """
        strategy = self.config.strategy

        if strategy == PreferenceDerivationStrategy.RANKING_SCORE_THRESHOLD:
            return self._from_ranking_threshold(ranking_score, score_a, score_b)

        elif strategy == PreferenceDerivationStrategy.RANKING_SCORE_DISCRETE:
            return self._from_ranking_discrete(preference_string, score_a, score_b)

        elif strategy == PreferenceDerivationStrategy.ERROR_DIFFERENCE:
            return self._from_error_difference(error_a, error_b)

        else:
            raise ValueError(f"Unknown strategy: {strategy}")

    def _from_ranking_threshold(
        self,
        ranking_score: Optional[int],
        score_a: Optional[float] = None,
        score_b: Optional[float] = None,
    ) -> Tuple[str, float]:
        """
        Derive preference from ranking score with thresholds.

        Ranking score interpretation (1-6):
            1 = A is much better than B
            2 = A is better than B
            3 = A is slightly better than B
            4 = B is slightly better than A
            5 = B is better than A
            6 = B is much better than A

        Args:
            ranking_score: Integer ranking 1-6
            score_a: Optional helpfulness score for A
            score_b: Optional helpfulness score for B

        Returns:
            Tuple of (preferred, confidence)
        """
        if ranking_score is None:
            # Fall back to score difference if available
            if score_a is not None and score_b is not None:
                diff = score_a - score_b
                if diff > 0.5:
                    return ("A", min(0.5 + diff * self.config.confidence_scale, 1.0))
                elif diff < -0.5:
                    return ("B", min(0.5 + abs(diff) * self.config.confidence_scale, 1.0))
                else:
                    return ("tie", self.config.confidence_floor)
            return ("tie", self.config.confidence_floor)

        # Standard threshold logic
        if ranking_score <= self.config.ranking_threshold_a:
            # A wins: confidence increases as ranking decreases
            confidence = (self.config.ranking_threshold_a + 1 - ranking_score) * 0.3 + 0.4
            return ("A", min(confidence, 1.0))

        elif ranking_score >= self.config.ranking_threshold_b:
            # B wins: confidence increases as ranking increases
            confidence = (ranking_score - self.config.ranking_threshold_b + 1) * 0.3 + 0.4
            return ("B", min(confidence, 1.0))

        else:
            # Tie: ranking 3 or 4
            return ("tie", self.config.confidence_floor)

    def _from_ranking_discrete(
        self,
        preference_string: Optional[str],
        score_a: Optional[float] = None,
        score_b: Optional[float] = None,
    ) -> Tuple[str, float]:
        """
        Derive preference from discrete preference string.

        Maps preference string to ranking score, then derives confidence.

        Args:
            preference_string: "A", "B", or "tie"
            score_a: Optional helpfulness score for A
            score_b: Optional helpfulness score for B

        Returns:
            Tuple of (preferred, confidence)
        """
        if preference_string is None:
            return ("tie", self.config.confidence_floor)

        # Normalize preference string
        normalized = preference_string.upper().strip()

        # Calculate confidence from score difference if available
        if score_a is not None and score_b is not None:
            score_diff = abs(score_a - score_b)
            if normalized == "TIE" or normalized not in ("A", "B"):
                confidence = self.config.confidence_floor
            else:
                confidence = min(0.5 + score_diff * self.config.confidence_scale, 1.0)
        else:
            # Default confidence based on preference
            confidence = 0.7 if normalized in ("A", "B") else self.config.confidence_floor

        # Return normalized preference
        if normalized == "A":
            return ("A", confidence)
        elif normalized == "B":
            return ("B", confidence)
        else:
            return ("tie", self.config.confidence_floor)

    def _from_error_difference(
        self,
        error_a: Optional[float],
        error_b: Optional[float],
    ) -> Tuple[str, float]:
        """
        Derive preference from error difference.

        Lower error is better. If errors are within tie_margin, it's a tie.

        Args:
            error_a: Error/deviation for candidate A
            error_b: Error/deviation for candidate B

        Returns:
            Tuple of (preferred, confidence)
        """
        if error_a is None or error_b is None:
            return ("tie", self.config.confidence_floor)

        diff = error_a - error_b

        # Check if within tie margin
        if abs(diff) < self.config.tie_margin:
            return ("tie", self.config.confidence_floor)

        # Lower error is better
        preferred = "A" if diff < 0 else "B"

        # Calculate confidence: scales from floor to 1.0 based on error difference
        scaled = (abs(diff) - self.config.tie_margin) / max(1e-6, 2 * self.config.tie_margin)
        confidence = min(1.0, max(self.config.confidence_floor, self.config.confidence_floor + scaled))

        return (preferred, confidence)

    def ranking_to_preference(self, ranking_score: int) -> str:
        """
        Convert ranking score to preference string.

        Args:
            ranking_score: Integer 1-6

        Returns:
            "A", "B", or "tie"
        """
        if ranking_score <= self.config.ranking_threshold_a:
            return "A"
        elif ranking_score >= self.config.ranking_threshold_b:
            return "B"
        else:
            return "tie"

    def preference_to_ranking(self, preference: str) -> int:
        """
        Convert preference string to ranking score.

        Maps to discrete values compatible with RANKING_SCORE_DISCRETE strategy.

        Args:
            preference: "A", "B", or "tie"

        Returns:
            Integer ranking 1, 3, or 6
        """
        normalized = preference.upper().strip()
        if normalized == "A":
            return 1
        elif normalized == "B":
            return 6
        else:
            return 3


# Default instances for common use cases
DEFAULT_GENRM_ENGINE = PreferenceEngine(
    PreferenceEngineConfig(
        strategy=PreferenceDerivationStrategy.RANKING_SCORE_THRESHOLD,
        ranking_threshold_a=2,
        ranking_threshold_b=5,
        confidence_floor=0.5,
    )
)

DEFAULT_ORACLE_ENGINE = PreferenceEngine(
    PreferenceEngineConfig(
        strategy=PreferenceDerivationStrategy.ERROR_DIFFERENCE,
        tie_margin=5.0,
        confidence_floor=0.5,
    )
)

"""
DSPy Module Wrapper for GenRM Judge.

Wraps GenRMJudge as a DSPy module to enable prompt optimization.
"""

import dspy
from typing import Optional, List, Tuple, Dict

from treepo._research.training.preference.genrm import GenRMJudge, GenRMResult


def _normalize_preference(raw_preference: str) -> str:
    """
    Normalize LLM preference output to 'A', 'B', or 'tie'.

    Handles various formats like 'A', 'a', 'Response A', 'summary A is better', etc.

    NOTE: This is a preprocessing step for raw LLM output. Once normalized,
    PreferenceEngine.RANKING_SCORE_DISCRETE can be used to compute confidence.
    """
    if not raw_preference:
        return 'tie'

    pref_lower = raw_preference.lower().strip()

    # Check for explicit tie indicators
    if 'tie' in pref_lower or 'equal' in pref_lower or 'neither' in pref_lower or 'both' in pref_lower:
        return 'tie'

    # Check for A indicators (prioritize explicit mentions)
    if pref_lower == 'a' or pref_lower.startswith('a ') or 'response a' in pref_lower or 'summary a' in pref_lower or 'option a' in pref_lower:
        return 'A'

    # Check for B indicators
    if pref_lower == 'b' or pref_lower.startswith('b ') or 'response b' in pref_lower or 'summary b' in pref_lower or 'option b' in pref_lower:
        return 'B'

    # Check for numeric indicators (1 = A, 2 = B)
    if pref_lower in ('1', 'response 1', 'summary 1'):
        return 'A'
    if pref_lower in ('2', 'response 2', 'summary 2'):
        return 'B'

    # Fallback: check if 'a' or 'b' appears anywhere
    if 'a' in pref_lower and 'b' not in pref_lower:
        return 'A'
    if 'b' in pref_lower and 'a' not in pref_lower:
        return 'B'

    # Default to tie if unclear
    return 'tie'


class GenRMComparisonSignature(dspy.Signature):
    """
    Signature for GenRM pairwise comparison.

    Optimizable instructions for how GenRM should compare summaries.
    """

    context: str = dspy.InputField(
        desc="Description of what information should be preserved in the summary"
    )
    original_text: str = dspy.InputField(
        desc="The original text being summarized"
    )
    summary_a: str = dspy.InputField(
        desc="First candidate summary to compare"
    )
    summary_b: str = dspy.InputField(
        desc="Second candidate summary to compare"
    )
    law_type: str = dspy.InputField(
        desc="Type of OPS law being evaluated (sufficiency, idempotence, merge)"
    )

    # Output fields
    preference: str = dspy.OutputField(
        desc="Which summary is better: 'A', 'B', or 'tie'"
    )
    reasoning: str = dspy.OutputField(
        desc="Brief explanation of why this summary is preferred"
    )
    score_a: str = dspy.OutputField(
        desc="Helpfulness score for summary A (1-5)"
    )
    score_b: str = dspy.OutputField(
        desc="Helpfulness score for summary B (1-5)"
    )


class GenRMPromptSignature(dspy.Signature):
    """
    Signature for prompt tuning of GenRM comparisons.

    Produces an extra instruction block that is appended to GenRM's prompt.
    Keep the guidance short and concrete.
    """

    rubric: str = dspy.InputField(
        desc="Rubric describing what information should be preserved"
    )
    law_type: str = dspy.InputField(
        desc="OPS law type being evaluated (sufficiency, idempotence, merge)"
    )

    extra_context: str = dspy.OutputField(
        desc="Concise extra judging instructions for GenRM"
    )


class GenRMComparisonModule(dspy.Module):
    """
    DSPy module for pairwise comparison.

    Supports two modes:
    - GenRM mode: Uses NVIDIA's specialized GenRM reward model (default)
    - DSPy mode: Uses DSPy ChainOfThought with optimizable prompts (for testing/fallback)
    - Prompt-tuned GenRM: Uses DSPy to optimize extra context added to GenRM prompts

    When using GenRM mode, the comparison uses the specialized reward model API.
    When using DSPy mode, comparison prompts can be optimized via DSPy optimizers.
    """

    def __init__(
        self,
        genrm_judge: Optional[GenRMJudge] = None,
        base_url: Optional[str] = None,
        model_name: Optional[str] = None,
        use_dspy_predictor: bool = False,
        use_dspy_prompt: bool = False,
        cache_prompt: bool = True,
        prompt_lm: Optional[dspy.LM] = None,
    ):
        """
        Initialize comparison module.

        Args:
            genrm_judge: Existing GenRMJudge instance (or create new one)
            base_url: GenRM server URL (None = auto-detect from config)
            model_name: GenRM model name (auto-detected if None)
            use_dspy_predictor: If True, use DSPy ChainOfThought instead of GenRM
            use_dspy_prompt: If True, use DSPy to generate extra prompt context for GenRM
            cache_prompt: Cache prompt guidance by (rubric, law_type)
        """
        super().__init__()

        if use_dspy_predictor and use_dspy_prompt:
            raise ValueError("use_dspy_predictor and use_dspy_prompt cannot both be True")

        self.use_dspy_predictor = use_dspy_predictor
        self.use_dspy_prompt = use_dspy_prompt
        self.cache_prompt = cache_prompt
        self.prompt_lm = prompt_lm
        self._prompt_cache: Dict[Tuple[str, str], str] = {}

        # Create DSPy predictor for comparison (optimizable via DSPy)
        self.compare = dspy.ChainOfThought(GenRMComparisonSignature)

        # Create DSPy predictor for prompt guidance (optimizable via DSPy)
        if self.use_dspy_prompt:
            self.prompt_adapter = dspy.Predict(GenRMPromptSignature)

        # Create GenRM judge (used when use_dspy_predictor=False)
        if not use_dspy_predictor:
            if genrm_judge is not None:
                self.judge = genrm_judge
            else:
                self.judge = GenRMJudge(
                    base_url=base_url,
                    model_name=model_name,
                )
        else:
            self.judge = None

    def forward(
        self,
        context: str,
        original_text: str,
        summary_a: str,
        summary_b: str,
        law_type: str = "sufficiency",
    ) -> dspy.Prediction:
        """
        Compare two summaries.

        Uses either GenRM (specialized reward model) or DSPy predictor
        depending on initialization settings.

        Args:
            context: What information to preserve
            original_text: Original text
            summary_a: First summary
            summary_b: Second summary
            law_type: Type of law (sufficiency, idempotence, merge)

        Returns:
            dspy.Prediction with preference, reasoning, scores
        """
        if self.use_dspy_predictor:
            # Use DSPy ChainOfThought predictor (optimizable)
            result = self.compare(
                context=context,
                original_text=original_text,
                summary_a=summary_a,
                summary_b=summary_b,
                law_type=law_type,
            )

            # Parse scores from string output
            try:
                score_a = float(result.score_a)
            except (ValueError, TypeError):
                score_a = 3.0  # Default middle score

            try:
                score_b = float(result.score_b)
            except (ValueError, TypeError):
                score_b = 3.0

            # Normalize preference to ensure consistent 'A', 'B', or 'tie'
            normalized_preference = _normalize_preference(result.preference)

            # Derive ranking_score from normalized preference
            if normalized_preference == "A":
                ranking_score = 1
            elif normalized_preference == "B":
                ranking_score = 6
            else:
                ranking_score = 3

            # Derive confidence from score difference (similar to GenRM)
            score_diff = abs(score_a - score_b)
            if normalized_preference == 'tie':
                confidence = 0.5
            else:
                # Higher score difference = higher confidence (0.5 to 1.0)
                confidence = min(0.5 + score_diff * 0.125, 1.0)

            return dspy.Prediction(
                preference=normalized_preference,
                reasoning=result.reasoning,
                score_a=str(score_a),
                score_b=str(score_b),
                helpfulness_a=score_a,
                helpfulness_b=score_b,
                ranking_score=ranking_score,
                confidence=confidence,
            )
        elif self.use_dspy_prompt:
            extra_context = self._get_extra_context(context, law_type)
            result: GenRMResult = self.judge.compare(
                context=context,
                original_text=original_text,
                summary_a=summary_a,
                summary_b=summary_b,
                law_type=law_type,
                extra_context=extra_context,
            )

            preference = result.preferred
            return dspy.Prediction(
                preference=preference,
                reasoning=result.reasoning,
                score_a=str(result.helpfulness_a),
                score_b=str(result.helpfulness_b),
                helpfulness_a=result.helpfulness_a,
                helpfulness_b=result.helpfulness_b,
                ranking_score=result.ranking_score,
                confidence=result.confidence,
            )
        else:
            # Use GenRM specialized reward model
            result: GenRMResult = self.judge.compare(
                context=context,
                original_text=original_text,
                summary_a=summary_a,
                summary_b=summary_b,
                law_type=law_type,
            )

            # Convert GenRM result to DSPy prediction format
            preference = result.preferred  # 'A', 'B', or 'tie'

            return dspy.Prediction(
                preference=preference,
                reasoning=result.reasoning,
                score_a=str(result.helpfulness_a),
                score_b=str(result.helpfulness_b),
                helpfulness_a=result.helpfulness_a,
                helpfulness_b=result.helpfulness_b,
                ranking_score=result.ranking_score,
                confidence=result.confidence,
            )

    def _get_extra_context(self, rubric: str, law_type: str) -> str:
        """Generate (or reuse) extra judging context for GenRM."""
        cache_key = (rubric, law_type)
        if self.cache_prompt and cache_key in self._prompt_cache:
            return self._prompt_cache[cache_key]

        extra_context = ""
        try:
            if self.prompt_lm is not None:
                with dspy.context(lm=self.prompt_lm):
                    result = self.prompt_adapter(rubric=rubric, law_type=law_type)
            else:
                result = self.prompt_adapter(rubric=rubric, law_type=law_type)
            extra_context = getattr(result, "extra_context", "")
        except Exception:
            extra_context = ""

        if not isinstance(extra_context, str):
            extra_context = str(extra_context)

        # Use full context - truncation corrupts the prompt
        if self.cache_prompt:
            self._prompt_cache[cache_key] = extra_context

        return extra_context

    def get_prompt_context(self, rubric: str, law_type: str) -> str:
        """Return the extra GenRM prompt context for a rubric/law type."""
        if not self.use_dspy_prompt:
            return ""
        return self._get_extra_context(rubric, law_type)

    def compare_batch(
        self,
        comparisons: List[Tuple[str, str, str, str, str]],
    ) -> List[dspy.Prediction]:
        """
        Compare multiple pairs in parallel.

        Args:
            comparisons: List of (context, original, summary_a, summary_b, law_type) tuples

        Returns:
            List of predictions
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed

        results = [None] * len(comparisons)

        with ThreadPoolExecutor(max_workers=10) as executor:
            future_to_idx = {
                executor.submit(
                    self.forward,
                    context=comp[0],
                    original_text=comp[1],
                    summary_a=comp[2],
                    summary_b=comp[3],
                    law_type=comp[4],
                ): idx
                for idx, comp in enumerate(comparisons)
            }

            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    results[idx] = future.result()
                except Exception as e:
                    # Return error prediction
                    results[idx] = dspy.Prediction(
                        preference="tie",
                        reasoning=f"Error: {str(e)}",
                        score_a="0",
                        score_b="0",
                        helpfulness_a=0.0,
                        helpfulness_b=0.0,
                        ranking_score=3,
                        confidence=0.0,
                    )

        return results

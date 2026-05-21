"""
Manifesto RILE Pipeline Components.

DSPy modules and pipelines for processing political manifestos through
OPS trees with RILE scoring.

These components are manifesto-specific and use RILE preservation rubrics.

**2× concatenation invariant (repo-wide):** the LLM summarizer/merger must
be *able* to emit up to 2× its input length so that concatenation (carrying
both leaves through the merge) remains a valid operation. This mirrors
``FNO_HEAD_MIN_RATIO=2`` in
``parallel/unified_g_v1/src/unified_g_v1/core/tensor_program.py`` which
enforces ``state_dim >= 2 * summary_dim`` for learned mergers on the
sketch/FNO side. Same principle, different space: dimensional lower bound
there vs. token-length upper bound here. In practice we rarely use all 2×;
the prompt-level *target* is much tighter (see ``target_token_ratio``).
"""

import logging
from typing import Optional, TYPE_CHECKING
from urllib.parse import urlparse

import dspy

from treepo._research.tree.builder import TreeBuilder, BuildConfig
from treepo._research.core.protocols import format_merge_input
from treepo._research.core.prompting import parse_numeric_score
from .pipeline_config import (
    CONCAT_RATIO,
    DEFAULT_TARGET_RATIO,
    DEFAULT_PROMPT_OVERHEAD_TOKENS,
    DEFAULT_INPUT_SAFETY_RATIO,
    DEFAULT_INPUT_SAFETY_FLOOR,
    DEFAULT_MIN_TARGET_TOKENS,
    DEFAULT_MIN_BUDGET_TOKENS,
    resolve_context_window,
)
from .rubrics import RILE_PRESERVATION_RUBRIC, RILE_TASK_CONTEXT
from .constants import RILE_MIN, RILE_MAX

if TYPE_CHECKING:
    from treepo._research.core.strategy import DSPyStrategy, TournamentStrategy, TournamentConfig

logger = logging.getLogger(__name__)


# =============================================================================
# DSPy Signatures
# =============================================================================

class RILESummarize(dspy.Signature):
    """Summarize political text preserving information relevant to left-right positioning."""
    rubric: str = dspy.InputField(desc="What information to preserve for RILE scoring")
    text: str = dspy.InputField(desc="Political text chunk to summarize")

    summary: str = dspy.OutputField(desc="Concise summary preserving left/right position indicators")


class RILEMerge(dspy.Signature):
    """Merge two summaries while preserving political position information."""
    rubric: str = dspy.InputField(desc="What information to preserve for RILE scoring")
    summary1: str = dspy.InputField(desc="First summary to merge")
    summary2: str = dspy.InputField(desc="Second summary to merge")

    merged_summary: str = dspy.OutputField(desc="Combined summary preserving all position indicators from both inputs")


class RILEScoreSignature(dspy.Signature):
    """Score political text on left-right scale."""
    task_context: str = dspy.InputField(desc="The RILE scoring task and scale explanation")
    summary: str = dspy.InputField(desc="Summarized political manifesto to score")

    reasoning: str = dspy.OutputField(desc="Analysis identifying left vs right indicators and their balance")
    score: float = dspy.OutputField(desc="RILE score from -100 (far left) to +100 (far right)")


# =============================================================================
# Helper Functions
# =============================================================================

# Budget ratios + context-window resolution live in pipeline_config. See
# pipeline_config.CONCAT_RATIO / DEFAULT_TARGET_RATIO for the invariant
# and its link to ``FNO_HEAD_MIN_RATIO`` in
# ``parallel/unified_g_v1/src/unified_g_v1/core/tensor_program.py``.

# Rough char→token ratio for English+Latin-script. Gemma tokenizer averages
# ~3.3-3.7 chars per token across the manifesto corpus.
_CHARS_PER_TOKEN = 3.3


def _estimate_tokens(text: str) -> int:
    """Cheap heuristic: char count / 3.3. Good enough for budgeting, not tokenization."""
    return max(1, int(len(text) / _CHARS_PER_TOKEN))


def compute_output_budget(
    input_text: str,
    *,
    ratio: float = CONCAT_RATIO,
    target_ratio: float = DEFAULT_TARGET_RATIO,
    hard_max: Optional[int] = None,
    min_budget: int = DEFAULT_MIN_BUDGET_TOKENS,
    min_target: int = DEFAULT_MIN_TARGET_TOKENS,
    context_window: Optional[int] = None,
    prompt_overhead_tokens: int = DEFAULT_PROMPT_OVERHEAD_TOKENS,
    input_safety_ratio: float = DEFAULT_INPUT_SAFETY_RATIO,
    input_safety_floor: int = DEFAULT_INPUT_SAFETY_FLOOR,
) -> tuple[int, int]:
    """Returns ``(target_tokens, hard_max_tokens)`` for a summarize/merge call.

    - ``ratio`` (default 2.0) is the **hard ceiling** at the API level —
      concatenation up to ``ratio × input`` is physically allowed. This is
      per user spec: "sometimes the best answer is concatenation."
    - ``target_ratio`` (default 0.3) is the **soft target** communicated to
      the LM via the prompt — nudges compression toward Benoit-style
      short summaries without forbidding the 2× escape hatch.
    - ``context_window`` dynamically caps the hard max so that
      ``input + overhead + safety + max_tokens <= context_window``.
    - ``input_safety_ratio`` / ``input_safety_floor`` add a deterministic
      cushion on top of ``prompt_overhead_tokens`` to cover tokenizer
      estimation error and variable rubric/format overhead. The pad is
      ``max(ratio × input_tokens, floor)`` — whichever is larger wins, so
      small inputs always get at least ``floor`` and large inputs scale.
    - ``hard_max`` (optional) applies an absolute token ceiling *on top* of
      the ratio-based cap (rarely needed; leave None).
    """
    input_tokens = _estimate_tokens(input_text)
    target = max(min_target, int(target_ratio * input_tokens))
    hmax = int(ratio * input_tokens)
    if hard_max is not None:
        hmax = min(hmax, hard_max)
    if context_window is not None:
        safety_pad = max(int(input_safety_ratio * input_tokens), int(input_safety_floor))
        ctx_cap = context_window - input_tokens - prompt_overhead_tokens - safety_pad
        hmax = min(hmax, ctx_cap)
    hmax = max(min_budget, hmax)
    target = min(target, hmax)  # target can't exceed hard max
    return target, hmax


def _budget_instruction(target_tokens: int, hard_max_tokens: int) -> str:
    """Budget annotation appended to the rubric. Gives the LM both:

    - a *target* length to aim for (nudges compression)
    - a *hard max* it can stretch to if merging would genuinely lose info
      (allows ~concatenation-sized outputs for cases where that's best)
    """
    target_words = max(50, int(target_tokens / 1.33))
    max_words = max(target_words, int(hard_max_tokens / 1.33))
    return (
        f"\n\nOutput length: target ~{target_words} words (compress aggressively "
        f"when content is redundant). Hard max: {max_words} words (≤{hard_max_tokens} "
        f"tokens). You may exceed the target only if merging would lose information "
        f"the rubric says to preserve — concatenation of the inputs up to the hard "
        f"max is allowed when that is genuinely the best representation."
    )


def is_placeholder(text: str) -> bool:
    """Check if text is a template placeholder instead of real content."""
    if not text or len(text) < 50:
        placeholders = ['[', ']', 'summary', 'merged', 'content', 'text here']
        text_lower = text.lower()
        return any(p in text_lower for p in placeholders)
    return False


def _call_with_budget(predictor, *, max_tokens: int, **fields):
    """Invoke a dspy predictor with an explicit per-call max_tokens cap.

    Reviewers should grep for this name to confirm that every LM call site
    inside this package honors an output-token budget. New dspy.Predict /
    dspy.ChainOfThought calls MUST go through this helper or explicitly
    pass ``config={"max_tokens": ...}``; leaving the budget implicit lets
    the LM fall back to its default (4200 on our vLLM profile), which
    overflows a 12K context window when the input is long.
    """
    return predictor(**fields, config={"max_tokens": int(max_tokens)})


# =============================================================================
# DSPy Modules
# =============================================================================

def _infer_context_window() -> Optional[int]:
    """Best-effort read of max_model_len. Preference order:
    1. MANIFESTO_CONTEXT_WINDOW env var (see pipeline_config)
    2. lm.kwargs["max_model_len"] if present
    3. Fallback attribute on the LM object
    """
    env_value = resolve_context_window()
    if env_value is not None:
        return env_value
    lm = dspy.settings.lm
    if lm is None:
        return None
    lm_kwargs = lm.kwargs if hasattr(lm, "kwargs") else {}
    for key in ("max_model_len", "context_window", "context_window_size"):
        cw = lm_kwargs.get(key)
        if cw:
            return int(cw)
    attr_cw = getattr(lm, "max_model_len", None)
    if attr_cw:
        return int(attr_cw)

    api_bases = []
    api_base = lm_kwargs.get("api_base")
    if api_base:
        api_bases.append(str(api_base))
    api_bases.extend(str(b) for b in getattr(lm, "_api_bases", []) or [])
    for base in api_bases:
        try:
            port = urlparse(base).port
            if port is None:
                continue
            from treepo._research.core.model_detection import get_context_window_from_port
            return int(get_context_window_from_port(port=int(port)))
        except Exception:  # noqa: BLE001
            continue
    return None


class ManifestoSummarizer(dspy.Module):
    """DSPy module for summarizing chunks - optimizable by DSPy.

    Two-tier output control (per user spec: allow ``ratio`` as an escape
    hatch for cases where concatenation is the best representation, but
    typically compress):

    - **Hard max** (``output_token_ratio × input``, default 2.0): the API
      ``max_tokens`` — model *can* use up to this much.
    - **Soft target** (``target_token_ratio × input``, default 0.3): what
      we ask for in the prompt — nudges the model to compress.

    Both are capped by ``context_window`` so the whole request fits.
    """

    def __init__(
        self,
        use_cot: bool = False,
        *,
        output_token_ratio: float = CONCAT_RATIO,
        target_token_ratio: float = DEFAULT_TARGET_RATIO,
        hard_max_output_tokens: Optional[int] = None,
        min_output_tokens: int = DEFAULT_MIN_BUDGET_TOKENS,
        min_target_tokens: int = DEFAULT_MIN_TARGET_TOKENS,
        context_window: Optional[int] = None,
        prompt_overhead_tokens: int = DEFAULT_PROMPT_OVERHEAD_TOKENS,
    ):
        super().__init__()
        self.output_token_ratio = float(output_token_ratio)
        self.target_token_ratio = float(target_token_ratio)
        self.hard_max_output_tokens = hard_max_output_tokens
        self.min_output_tokens = int(min_output_tokens)
        self.min_target_tokens = int(min_target_tokens)
        self.context_window = context_window
        self.prompt_overhead_tokens = int(prompt_overhead_tokens)
        if use_cot:
            self.summarize = dspy.ChainOfThought(RILESummarize)
        else:
            self.summarize = dspy.Predict(RILESummarize)

    def forward(self, text: str, rubric: str = RILE_PRESERVATION_RUBRIC) -> str:
        ctx = self.context_window if self.context_window is not None else _infer_context_window()
        target, hmax = compute_output_budget(
            text,
            ratio=self.output_token_ratio,
            target_ratio=self.target_token_ratio,
            hard_max=self.hard_max_output_tokens,
            min_budget=self.min_output_tokens,
            min_target=self.min_target_tokens,
            context_window=ctx,
            prompt_overhead_tokens=self.prompt_overhead_tokens,
        )
        effective_rubric = rubric + _budget_instruction(target, hmax)
        cfg = {"max_tokens": int(hmax)}
        result = self.summarize(rubric=effective_rubric, text=text, config=cfg)
        summary = result.summary

        if is_placeholder(summary):
            logger.warning(f"Got placeholder summary: {summary[:50]}... Retrying...")
            result = self.summarize(rubric=effective_rubric, text=text, config=cfg)
            summary = result.summary
            if is_placeholder(summary):
                # Raise error instead of truncated fallback - truncation corrupts data
                raise ValueError(f"Failed to generate valid summary after retry. Got placeholder: {summary}")

        return summary


class UnifiedManifestoG(dspy.Module):
    """Unified manifesto summarizer ``g(content, rubric) -> summary``.

    This is the active manifesto tree-reduction module. Raw leaves call this
    module directly on chunk text; internal nodes call the same module on
    ``format_merge_input(left_summary, right_summary)``. The class intentionally
    reuses the existing two-tier output-budget logic from ``ManifestoSummarizer``.
    """

    def __init__(
        self,
        use_cot: bool = False,
        *,
        output_token_ratio: float = CONCAT_RATIO,
        target_token_ratio: float = DEFAULT_TARGET_RATIO,
        hard_max_output_tokens: Optional[int] = None,
        min_output_tokens: int = DEFAULT_MIN_BUDGET_TOKENS,
        min_target_tokens: int = DEFAULT_MIN_TARGET_TOKENS,
        context_window: Optional[int] = None,
        prompt_overhead_tokens: int = DEFAULT_PROMPT_OVERHEAD_TOKENS,
    ):
        super().__init__()
        self.output_token_ratio = float(output_token_ratio)
        self.target_token_ratio = float(target_token_ratio)
        self.hard_max_output_tokens = hard_max_output_tokens
        self.min_output_tokens = int(min_output_tokens)
        self.min_target_tokens = int(min_target_tokens)
        self.context_window = context_window
        self.prompt_overhead_tokens = int(prompt_overhead_tokens)
        if use_cot:
            self.summarize = dspy.ChainOfThought(RILESummarize)
        else:
            self.summarize = dspy.Predict(RILESummarize)

    def forward(self, content: str, rubric: str = RILE_PRESERVATION_RUBRIC) -> str:
        ctx = self.context_window if self.context_window is not None else _infer_context_window()
        target, hmax = compute_output_budget(
            content,
            ratio=self.output_token_ratio,
            target_ratio=self.target_token_ratio,
            hard_max=self.hard_max_output_tokens,
            min_budget=self.min_output_tokens,
            min_target=self.min_target_tokens,
            context_window=ctx,
            prompt_overhead_tokens=self.prompt_overhead_tokens,
        )
        effective_rubric = rubric + _budget_instruction(target, hmax)
        result = self.summarize(
            rubric=effective_rubric,
            text=content,
            config={"max_tokens": int(hmax)},
        )
        summary = result.summary

        if is_placeholder(summary):
            logger.warning(f"Got placeholder unified-g summary: {summary[:50]}... Retrying...")
            result = self.summarize(
                rubric=effective_rubric,
                text=content,
                config={"max_tokens": int(hmax)},
            )
            summary = result.summary
            if is_placeholder(summary):
                raise ValueError(f"Failed to generate valid unified-g summary after retry. Got placeholder: {summary}")

        return summary


class ManifestoMerger(dspy.Module):
    """DSPy module for merging summaries - optimizable by DSPy.

    Two-tier output control (see ``ManifestoSummarizer``). Inputs here are
    two summaries; the hard ceiling is ``ratio × (tokens(s1) + tokens(s2))``
    so the merger can concatenate the inputs if that's genuinely the best
    answer. The soft target nudges compression.
    """

    def __init__(
        self,
        use_cot: bool = False,
        *,
        output_token_ratio: float = CONCAT_RATIO,
        target_token_ratio: float = DEFAULT_TARGET_RATIO,
        hard_max_output_tokens: Optional[int] = None,
        min_output_tokens: int = DEFAULT_MIN_BUDGET_TOKENS,
        min_target_tokens: int = DEFAULT_MIN_TARGET_TOKENS,
        context_window: Optional[int] = None,
        prompt_overhead_tokens: int = DEFAULT_PROMPT_OVERHEAD_TOKENS,
    ):
        super().__init__()
        self.output_token_ratio = float(output_token_ratio)
        self.target_token_ratio = float(target_token_ratio)
        self.hard_max_output_tokens = hard_max_output_tokens
        self.min_output_tokens = int(min_output_tokens)
        self.min_target_tokens = int(min_target_tokens)
        self.context_window = context_window
        self.prompt_overhead_tokens = int(prompt_overhead_tokens)
        if use_cot:
            self.merge = dspy.ChainOfThought(RILEMerge)
        else:
            self.merge = dspy.Predict(RILEMerge)

    def forward(self, summary1: str, summary2: str, rubric: str = RILE_PRESERVATION_RUBRIC) -> str:
        ctx = self.context_window if self.context_window is not None else _infer_context_window()
        target, hmax = compute_output_budget(
            summary1 + summary2,
            ratio=self.output_token_ratio,
            target_ratio=self.target_token_ratio,
            hard_max=self.hard_max_output_tokens,
            min_budget=self.min_output_tokens,
            min_target=self.min_target_tokens,
            context_window=ctx,
            prompt_overhead_tokens=self.prompt_overhead_tokens,
        )
        effective_rubric = rubric + _budget_instruction(target, hmax)
        cfg = {"max_tokens": int(hmax)}
        result = self.merge(rubric=effective_rubric, summary1=summary1, summary2=summary2, config=cfg)
        merged = result.merged_summary

        if is_placeholder(merged):
            logger.warning(f"Got placeholder merge: {merged[:50]}... Retrying...")
            result = self.merge(rubric=effective_rubric, summary1=summary1, summary2=summary2, config=cfg)
            merged = result.merged_summary
            if is_placeholder(merged):
                merged = f"{summary1}\n\n{summary2}"

        return merged


class ManifestoScorer(dspy.Module):
    """DSPy module for scoring - optimizable by DSPy."""

    def __init__(self, use_cot: bool = False, *, max_output_tokens: Optional[int] = None):
        super().__init__()
        from .pipeline_config import DEFAULT_SCORER_MAX_TOKENS
        self.max_output_tokens = (
            int(max_output_tokens) if max_output_tokens is not None
            else DEFAULT_SCORER_MAX_TOKENS
        )
        if use_cot:
            self.score = dspy.ChainOfThought(RILEScoreSignature)
        else:
            self.score = dspy.Predict(RILEScoreSignature)

    def forward(self, summary: str, task_context: str = RILE_TASK_CONTEXT) -> dict:
        result = _call_with_budget(
            self.score,
            max_tokens=self.max_output_tokens,
            task_context=task_context,
            summary=summary,
        )
        raw_score = parse_numeric_score(
            str(getattr(result, "score", "")),
            min_value=RILE_MIN,
            max_value=RILE_MAX,
            allow_llm_fallback=True,
        )
        if raw_score is None:
            raw_score = parse_numeric_score(
                str(result),
                min_value=RILE_MIN,
                max_value=RILE_MAX,
                allow_llm_fallback=True,
            )
        if raw_score is None:
            raw_score = 0.0
        raw_score = max(RILE_MIN, min(RILE_MAX, raw_score))
        normalized = (raw_score - RILE_MIN) / (RILE_MAX - RILE_MIN)
        normalized = max(0.0, min(1.0, normalized))
        return {
            'score': normalized,
            'reasoning': result.reasoning
        }


# =============================================================================
# Strategy-Compatible Wrappers
# =============================================================================

class StrategyCompatibleSummarizer(dspy.Module):
    """
    DSPy summarizer compatible with DSPyStrategy parameter names.

    Wraps UnifiedManifestoG and keeps the canonical parameter names:
    - content -> content
    - rubric -> rubric
    """

    def __init__(self, use_cot: bool = False):
        super().__init__()
        self._inner = UnifiedManifestoG(use_cot=use_cot)

    def forward(self, content: str, rubric: str = RILE_PRESERVATION_RUBRIC) -> str:
        """Forward with DSPyStrategy-compatible parameter names."""
        return self._inner(content=content, rubric=rubric)


class StrategyCompatibleMerger(dspy.Module):
    """
    DSPy merger compatible with DSPyStrategy parameter names.

    Legacy compatibility wrapper. It now delegates to UnifiedManifestoG with
    formatted merge input instead of instantiating ManifestoMerger:
    - left_summary -> summary1
    - right_summary -> summary2
    - rubric -> rubric (unchanged)
    """

    def __init__(self, use_cot: bool = False):
        super().__init__()
        self._inner = UnifiedManifestoG(use_cot=use_cot)

    def forward(self, left_summary: str, right_summary: str, rubric: str = RILE_PRESERVATION_RUBRIC) -> str:
        """Forward with DSPyStrategy-compatible parameter names."""
        return self._inner(content=format_merge_input(left_summary, right_summary), rubric=rubric)


# =============================================================================
# Full Pipelines
# =============================================================================

class ManifestoPipeline(dspy.Module):
    """
    Full DSPy pipeline: chunk -> summarize -> merge -> score.
    The entire pipeline is optimizable by DSPy.
    Uses parallel processing for chunk summarization and merging.
    """

    def __init__(
        self,
        chunk_size: int = 2000,
        *,
        chunk_tokens: Optional[int] = None,
        use_cot: bool = False,
    ):
        super().__init__()
        self.chunk_size = chunk_size
        self.chunk_tokens = None if chunk_tokens is None else int(chunk_tokens)
        self.g = UnifiedManifestoG(use_cot=use_cot)
        self.scorer = ManifestoScorer(use_cot=use_cot)

    def forward(self, text: str, rubric: str = RILE_PRESERVATION_RUBRIC,
                task_context: str = RILE_TASK_CONTEXT) -> dspy.Prediction:
        """Process a full manifesto through the pipeline with parallel execution."""
        from treepo._research.preprocessing.chunker import chunk_for_ops
        from concurrent.futures import ThreadPoolExecutor, as_completed

        chunks = chunk_for_ops(
            text,
            max_chars=self.chunk_size,
            max_tokens=self.chunk_tokens,
            strategy="axis",
        )

        if not chunks:
            return dspy.Prediction(
                score=0.5,
                reasoning="No text to process",
                final_summary="",
            )

        def summarize_chunk(chunk_text):
            return self.g(content=chunk_text, rubric=rubric)

        summaries = [None] * len(chunks)
        with ThreadPoolExecutor(max_workers=len(chunks)) as executor:
            future_to_idx = {
                executor.submit(summarize_chunk, chunk.text): i
                for i, chunk in enumerate(chunks)
            }
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    summaries[idx] = future.result()
                except Exception as e:
                    # Re-raise instead of truncated fallback - truncation corrupts data
                    logger.error(f"Chunk {idx} summarization failed: {e}")
                    raise

        while len(summaries) > 1:
            pairs = []
            odd_summary = None
            for i in range(0, len(summaries), 2):
                if i + 1 < len(summaries):
                    pairs.append((summaries[i], summaries[i+1]))
                else:
                    odd_summary = summaries[i]

            next_level = [None] * len(pairs)
            if pairs:
                def merge_pair(s1, s2):
                    return self.g(content=format_merge_input(s1, s2), rubric=rubric)

                with ThreadPoolExecutor(max_workers=len(pairs)) as executor:
                    future_to_idx = {
                        executor.submit(merge_pair, s1, s2): i
                        for i, (s1, s2) in enumerate(pairs)
                    }
                    for future in as_completed(future_to_idx):
                        idx = future_to_idx[future]
                        try:
                            next_level[idx] = future.result()
                        except Exception:
                            s1, s2 = pairs[idx]
                            next_level[idx] = f"{s1}\n\n{s2}"

            summaries = [s for s in next_level if s is not None]
            if odd_summary is not None:
                summaries.append(odd_summary)

        final_summary = summaries[0] if summaries else ""
        score_result = self.scorer(summary=final_summary, task_context=task_context)
        score_value = score_result.get("score", 0.5)

        return dspy.Prediction(
            score=score_value,
            reasoning=score_result.get("reasoning", ""),
            final_summary=final_summary
        )


class ManifestoPipelineWithStrategy(dspy.Module):
    """
    DSPy pipeline using the strategy pattern with TreeBuilder.

    This pipeline:
    1. Uses DSPyStrategy to wrap DSPy modules
    2. Uses TreeBuilder for tree construction
    3. Can optionally use TournamentStrategy for preference collection

    The modules remain optimizable by DSPy while benefiting from:
    - Unified tree-building logic
    - Optional tournament selection and preference collection
    - Cleaner separation of concerns

    Usage:
        # Basic usage
        pipeline = ManifestoPipelineWithStrategy()
        result = pipeline(text="...")

        # With tournament selection (for learning)
        pipeline = ManifestoPipelineWithStrategy(judge=genrm_judge)
        result = pipeline(text="...")
        preferences = pipeline.get_preferences()
    """

    def __init__(
        self,
        chunk_size: int = 2000,
        chunk_tokens: Optional[int] = None,
        judge=None,
        tournament_k: int = 4,
        tournament_temperature: float = 0.9,
        leaf_module: Optional[dspy.Module] = None,
        merge_module: Optional[dspy.Module] = None,
        scorer: Optional[dspy.Module] = None,
        use_cot: bool = False,
    ):
        """
        Initialize the strategy-based pipeline.

        Args:
            chunk_size: Maximum chunk size for text splitting
            judge: Optional GenRMJudge or GenRMComparisonModule for tournament selection
            tournament_k: Number of candidates for tournament (default 4)
            tournament_temperature: Temperature for candidate generation (default 0.9)
            leaf_module: Optional leaf summarizer module (content, rubric -> summary)
            merge_module: Optional legacy split merge module. Defaults to None so
                DSPyStrategy uses unified g for both leaves and internal nodes.
            scorer: Optional scorer module (summary, task_context -> dict/prediction)
        """
        super().__init__()
        self.chunk_size = chunk_size
        self.chunk_tokens = None if chunk_tokens is None else int(chunk_tokens)

        self.leaf_module = leaf_module or StrategyCompatibleSummarizer(use_cot=use_cot)
        self.merge_module = merge_module
        self.scorer = scorer or ManifestoScorer(use_cot=use_cot)

        self._judge = judge
        self._tournament_k = tournament_k
        self._tournament_temperature = tournament_temperature

        self._last_strategy = None
        self._last_builder = None

    def _create_strategy(self):
        """Create the strategy stack (called per forward pass)."""
        # Lazy import to avoid circular dependency
        from treepo._research.core.strategy import DSPyStrategy, TournamentStrategy, TournamentConfig

        dspy_temperature = 0.7
        dspy_max_tokens = None
        try:
            current_lm = dspy.settings.lm
            if current_lm is not None:
                if hasattr(current_lm, "temperature"):
                    dspy_temperature = float(getattr(current_lm, "temperature") or dspy_temperature)
                elif hasattr(current_lm, "kwargs"):
                    dspy_temperature = float(getattr(current_lm, "kwargs", {}).get("temperature", dspy_temperature))
                if hasattr(current_lm, "max_tokens"):
                    dspy_max_tokens = getattr(current_lm, "max_tokens", None)
        except Exception:
            pass

        base_strategy = DSPyStrategy(
            leaf_module=self.leaf_module,
            merge_module=self.merge_module,
            unified_mode=self.merge_module is None,
            default_temperature=float(dspy_temperature),
            max_tokens=None if dspy_max_tokens is None else int(dspy_max_tokens),
        )

        if self._judge is not None:
            config = TournamentConfig(
                k=self._tournament_k,
                temperature=self._tournament_temperature,
            )
            return TournamentStrategy(base=base_strategy, judge=self._judge, config=config)
        else:
            return base_strategy

    def forward(
        self,
        text: str,
        rubric: str = RILE_PRESERVATION_RUBRIC,
        task_context: str = RILE_TASK_CONTEXT,
    ) -> dspy.Prediction:
        """
        Process a manifesto through the strategy-based pipeline.

        Args:
            text: Manifesto text to process
            rubric: Information preservation criteria
            task_context: Task context for RILE scoring

        Returns:
            dspy.Prediction with score, reasoning, final_summary
        """
        if not text or len(text.strip()) == 0:
            return dspy.Prediction(
                score=0.5,
                reasoning="No text to process",
                final_summary=""
            )

        strategy = self._create_strategy()
        self._last_strategy = strategy

        config = BuildConfig(
            max_chunk_chars=self.chunk_size,
            max_chunk_tokens=self.chunk_tokens,
        )
        builder = TreeBuilder(strategy=strategy, config=config)
        self._last_builder = builder

        try:
            result = builder.build_sync(text, rubric)
            final_summary = result.tree.root.summary
        except Exception as e:
            # Re-raise instead of truncated fallback - truncation corrupts data
            logger.error(f"Tree building failed: {e}")
            raise

        try:
            score_result = self.scorer(summary=final_summary, task_context=task_context)
        except TypeError:
            score_result = self.scorer(text=final_summary, task_context=task_context)

        score_value = None
        reasoning = ""
        if isinstance(score_result, dict):
            if "score" in score_result:
                score_value = score_result.get("score")
            reasoning = score_result.get("reasoning", "") or ""
        else:
            if hasattr(score_result, "score"):
                score_value = getattr(score_result, "score")
            reasoning = getattr(score_result, "reasoning", "") or ""

        if score_value is None:
            score_value = 0.5

        return dspy.Prediction(
            score=float(score_value),
            reasoning=reasoning,
            final_summary=final_summary
        )

    def get_preferences(self):
        """Get collected preferences from tournament selection."""
        if self._last_strategy is not None and hasattr(self._last_strategy, 'get_preferences'):
            return self._last_strategy.get_preferences()
        return []

    def reset_preferences(self):
        """Reset collected preferences between documents."""
        if self._last_strategy is not None and hasattr(self._last_strategy, 'reset_preferences'):
            self._last_strategy.reset_preferences()


# =============================================================================
# Training Helpers
# =============================================================================

def create_training_examples(samples: list) -> list:
    """Create DSPy training examples from samples with ground truth."""
    examples = []
    for sample in samples:
        normalized_score = (sample.rile - RILE_MIN) / (RILE_MAX - RILE_MIN)
        normalized_score = max(0.0, min(1.0, normalized_score))
        example = dspy.Example(
            text=sample.text,
            rubric=RILE_PRESERVATION_RUBRIC,
            task_context=RILE_TASK_CONTEXT,
            score=normalized_score,
        ).with_inputs('text', 'rubric', 'task_context')
        examples.append(example)
    return examples


def rile_metric(example, prediction, trace=None) -> float:
    """
    DSPy metric: how close is prediction to ground truth RILE?
    Returns 1.0 for perfect, 0.0 for >=1.0 normalized difference.
    """
    try:
        pred_value = getattr(prediction, "score", None)
        true_value = getattr(example, "score", None)
        if pred_value is None or true_value is None:
            return 0.0
        pred_score = float(pred_value)
        true_score = float(true_value)
        error = abs(pred_score - true_score)
        return max(0.0, 1.0 - error)
    except (ValueError, TypeError, AttributeError):
        return 0.0

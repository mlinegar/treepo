"""
Context window management for LLM token allocation.

This module provides centralized management of context window allocation,
ensuring that input + output tokens never exceed the model's context limit.
Token budgets are defined as percentages of the context window, making them
portable across models with different context sizes.
"""

from dataclasses import dataclass
from typing import Optional
import logging

logger = logging.getLogger(__name__)


# Default allocation percentages
DEFAULT_INPUT_FRACTION = 0.60      # 60% for input
DEFAULT_OUTPUT_FRACTION = 0.35    # 35% for output
DEFAULT_SAFETY_MARGIN = 0.05      # 5% buffer

# Minimum output tokens (even for very long inputs)
MIN_OUTPUT_TOKENS = 512

# Default context window if detection fails
DEFAULT_CONTEXT_WINDOW = 32768


@dataclass
class ContextWindowManager:
    """
    Manages token allocation based on model context window.

    Uses percentage-based allocation to ensure input + output never exceeds
    the model's context window. All values are derived from the context_window
    and allocation fractions.

    Example:
        >>> manager = ContextWindowManager(context_window=32768)
        >>> manager.max_input_tokens
        19660
        >>> manager.max_output_tokens
        11468
        >>> manager.get_safe_max_tokens(input_tokens=17808)
        11468  # Fits within remaining space

    For task-specific allocations:
        >>> scorer_manager = ContextWindowManager(
        ...     context_window=32768,
        ...     input_fraction=0.85,
        ...     output_fraction=0.10
        ... )
    """

    context_window: int
    input_fraction: float = DEFAULT_INPUT_FRACTION
    output_fraction: float = DEFAULT_OUTPUT_FRACTION
    safety_margin: float = DEFAULT_SAFETY_MARGIN

    def __post_init__(self):
        """Validate that fractions sum to <= 1.0."""
        total = self.input_fraction + self.output_fraction + self.safety_margin
        if total > 1.0:
            raise ValueError(
                f"Allocation fractions must sum to <= 1.0, got {total:.2f} "
                f"(input={self.input_fraction}, output={self.output_fraction}, "
                f"safety={self.safety_margin})"
            )

    @property
    def max_input_tokens(self) -> int:
        """Maximum tokens allowed for input (prompts, content)."""
        return int(self.context_window * self.input_fraction)

    @property
    def max_output_tokens(self) -> int:
        """Maximum tokens allowed for output (generation)."""
        return int(self.context_window * self.output_fraction)

    @property
    def safety_tokens(self) -> int:
        """Tokens reserved as safety buffer."""
        return int(self.context_window * self.safety_margin)

    def get_safe_max_tokens(self, input_tokens: int) -> int:
        """
        Get safe max_tokens for generation given actual input size.

        This is the key method for preventing context overflow. It calculates
        how many tokens are available for output after accounting for input
        and safety margin, then caps at max_output_tokens.

        Args:
            input_tokens: Actual number of input tokens in the request

        Returns:
            Safe max_tokens value that won't exceed context window
        """
        available = self.context_window - input_tokens - self.safety_tokens
        safe_tokens = max(MIN_OUTPUT_TOKENS, min(available, self.max_output_tokens))

        if available < MIN_OUTPUT_TOKENS:
            logger.warning(
                f"Input tokens ({input_tokens}) leaves only {available} for output. "
                f"Using minimum {MIN_OUTPUT_TOKENS}. May exceed context window!"
            )

        return safe_tokens

    def get_chunk_size(self, reserved_for_output: Optional[int] = None) -> int:
        """
        Get safe chunk size for document processing.

        Calculates maximum chunk size that leaves room for output generation.

        Args:
            reserved_for_output: Tokens to reserve for output.
                               Defaults to max_output_tokens.

        Returns:
            Maximum tokens per chunk
        """
        output_reserve = reserved_for_output or self.max_output_tokens
        return self.context_window - output_reserve - self.safety_tokens

    def would_fit(self, input_tokens: int, output_tokens: int) -> bool:
        """
        Check if a request would fit within the context window.

        Args:
            input_tokens: Number of input tokens
            output_tokens: Requested max_tokens for output

        Returns:
            True if request fits within context window
        """
        total = input_tokens + output_tokens + self.safety_tokens
        return total <= self.context_window

    def __repr__(self) -> str:
        return (
            f"ContextWindowManager(context={self.context_window}, "
            f"max_input={self.max_input_tokens}, max_output={self.max_output_tokens})"
        )


def create_manager_for_task(
    context_window: int,
    task: str = "default"
) -> ContextWindowManager:
    """
    Create a ContextWindowManager with task-appropriate allocations.

    Different tasks have different input/output ratio needs:
    - Summarizer: Lots of input (document), moderate output (summary)
    - Scorer: Lots of input (document + rubric), minimal output (score)
    - Default: Balanced allocation

    Args:
        context_window: Model's context window size
        task: Task name ("summarizer", "scorer", "default")

    Returns:
        Configured ContextWindowManager
    """
    task_allocations = {
        "summarizer": {
            "input_fraction": 0.70,
            "output_fraction": 0.25,
            "safety_margin": 0.05,
        },
        "scorer": {
            # Scorers need room for chain-of-thought reasoning output
            "input_fraction": 0.70,
            "output_fraction": 0.25,
            "safety_margin": 0.05,
        },
        "default": {
            "input_fraction": DEFAULT_INPUT_FRACTION,
            "output_fraction": DEFAULT_OUTPUT_FRACTION,
            "safety_margin": DEFAULT_SAFETY_MARGIN,
        },
    }

    allocations = task_allocations.get(task, task_allocations["default"])

    return ContextWindowManager(
        context_window=context_window,
        **allocations
    )

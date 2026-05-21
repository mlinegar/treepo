"""Single source of truth for manifesto-pipeline tunables.

All LLM-output budget ratios, worker counts, and context-window defaults
live here so downstream modules and scripts don't drift.

**Design principle:** keep the required surface small. Most pipelines
should need only MANIFESTO_CONTEXT_WINDOW as an env var and the three
`--*-workers` args. Everything else defaults to values that hold across
Gemma-4-31B-NVFP4 on 4× A100 (our current rig) and don't need tuning per
script.

**2× concatenation invariant (repo-wide):** `CONCAT_RATIO` mirrors
`FNO_HEAD_MIN_RATIO` in
``parallel/unified_g_v1/src/unified_g_v1/core/tensor_program.py``. Both
guarantee that a merger can carry both leaves through unchanged if merging
would lose information. Different spaces (token count here, state-dim
there), same principle. If you change one, change the other.
"""

from __future__ import annotations

import os


# --------------------------------------------------------------------------
# Output-length budget (applies to summarizer + merger LLM modules)
# --------------------------------------------------------------------------

# Hard upper bound on API max_tokens (per-call ratio cap): output ≤ ratio × input.
# Must match ``FNO_HEAD_MIN_RATIO`` in unified_g_v1/core/tensor_program.py.
CONCAT_RATIO: float = 2.0

# Soft target length communicated via the prompt. 0.15 lands at ~400 words
# for a 24K-char per-dim chunk (matching Benoit's 300-400 word spec on full
# manifestos, rather than per-chunk), and shrinks the scorer's prompt from
# ~5K to ~3.5K tokens once the tree merges up. Previously 0.3 — dropped
# after the 2026-04-21 throughput diagnosis showed the fleet is
# throughput-bound at server capacity, so shorter prompts = more KV
# headroom = slightly more aggregate parallelism.
DEFAULT_TARGET_RATIO: float = 0.15

# Assumed prompt overhead (rubric + formatting + reasoning field headers)
# when capping output for context-window safety.
DEFAULT_PROMPT_OVERHEAD_TOKENS: int = 1500

# Safety pad layered on top of the fixed prompt_overhead estimate. Our
# char/token estimator is rough (fixed _CHARS_PER_TOKEN ratio) and real
# prompts also carry a variable rubric + DSPy adapter overhead, so a
# deterministic cushion matters. We take the MAX of a 20% multiplicative
# factor on the estimated input token count and a 500-token absolute
# floor — whichever is larger "wins" so small inputs still get a real
# pad and large inputs scale up proportionally. Prevents off-by-one-ish
# ContextWindowExceededError at long-chunk summarize calls.
DEFAULT_INPUT_SAFETY_RATIO: float = 0.2   # 1.2x effective input cap
DEFAULT_INPUT_SAFETY_FLOOR: int = 500     # absolute lower bound, tokens

# Floor values so tiny inputs still get usable budgets.
DEFAULT_MIN_TARGET_TOKENS: int = 128
DEFAULT_MIN_BUDGET_TOKENS: int = 256


# --------------------------------------------------------------------------
# Context window
# --------------------------------------------------------------------------

# vLLM's max_model_len for the active profile. Read at call time via env
# var; ScriptLauncher should export this to match the running server.
DEFAULT_CONTEXT_WINDOW_ENV: str = "MANIFESTO_CONTEXT_WINDOW"


def resolve_context_window() -> int | None:
    """Read the active context window from env. Returns None if unset."""
    v = os.environ.get(DEFAULT_CONTEXT_WINDOW_ENV)
    if not v:
        return None
    try:
        return int(v)
    except ValueError:
        return None


# --------------------------------------------------------------------------
# Parallelism defaults
# --------------------------------------------------------------------------
#
# Targets the ~50-100 concurrent sweet spot for a single TP=4 Gemma-4-31B
# server. At these defaults, one pipeline process with 4 workers × 4 leaves
# × 4 scoring produces ~20 concurrent requests, which leaves headroom for
# multiple pipelines running in parallel without hitting vLLM preemption.

DEFAULT_MANIFESTO_WORKERS: int = 4
DEFAULT_SUMMARY_WORKERS: int = 4
DEFAULT_SCORING_WORKERS: int = 4


# --------------------------------------------------------------------------
# "reasoning + label" style predictor caps
# --------------------------------------------------------------------------
#
# Applies to any dspy.Predict whose OutputField pattern is short reasoning
# plus a structured label/score (DimensionScorer, JointDimensionScorer,
# ManifestoScorer, RilePreservationComparator, future scorers). 256 tokens
# covers 50-150 tokens of reasoning + the integer label with headroom;
# actual observed scorer outputs rarely exceed 200 tokens. A tight cap
# matters at server saturation — decode time scales linearly in the cap
# regardless of the model's true output length, so dropping 768→256 cuts
# per-call latency by ~3× under heavy concurrency.
DEFAULT_SCORER_MAX_TOKENS: int = 256

# Drift-explanation comparators emit slightly longer prose (they explain
# what changed between two summaries). 512 tokens is a middle ground;
# separated so callers can tune independently later.
DEFAULT_COMPARATOR_MAX_TOKENS: int = 512

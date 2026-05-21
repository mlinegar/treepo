"""
Centralized constants for the OPS framework.

This module contains all magic numbers, thresholds, and configuration defaults
that were previously scattered across the codebase. Use these constants to
ensure consistency and make values easy to find and update.

IMPORTANT: Content should NEVER be truncated. Only log output may be truncated.
"""

# =============================================================================
# LOGGING TRUNCATION
# =============================================================================
# This is the ONLY acceptable truncation - for log output display.
# Content (document text, summaries, etc.) should NEVER be truncated.
LOG_TRUNCATE_LENGTH = 500


# =============================================================================
# DEFAULT PORTS
# =============================================================================
DEFAULT_TASK_MODEL_PORT = 8000
DEFAULT_JUDGE_MODEL_PORT = 8001
DEFAULT_SGLANG_PORT = 30000


# =============================================================================
# TOKEN LIMITS
# =============================================================================
# These are defaults - they should be overridable via configuration.
DEFAULT_MAX_TOKENS = 16384
DEFAULT_CHUNK_TOKENS = 2000
DEFAULT_MAX_INIT_PROMPT_TOKENS = 4000

# =============================================================================
# CONTEXT WINDOW ALLOCATION
# =============================================================================
# Percentages of context window for input/output/safety margin.
# These ensure input + output never exceeds the model's context window.
DEFAULT_CONTEXT_WINDOW = 32768
DEFAULT_INPUT_FRACTION = 0.60       # 60% for input (prompts, content)
DEFAULT_OUTPUT_FRACTION = 0.35      # 35% for output (generation)
DEFAULT_SAFETY_MARGIN = 0.05        # 5% buffer for safety


# =============================================================================
# TIMEOUTS (in seconds)
# =============================================================================
HEALTH_CHECK_TIMEOUT = 5
REQUEST_TIMEOUT = 600  # 10 minutes for large models (e.g., 235B GenRM)
BATCH_POLL_INTERVAL = 0.1  # For async batch processing


# =============================================================================
# ERROR THRESHOLDS
# =============================================================================
# Used for bootstrap loops and training data weighting
ERROR_THRESHOLD_HIGH = 30.0
ERROR_THRESHOLD_LOW = 10.0

# Bootstrap loop targets
BOOTSTRAP_TARGET_P_SUFF = 0.05
BOOTSTRAP_TARGET_P_MERGE = 0.05
BOOTSTRAP_TARGET_P_IDEM = 0.10
BOOTSTRAP_CONVERGENCE_THRESHOLD = 0.01
BOOTSTRAP_SAMPLE_RATE = 0.10
BOOTSTRAP_MAX_TRAINING_EXAMPLES = 100


# =============================================================================
# CONCURRENCY DEFAULTS
# =============================================================================
DEFAULT_MAX_CONCURRENT_REQUESTS = 20
DEFAULT_MAX_CONCURRENT_DOCUMENTS = 50
AUDIT_MAX_WORKERS = 32
HIGH_THROUGHPUT_MAX_REQUESTS = 200
PRECACHE_MAX_WORKERS = 512
DEFAULT_BATCH_SIZE = 200


# =============================================================================
# SAMPLING PARAMETERS
# =============================================================================
DEFAULT_TEMPERATURE = 0.7
DEFAULT_TOP_P = 0.95
DIVERSE_TEMPERATURES = [0.3, 0.5, 0.7, 0.9]


# =============================================================================
# VALIDATION THRESHOLDS
# =============================================================================
MIN_SUMMARY_WORDS = 10
MIN_TRAINING_EXAMPLES = 10
MIN_PREFERENCE_EXAMPLES = 20
DOCUMENT_LABEL_CONFIDENCE = 0.75


# =============================================================================
# CHUNK SIZE LIMITS
# =============================================================================
MAX_CHUNK_CHARS = 2000
MIN_CHUNK_CHARS = 100
DEFAULT_MAX_DOC_CHARS = 4000  # For initialization
EXTENDED_MAX_DOC_CHARS = 8000  # For extended processing

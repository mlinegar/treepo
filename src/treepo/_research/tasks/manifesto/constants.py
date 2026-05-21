"""
RILE scoring constants for the Manifesto Project.

This module defines the fixed bounds for RILE scores.
RILE (Right-Left) is the Manifesto Project's measure of political positioning.

Usage:
    from treepo._research.tasks.manifesto import RILE_MIN, RILE_MAX, RILE_RANGE

    # Use RILE_RANGE with normalize_error_to_score
    from treepo._research.core.scoring import normalize_error_to_score
    score = normalize_error_to_score(error, max_error=RILE_RANGE)
"""

# =============================================================================
# RILE Score Bounds (Fixed by Manifesto Project definition)
# =============================================================================

RILE_MIN: float = -100.0  # Far left
RILE_MAX: float = 100.0   # Far right
RILE_RANGE: float = RILE_MAX - RILE_MIN  # 200.0 (full scale)

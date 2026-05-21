"""Backward-compatible re-export of synthetic data utilities.

Some entrypoints still import ``src.training.synthetic_data``.
The canonical implementation now lives in ``src.training.synthetic``.
"""

from treepo._research.training.synthetic import *  # noqa: F401,F403

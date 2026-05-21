"""Path setup for treepo.cld tests.

Adds the workspace root to ``sys.path`` so ``from treepo._research.ctreepo...`` resolves
during ``pytest treepo.cld/`` runs (same pattern as ``tests/conftest.py``).
"""

from __future__ import annotations

import sys
from pathlib import Path

_WORKSPACE_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(_WORKSPACE_ROOT))

_TREEPO_CLD_SRC = _WORKSPACE_ROOT / "treepo.cld" / "src"
if str(_TREEPO_CLD_SRC) not in sys.path:
    sys.path.insert(0, str(_TREEPO_CLD_SRC))

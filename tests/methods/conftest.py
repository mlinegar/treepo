"""Path setup for treepo.methods tests."""

from __future__ import annotations

import sys
from pathlib import Path

_WORKSPACE_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(_WORKSPACE_ROOT))

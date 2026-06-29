"""Project-wide test path setup."""

from __future__ import annotations

import sys
from pathlib import Path

_TREEPO_SRC = Path(__file__).resolve().parent.parent / "src"
if _TREEPO_SRC.is_dir() and str(_TREEPO_SRC) not in sys.path:
    sys.path.insert(0, str(_TREEPO_SRC))

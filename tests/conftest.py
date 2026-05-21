"""Project-wide test path setup.

Inserts ``src/`` at the front of ``sys.path`` so ``from treepo.X import Y``
always resolves to the in-tree package, not whatever may be installed in
site-packages. Lets tests run cleanly without ``pip install -e .`` first.
"""

from __future__ import annotations

import sys
from pathlib import Path

_TREEPO_SRC = Path(__file__).resolve().parent.parent / "src"
if _TREEPO_SRC.is_dir() and str(_TREEPO_SRC) not in sys.path:
    sys.path.insert(0, str(_TREEPO_SRC))

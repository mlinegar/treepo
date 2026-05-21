from __future__ import annotations

# Backwards-compatible forwarder (v1): keep existing imports/tests/scripts working.
from treepo._research.ctreepo.sim.core import tensor_lda_book_benchmark as _mod

__all__ = [k for k in _mod.__dict__.keys() if not k.startswith("__")]
globals().update({k: v for k, v in _mod.__dict__.items() if not k.startswith("__")})
del _mod

"""`CardinalitySketch` adapter wrapping `datasketches.hll_sketch`.

Apache DataSketches is the canonical reference implementation used by Apache
Druid, Yahoo, and cited across the streaming literature. This adapter wraps
its `hll_sketch` + `hll_union` so we can run the exact same benchmark against
both the native (`hll_native.py`) and canonical implementations.

Requires `datasketches` (`uv sync --extra sketches`). The import is gated so
that the top-level `treepo.bench.sketches` module works without the optional dep.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable

try:
    import datasketches as _ds
except ImportError:  # pragma: no cover
    _ds = None  # type: ignore[assignment]


def _require_datasketches() -> None:
    if _ds is None:
        raise ImportError(
            "datasketches is required for HLL parity benchmarks. "
            "Install with: uv sync --extra sketches"
        )


def _hll_rse(precision: int) -> float:
    return 1.04 / math.sqrt(float(1 << int(precision)))


@dataclass(frozen=True)
class HLLDatasketchesAdapter:
    """Wraps `datasketches.hll_sketch` as a `CardinalitySketch`.

    Merge is the classical union implemented by `datasketches.hll_union`.

    Note on byte-determinism: Apache DataSketches' HLL has internal mode
    transitions (list → sparse → dense) triggered by cardinality thresholds.
    A flat `hll_sketch` that has seen few items sits in list mode; a
    union-merged result always transitions to dense. Their serialized bytes
    differ even though both faithfully represent the same multiset. We
    therefore set `is_byte_deterministic=False` and define `state_equal` as
    functional (estimate) equivalence within a tight relative tolerance. This
    is what Proposition 1 actually claims — oracle-equivalence of summaries,
    not byte-identity of representations. Byte-identity is the stronger
    property held by the native adapter.
    """

    precision: int  # datasketches `lg_config_k`

    name: str = "hll_datasketches"
    is_commutative: bool = True
    is_associative: bool = True
    is_idempotent: bool = True
    is_byte_deterministic: bool = False

    @property
    def config(self) -> dict:
        return {
            "backend": "datasketches",
            "precision": int(self.precision),
        }

    def empty(self) -> Any:
        _require_datasketches()
        return _ds.hll_sketch(int(self.precision))

    def update(self, s: Any, item: int) -> Any:
        s.update(int(item))
        return s

    def encode(self, items: Iterable[int]) -> Any:
        _require_datasketches()
        sk = _ds.hll_sketch(int(self.precision))
        for tok in items:
            sk.update(int(tok))
        return sk

    def merge(self, a: Any, b: Any) -> Any:
        _require_datasketches()
        u = _ds.hll_union(int(self.precision))
        u.update(a)
        u.update(b)
        return u.get_result()

    def query(self, s: Any, q: None = None) -> float:
        return float(s.get_estimate())

    def serialize(self, s: Any) -> bytes:
        return bytes(s.serialize_updatable())

    def serialized_size_bytes(self, s: Any) -> float:
        return float(len(self.serialize(s)))

    def state_equal(self, a: Any, b: Any) -> bool:
        if self.serialize(a) == self.serialize(b):
            return True
        est_a = float(a.get_estimate())
        est_b = float(b.get_estimate())
        scale = max(1.0, abs(est_a), abs(est_b))
        # Tolerance is two HLL relative standard errors. Internal mode
        # transitions (list→sparse→dense) introduce small representation
        # shifts; the merge remains a *valid* mergeable-summary operation so
        # estimates stay within HLL's theoretical noise of each other.
        tol = 2.0 * _hll_rse(int(self.precision))
        return abs(est_a - est_b) / scale <= tol

    def memory_bytes(self, s: Any) -> float:
        return float(s.get_updatable_serialization_bytes())

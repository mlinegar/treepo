"""Cardinality sketch adapters backed by Apache DataSketches.

These adapters deliberately keep the state object as the official
DataSketches sketch type and expose only the small TreePO protocol surface.
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
            "datasketches is required for official sketch benchmarks. "
            "Install with: uv sync --extra sketches"
        )


def _rel_close(a: float, b: float, *, tol: float) -> bool:
    scale = max(1.0, abs(float(a)), abs(float(b)))
    return abs(float(a) - float(b)) / scale <= float(tol)


@dataclass(frozen=True)
class CPCDatasketchesAdapter:
    """Apache DataSketches CPC distinct-count adapter."""

    lg_k: int = 10

    name: str = "cpc_datasketches"
    is_commutative: bool = True
    is_associative: bool = True
    is_idempotent: bool = True
    is_byte_deterministic: bool = False

    @property
    def config(self) -> dict:
        return {"backend": "datasketches", "family": "cpc", "lg_k": int(self.lg_k)}

    def empty(self) -> Any:
        _require_datasketches()
        return _ds.cpc_sketch(int(self.lg_k))

    def update(self, s: Any, item: int | str) -> Any:
        s.update(item)
        return s

    def encode(self, items: Iterable[int | str]) -> Any:
        sk = self.empty()
        for item in items:
            sk.update(item)
        return sk

    def merge(self, a: Any, b: Any) -> Any:
        _require_datasketches()
        union = _ds.cpc_union(int(self.lg_k))
        union.update(a)
        union.update(b)
        return union.get_result()

    def query(self, s: Any, q: None = None) -> float:
        return float(s.get_estimate())

    def serialize(self, s: Any) -> bytes:
        return bytes(s.serialize())

    def serialized_size_bytes(self, s: Any) -> float:
        return float(len(self.serialize(s)))

    def state_equal(self, a: Any, b: Any) -> bool:
        if self.serialize(a) == self.serialize(b):
            return True
        # CPC has an internal HIP estimator and merged-state flag, so compare
        # functional cardinality rather than byte layout after unions.
        tol = 2.0 / math.sqrt(float(1 << int(self.lg_k)))
        return _rel_close(float(a.get_estimate()), float(b.get_estimate()), tol=tol)

    def memory_bytes(self, s: Any) -> float:
        return self.serialized_size_bytes(s)


@dataclass(frozen=True)
class ThetaDatasketchesAdapter:
    """Apache DataSketches Theta/KMV distinct-count adapter."""

    lg_k: int = 12

    name: str = "theta_datasketches"
    is_commutative: bool = True
    is_associative: bool = True
    is_idempotent: bool = True
    is_byte_deterministic: bool = False

    @property
    def config(self) -> dict:
        return {"backend": "datasketches", "family": "theta", "lg_k": int(self.lg_k)}

    def empty(self) -> Any:
        _require_datasketches()
        return _ds.update_theta_sketch(int(self.lg_k))

    def update(self, s: Any, item: int | str) -> Any:
        s.update(item)
        return s

    def encode(self, items: Iterable[int | str]) -> Any:
        sk = self.empty()
        for item in items:
            sk.update(item)
        return sk.compact()

    def merge(self, a: Any, b: Any) -> Any:
        _require_datasketches()
        union = _ds.theta_union(int(self.lg_k))
        union.update(a)
        union.update(b)
        return union.get_result()

    def query(self, s: Any, q: None = None) -> float:
        return float(s.get_estimate())

    def serialize(self, s: Any) -> bytes:
        return bytes(s.serialize())

    def serialized_size_bytes(self, s: Any) -> float:
        return float(len(self.serialize(s)))

    def state_equal(self, a: Any, b: Any) -> bool:
        if self.serialize(a) == self.serialize(b):
            return True
        retained = max(1.0, float(getattr(a, "num_retained", 1)), float(getattr(b, "num_retained", 1)))
        tol = 2.0 / math.sqrt(retained)
        return _rel_close(float(a.get_estimate()), float(b.get_estimate()), tol=tol)

    def memory_bytes(self, s: Any) -> float:
        return self.serialized_size_bytes(s)


def theta_union_estimate(a: Any, b: Any, *, lg_k: int = 12) -> float:
    _require_datasketches()
    union = _ds.theta_union(int(lg_k))
    union.update(a)
    union.update(b)
    return float(union.get_result().get_estimate())


def theta_intersection_estimate(a: Any, b: Any) -> float:
    _require_datasketches()
    inter = _ds.theta_intersection()
    inter.update(a)
    inter.update(b)
    return float(inter.get_result().get_estimate())


def theta_a_not_b_estimate(a: Any, b: Any) -> float:
    _require_datasketches()
    diff = _ds.theta_a_not_b()
    return float(diff.compute(a, b).get_estimate())

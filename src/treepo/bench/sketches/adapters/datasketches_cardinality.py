"""Cardinality sketch adapters backed by Apache DataSketches.

These adapters deliberately keep the state object as the official
DataSketches sketch type and expose only the small TreePO protocol surface.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable

from treepo.bench.sketches.adapters._datasketches import _ds, _require_datasketches


def _rel_close(a: float, b: float, *, tol: float) -> bool:
    scale = max(1.0, abs(float(a)), abs(float(b)))
    return abs(float(a) - float(b)) / scale <= float(tol)


class _CardinalityAdapterBase:
    """Shared union-object merge surface for CPC/Theta adapters."""

    is_commutative = True
    is_associative = True
    is_idempotent = True
    is_byte_deterministic = False

    def _new_sketch(self) -> Any:
        raise NotImplementedError

    def _new_union(self) -> Any:
        raise NotImplementedError

    def _state_tolerance(self, a: Any, b: Any) -> float:
        raise NotImplementedError

    def empty(self) -> Any:
        _require_datasketches()
        return self._new_sketch()

    def encode(self, items: Iterable[int | str]) -> Any:
        sk = self.empty()
        for item in items:
            sk.update(item)
        return sk

    def merge(self, a: Any, b: Any) -> Any:
        _require_datasketches()
        union = self._new_union()
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
        return _rel_close(
            float(a.get_estimate()),
            float(b.get_estimate()),
            tol=self._state_tolerance(a, b),
        )

    def memory_bytes(self, s: Any) -> float:
        return self.serialized_size_bytes(s)


@dataclass(frozen=True)
class CPCDatasketchesAdapter(_CardinalityAdapterBase):
    """Apache DataSketches CPC distinct-count adapter."""

    lg_k: int = 10

    name: str = "cpc_datasketches"

    @property
    def config(self) -> dict:
        return {"backend": "datasketches", "family": "cpc", "lg_k": int(self.lg_k)}

    def _new_sketch(self) -> Any:
        return _ds.cpc_sketch(int(self.lg_k))

    def _new_union(self) -> Any:
        return _ds.cpc_union(int(self.lg_k))

    def _state_tolerance(self, a: Any, b: Any) -> float:
        # CPC has an internal HIP estimator and merged-state flag, so compare
        # functional cardinality rather than byte layout after unions.
        del a, b
        return 2.0 / math.sqrt(float(1 << int(self.lg_k)))


@dataclass(frozen=True)
class ThetaDatasketchesAdapter(_CardinalityAdapterBase):
    """Apache DataSketches Theta/KMV distinct-count adapter."""

    lg_k: int = 12

    name: str = "theta_datasketches"

    @property
    def config(self) -> dict:
        return {"backend": "datasketches", "family": "theta", "lg_k": int(self.lg_k)}

    def _new_sketch(self) -> Any:
        return _ds.update_theta_sketch(int(self.lg_k))

    def _new_union(self) -> Any:
        return _ds.theta_union(int(self.lg_k))

    def encode(self, items: Iterable[int | str]) -> Any:
        sk = self.empty()
        for item in items:
            sk.update(item)
        return sk.compact()

    def _state_tolerance(self, a: Any, b: Any) -> float:
        retained = max(1.0, float(getattr(a, "num_retained", 1)), float(getattr(b, "num_retained", 1)))
        return 2.0 / math.sqrt(retained)


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

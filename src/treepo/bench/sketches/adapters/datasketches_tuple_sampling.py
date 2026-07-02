"""Tuple and sampling sketch adapters backed by Apache DataSketches."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable

from treepo.bench.sketches.adapters._datasketches import (
    _ds,
    _require_datasketches,
    _split_weighted_item,
)


@dataclass(frozen=True)
class TupleAccumulatorDatasketchesAdapter:
    """Apache DataSketches tuple sketch with accumulator summaries.

    Items may be raw keys, which receive value 1, or `(key, value)` pairs.
    The default query returns the distinct-key estimate. Query
    `"summary_sum"` returns the sum of retained accumulator summaries, which is
    exact while the sketch is not in estimation mode and useful as a compact
    tree-merge sanity check.
    """

    lg_k: int = 12

    name: str = "tuple_accumulator_datasketches"
    is_commutative: bool = True
    is_associative: bool = True
    is_idempotent: bool = False
    is_byte_deterministic: bool = True

    @property
    def config(self) -> dict:
        return {"backend": "datasketches", "family": "tuple_accumulator", "lg_k": int(self.lg_k)}

    def _policy(self) -> Any:
        _require_datasketches()
        return _ds.AccumulatorPolicy()

    def _serde(self) -> Any:
        _require_datasketches()
        return _ds.PyLongsSerDe()

    def empty(self) -> Any:
        _require_datasketches()
        return _ds.update_tuple_sketch(self._policy(), int(self.lg_k)).compact()

    def encode(self, items: Iterable[Any]) -> Any:
        _require_datasketches()
        sk = _ds.update_tuple_sketch(self._policy(), int(self.lg_k))
        for item in items:
            key, value = _split_weighted_item(item)
            sk.update(key, int(value))
        return sk.compact()

    def merge(self, a: Any, b: Any) -> Any:
        _require_datasketches()
        union = _ds.tuple_union(self._policy(), int(self.lg_k))
        union.update(a)
        union.update(b)
        return union.get_result()

    def query(self, s: Any, q: str | None = None) -> float:
        if q == "summary_sum":
            return float(sum(float(summary) for _, summary in s))
        return float(s.get_estimate())

    def serialize(self, s: Any) -> bytes:
        return bytes(s.serialize(self._serde()))

    def serialized_size_bytes(self, s: Any) -> float:
        return float(len(self.serialize(s)))

    def state_equal(self, a: Any, b: Any) -> bool:
        return self.serialize(a) == self.serialize(b)

    def memory_bytes(self, s: Any) -> float:
        return self.serialized_size_bytes(s)


@dataclass(frozen=True)
class VarOptStringsDatasketchesAdapter:
    """Apache DataSketches VarOpt weighted sampling adapter for string items."""

    k: int = 64

    name: str = "varopt_strings_datasketches"
    is_commutative: bool = True
    is_associative: bool = True
    is_idempotent: bool = False
    is_byte_deterministic: bool = False

    @property
    def config(self) -> dict:
        return {"backend": "datasketches", "family": "varopt_strings", "k": int(self.k)}

    def _serde(self) -> Any:
        _require_datasketches()
        return _ds.PyStringsSerDe()

    def empty(self) -> Any:
        _require_datasketches()
        return _ds.var_opt_sketch(int(self.k))

    def encode(self, items: Iterable[str | tuple[str, float]]) -> Any:
        sk = self.empty()
        for item in items:
            if isinstance(item, tuple) and len(item) == 2:
                key, weight = item
                sk.update(str(key), float(weight))
            else:
                sk.update(str(item))
        return sk

    def merge(self, a: Any, b: Any) -> Any:
        _require_datasketches()
        union = _ds.var_opt_union(int(self.k))
        union.update(a)
        union.update(b)
        return union.get_result()

    def query(self, s: Any, q: str | Callable[[str], bool] | None = None) -> float:
        if q == "num_samples":
            return float(s.num_samples)
        if callable(q):
            return float(s.estimate_subset_sum(q)["estimate"])
        return float(s.estimate_subset_sum(lambda _item: True)["estimate"])

    def serialize(self, s: Any) -> bytes:
        return bytes(s.serialize(self._serde()))

    def serialized_size_bytes(self, s: Any) -> float:
        return float(s.get_serialized_size_bytes(self._serde()))

    def state_equal(self, a: Any, b: Any) -> bool:
        if self.serialize(a) == self.serialize(b):
            return True
        return int(a.n) == int(b.n) and abs(self.query(a, None) - self.query(b, None)) < 1e-9

    def memory_bytes(self, s: Any) -> float:
        return self.serialized_size_bytes(s)

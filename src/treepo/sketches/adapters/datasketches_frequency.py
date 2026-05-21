"""Frequency sketch adapters backed by Apache DataSketches."""

from __future__ import annotations

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
            "Install with: pip install 'treepo[sketches]'"
        )


def _split_weighted_item(item: Any) -> tuple[Any, float | int]:
    """Accept either a bare key or a ``(key, weight)`` update."""
    if isinstance(item, tuple) and len(item) == 2:
        return item[0], item[1]
    return item, 1


@dataclass(frozen=True)
class CountMinDatasketchesAdapter:
    """Apache DataSketches Count-Min adapter for point-frequency queries."""

    num_hashes: int = 5
    num_buckets: int = 256

    name: str = "count_min_datasketches"
    is_commutative: bool = True
    is_associative: bool = True
    is_idempotent: bool = False
    is_byte_deterministic: bool = True

    @property
    def config(self) -> dict:
        return {
            "backend": "datasketches",
            "family": "count_min",
            "num_hashes": int(self.num_hashes),
            "num_buckets": int(self.num_buckets),
        }

    def empty(self) -> Any:
        _require_datasketches()
        # Keep the default DataSketches seed so deserialize() round-trips.
        return _ds.count_min_sketch(int(self.num_hashes), int(self.num_buckets))

    def update(self, s: Any, item: int | str | tuple[int | str, int | float]) -> Any:
        key, weight = _split_weighted_item(item)
        s.update(key, weight)
        return s

    def encode(self, items: Iterable[int | str | tuple[int | str, int | float]]) -> Any:
        sk = self.empty()
        for item in items:
            self.update(sk, item)
        return sk

    def _copy(self, s: Any) -> Any:
        _require_datasketches()
        return _ds.count_min_sketch.deserialize(bytes(s.serialize()))

    def merge(self, a: Any, b: Any) -> Any:
        out = self._copy(a)
        out.merge(b)
        return out

    def query(self, s: Any, q: int | str) -> float:
        return float(s.get_estimate(q))

    def serialize(self, s: Any) -> bytes:
        return bytes(s.serialize())

    def serialized_size_bytes(self, s: Any) -> float:
        return float(s.get_serialized_size_bytes())

    def state_equal(self, a: Any, b: Any) -> bool:
        return self.serialize(a) == self.serialize(b)

    def memory_bytes(self, s: Any) -> float:
        return self.serialized_size_bytes(s)


@dataclass(frozen=True)
class FrequentStringsDatasketchesAdapter:
    """Apache DataSketches frequent strings adapter."""

    lg_max_map_size: int = 8

    name: str = "frequent_strings_datasketches"
    is_commutative: bool = True
    is_associative: bool = True
    is_idempotent: bool = False
    is_byte_deterministic: bool = True

    @property
    def config(self) -> dict:
        return {
            "backend": "datasketches",
            "family": "frequent_strings",
            "lg_max_map_size": int(self.lg_max_map_size),
        }

    def empty(self) -> Any:
        _require_datasketches()
        return _ds.frequent_strings_sketch(int(self.lg_max_map_size))

    def update(self, s: Any, item: int | str | tuple[int | str, int]) -> Any:
        key, weight = _split_weighted_item(item)
        s.update(str(key), int(weight))
        return s

    def encode(self, items: Iterable[int | str | tuple[int | str, int]]) -> Any:
        sk = self.empty()
        for item in items:
            self.update(sk, item)
        return sk

    def _copy(self, s: Any) -> Any:
        _require_datasketches()
        return _ds.frequent_strings_sketch.deserialize(bytes(s.serialize()))

    def merge(self, a: Any, b: Any) -> Any:
        out = self._copy(a)
        out.merge(b)
        return out

    def query(self, s: Any, q: int | str) -> float:
        return float(s.get_estimate(str(q)))

    def serialize(self, s: Any) -> bytes:
        return bytes(s.serialize())

    def serialized_size_bytes(self, s: Any) -> float:
        return float(s.get_serialized_size_bytes())

    def state_equal(self, a: Any, b: Any) -> bool:
        return self.serialize(a) == self.serialize(b)

    def memory_bytes(self, s: Any) -> float:
        return self.serialized_size_bytes(s)

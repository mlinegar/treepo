"""Frequency sketch adapters backed by Apache DataSketches."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

from treepo.bench.sketches.adapters._datasketches import (
    _ds,
    _require_datasketches,
    _split_weighted_item,
)


class _FrequencyAdapterBase:
    """Shared deserialize-copy merge surface for the frequency adapters."""

    is_commutative = True
    is_associative = True
    is_idempotent = False
    is_byte_deterministic = True

    def _new_sketch(self) -> Any:
        raise NotImplementedError

    def _sketch_type(self) -> Any:
        raise NotImplementedError

    def _update(self, sk: Any, key: Any, weight: float | int) -> None:
        raise NotImplementedError

    def empty(self) -> Any:
        _require_datasketches()
        return self._new_sketch()

    def encode(self, items: Iterable[int | str | tuple[int | str, int | float]]) -> Any:
        sk = self.empty()
        for item in items:
            key, weight = _split_weighted_item(item)
            self._update(sk, key, weight)
        return sk

    def merge(self, a: Any, b: Any) -> Any:
        _require_datasketches()
        out = self._sketch_type().deserialize(bytes(a.serialize()))
        out.merge(b)
        return out

    def serialize(self, s: Any) -> bytes:
        return bytes(s.serialize())

    def serialized_size_bytes(self, s: Any) -> float:
        return float(s.get_serialized_size_bytes())

    def state_equal(self, a: Any, b: Any) -> bool:
        return self.serialize(a) == self.serialize(b)

    def memory_bytes(self, s: Any) -> float:
        return self.serialized_size_bytes(s)


@dataclass(frozen=True)
class CountMinDatasketchesAdapter(_FrequencyAdapterBase):
    """Apache DataSketches Count-Min adapter for point-frequency queries."""

    num_hashes: int = 5
    num_buckets: int = 256

    name: str = "count_min_datasketches"

    @property
    def config(self) -> dict:
        return {
            "backend": "datasketches",
            "family": "count_min",
            "num_hashes": int(self.num_hashes),
            "num_buckets": int(self.num_buckets),
        }

    def _new_sketch(self) -> Any:
        # Keep the default DataSketches seed so deserialize() round-trips.
        return _ds.count_min_sketch(int(self.num_hashes), int(self.num_buckets))

    def _sketch_type(self) -> Any:
        return _ds.count_min_sketch

    def _update(self, sk: Any, key: Any, weight: float | int) -> None:
        sk.update(key, weight)

    def query(self, s: Any, q: int | str) -> float:
        return float(s.get_estimate(q))


@dataclass(frozen=True)
class FrequentStringsDatasketchesAdapter(_FrequencyAdapterBase):
    """Apache DataSketches frequent strings adapter."""

    lg_max_map_size: int = 8

    name: str = "frequent_strings_datasketches"

    @property
    def config(self) -> dict:
        return {
            "backend": "datasketches",
            "family": "frequent_strings",
            "lg_max_map_size": int(self.lg_max_map_size),
        }

    def _new_sketch(self) -> Any:
        return _ds.frequent_strings_sketch(int(self.lg_max_map_size))

    def _sketch_type(self) -> Any:
        return _ds.frequent_strings_sketch

    def _update(self, sk: Any, key: Any, weight: float | int) -> None:
        sk.update(str(key), int(weight))

    def query(self, s: Any, q: int | str) -> float:
        return float(s.get_estimate(str(q)))
